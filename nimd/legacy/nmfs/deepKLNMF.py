# deep_nmf.py

import numpy as np
import time
import warnings
from scipy.special import lambertw
from numpy.linalg import norm, inv
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Literal, Optional

import nmfs.multilayerKLNMF as multilayerNMF

# import multilayerKLNMF as multilayerNMF

# A small constant for numerical stability
EPS_STABILITY = 1e-9


@dataclass
class DeepNMFParams:
    """
    Parameters for Deep KL-NMF model.

    Attributes:
        L (int): Number of layers in the deep NMF model.
        maxiter (int): Iterations for initialization with multilayer NMF.
        outerit (int): Iterations for deep NMF alternating optimization.
        display (bool): If True, prints progress during optimization.
        min_vol (bool): If True, activates min-volume regularization.
        epsi (float): Numerical tolerance for convergence (suggested range: 1e-3 to 1e-10).
        beta (float): Controls divergence type for the objective. Allowed values: [0, 0.5, 1, 1.5, 2]. Some MU updates only allow [0, 1, 1.5].
        rngseed (float): Random seed for reproducibility.
        HnormType (Literal["rows", "cols"]): Normalization type for H. "rows" finds min value in each row, "cols" in each column, for MU initialization.
        normalize (Literal[1, 2, 3, 4]): Scaling method for enforcing stochasticity constraints:
            - 1: Columns of H sum to at most 1.
            - 2: Rows of H sum to exactly 1.
            - 3: Columns of W sum to exactly 1.
            - 4: Columns of H sum to exactly 1.
        accADMM (bool): If True, uses accelerated ADMM procedure for min-vol regularization.
        maxIterADMM (int): Maximum iterations for ADMM (default: 200).
        innerloop (int): Inner loop count for step 1 of ADMM (default: 1).
        rho (int): ADMM parameter for min-vol regularization, affects Z computation (suggested range: 10-100).
        thres (float): Stopping criterion for ADMM (default: 1e-4), based on ||Wi - Zi||.
        delta (List): Used in min-vol ADMM, typically ones(1, L).
        alpha (List): Used in min-vol regularization, assigned per layer (typically zeros(1, L)).
        alpha_tilde (List): Used to compute alpha, typically 0.05 * np.ones(1, L). Example values vary by dataset.
        W0 (Optional[np.ndarray]): Initial W matrix for beta nmf init
        H0 (Optional[np.ndarray]): Initial H matrix for beta nmf init
        Wl (Optional[List]): Initial W for each layer from multilayer nmf
        Hl (Optional[List]): Initial H for each layer from multilayer nmf
        lam (Optional[List]): Weight parameter for each layer (lambda).
    """

    L: int
    maxiter: int = 500
    outerit: int = 100
    display: bool = False
    min_vol: bool = False
    epsi: float = 1e-10
    beta: float = 1
    rngseed: float = 47
    HnormType: Literal["rows", "cols"] = "rows"
    normalize: Literal[1, 2, 3, 4] = 2
    accADMM: bool = False
    maxIterADMM: int = 200
    innerloop: int = 100
    rho: int = 20
    thres: float = 1e-4
    delta: List = field(init=False)
    alpha: List = field(init=False)
    alpha_tilde: List = field(init=False)
    W0: Optional[np.ndarray] = None
    H0: Optional[np.ndarray] = None
    Wl: Optional[List] = None
    Hl: Optional[List] = None
    # TODO: create an init for lambda
    lam: Optional[List] = None

    def __post_init__(self):
        self.alpha = np.zeros(self.L)
        self.delta = np.ones(self.L)
        self.alpha_tilde = np.ones(self.L) * 0.05


# ==============================================================================
# Helper Functions for Deep NMF
# ==============================================================================


def updatemu(Phi, Eta, Psi, W, mu, epsi):
    """
    Newton-Raphson procedure to find Lagrange multipliers mu.
    Used in the min-volume deep NMF updates.
    """
    F, K = W.shape
    do_loop = True
    max_iter_mu = 1000
    k = 0
    while do_loop and k < max_iter_mu:
        mu_prev = mu.copy()

        # Broadcasting mu.T handles the repmat operation
        Phi_mu = Phi + mu.T
        sqrt_term = np.sqrt(Phi_mu**2 + Eta)

        Mat = W * (sqrt_term - Phi_mu) / (Psi + EPS_STABILITY)
        xi = (np.sum(Mat, axis=0) - 1.0).reshape(-1, 1)

        Matp = (W / (Psi + EPS_STABILITY)) * (Phi_mu / sqrt_term - 1.0)
        xip = np.sum(Matp, axis=0).reshape(-1, 1)

        mu = mu - xi / (xip + EPS_STABILITY)

        if np.max(np.abs(mu - mu_prev)) <= epsi:
            do_loop = False
        k += 1

    if k == max_iter_mu:
        warnings.warn("updatemu: Newton-Raphson reached max iterations.")
    return mu


def updatemu_hrows(C, D, H, gamma_beta, mu, epsi):
    """
    Newton-Raphson procedure to find Lagrange multipliers mu for row-wise constraints.
    """
    K, T = C.shape
    delta = 1.0
    do_loop = True
    max_iter_mu = 10000
    k = 0
    while do_loop and k < max_iter_mu:
        mu_prev = mu.copy()

        # Broadcasting mu (K,1) handles the repmat operation
        Mat = H * (C / (D - mu + EPS_STABILITY))
        xi = np.sum(Mat, axis=1, keepdims=True) - delta

        Matp = H * C / (D - mu + EPS_STABILITY) ** 2
        xip = np.sum(Matp, axis=1, keepdims=True)

        mu = mu - xi / (xip + EPS_STABILITY)

        if np.max(np.abs(mu - mu_prev)) <= epsi:
            do_loop = False
        k += 1

    if k == max_iter_mu:
        warnings.warn("updatemu_hrows: Newton-Raphson reached max iterations.")
    return mu


# ==============================================================================
# ADMM Update Functions
# ==============================================================================


def update_wl_admm(
    W0,
    rho,
    maxIter,
    thres,
    innerloop,
    nu,
    epsi,
    rl,
    delta,
    alpha_l,
    lam_l,
    Wlminus1,
    Hl,
    Wp,
    accelerated=False,
):
    """Base function for ADMM procedure to update W_l."""
    # Initialization
    W = W0.copy()
    Z = W.copy()
    U = np.zeros_like(W0)
    mu = np.zeros((rl, 1))
    m, n = Wlminus1.shape

    residual = np.zeros(maxIter + 1)
    residual[0] = norm(W - Z, "fro")

    # Pre-computations
    Y = inv(W.T @ W + delta * np.eye(rl))
    Y_plus = np.maximum(0, Y)
    Y_minus = np.maximum(0, -Y)
    Eta_term1 = 8 * alpha_l / lam_l * (W0 @ (Y_plus + Y_minus))
    Eta_term2 = 4 * rho / lam_l * np.ones((m, rl))
    Eta_term3 = (Wlminus1 / (W0 @ Hl + EPS_STABILITY)) @ Hl.T
    Eta = (Eta_term1 + Eta_term2) * Eta_term3
    Psi = 4 * alpha_l / lam_l * W0 @ (Y_plus + Y_minus) + 2 * rho / lam_l * np.ones(
        (m, rl)
    )

    # Extrapolation variables (for accelerated version)
    if accelerated:
        Uchap, Zchap = U, Z
        Z_prev, U_prev = Z, U
        r_accel = 10

    # Main optimization loop
    start_time = time.process_time()
    k = 0
    res = float("inf")

    while k < maxIter and res > thres:
        # Extrapolation step
        if accelerated:
            gammak = k / (k + r_accel)
            Uchap = U + gammak * (U - U_prev)
            Zchap = Z + gammak * (Z - Z_prev)
        else:
            Uchap, Zchap = U, Z

        # W-minimization
        for i in range(innerloop):
            V = Zchap - Uchap
            Phi_term1 = np.ones((1, n)) @ Hl.T
            Phi = (
                np.tile(Phi_term1, (m, 1))
                - 4 * alpha_l / lam_l * (W0 @ Y_minus)
                - rho / lam_l * V
            )
            mu = updatemu(Phi, Eta, Psi, W, mu, epsi)

            # Update W
            Phi_mu = Phi + mu.T
            W = W0 * (np.sqrt(Phi_mu**2 + Eta) - Phi_mu) / (Psi + EPS_STABILITY)
            W = np.maximum(W, EPS_STABILITY)

        # Z-minimization
        V = W + Uchap
        b = -np.log(Wp + EPS_STABILITY) - nu * V
        # NOTE: Prevent overflow in np.exp(-b)
        b_clipped = np.clip(-b, -700, 700)
        Z = (lambertw(np.exp(-b_clipped) * nu).real) / (nu + EPS_STABILITY)

        # Dual updates
        if accelerated:
            U = Uchap + (W - Z)
            Z_prev, U_prev = Z, U
        else:
            U = U + W - Z

        # Compute residual
        res = norm(W - Z, "fro")
        residual[k + 1] = res
        k += 1

    tcpu = time.process_time() - start_time
    return W, Z, U, residual[:k], tcpu


# ==============================================================================
# Level Update Functions
# ==============================================================================


def level_update_deep_klnmf(H, X, W, Wp, lam, epsi, beta, HnormType):
    """Updates one level of the deep NMF model (without min-volume regularization)."""
    m, n = X.shape
    r1 = W.shape[1]

    # Update H using framework from (Leplat et al., 2021)
    prod = W @ H + EPS_STABILITY
    if beta == 1:
        C = W.T @ (X / prod)
        D = np.sum(W, axis=0).reshape(-1, 1)  # Same as W.T @ ones(m,n)
    elif beta == 3 / 2:
        C = W.T @ (prod ** (beta - 2) * X)
        D = W.T @ (prod ** (beta - 1))
    else:
        raise ValueError("beta must be 1 or 3/2 for this update rule.")

    if HnormType == "rows":
        # Find min value in each row of D to initialize mu
        I = np.argmin(D, axis=1)
        idx = (np.arange(r1), I)
        mu_0_H = (D[idx] - C[idx] * H[idx]).reshape(-1, 1)
        mu_H = updatemu_hrows(C, D, H, 1, mu_0_H, epsi)
        H = H * (C / (D - mu_H + EPS_STABILITY))

    elif HnormType == "cols":
        I = np.argmin(D, axis=0)
        idx = (I, np.arange(n))
        mu_0_H = (D[idx] - C[idx] * H[idx]).reshape(-1, 1)
        # Here we use the updatemu_hcols from the other file
        mu_H = multilayerNMF.updatemu_hcols(C, D, H, 1, mu_0_H, epsi)
        H = H * (C / (D - mu_H.T + EPS_STABILITY))

    H = np.maximum(H, EPS_STABILITY)

    # Update W
    Ht = H.T
    if beta == 1:
        a = np.sum(H, axis=1).reshape(1, -1) - lam * np.log(Wp + EPS_STABILITY)
        b = W * ((X / prod) @ Ht)
        lam_inv = 1.0 / (lam + EPS_STABILITY)
        W = np.maximum(
            EPS_STABILITY,
            lam_inv * b / (lambertw(lam_inv * b * np.exp(a * lam_inv)).real),
        )
    elif beta == 3 / 2:
        prod_pow = np.sqrt(prod)
        W_pow = np.sqrt(W)
        A = (1.0 / (W_pow + EPS_STABILITY)) * (prod_pow @ Ht) + 2 * lam
        B = W_pow * ((X / prod_pow) @ Ht)
        C_term = 2 * lam * np.sqrt(Wp)
        W = 0.25 * ((C_term + np.sqrt(C_term**2 + 4 * A * B)) / A) ** 2
        W = np.maximum(EPS_STABILITY, W)

    return W, H


def level_update_deep_minvol_klnmf(H, X, W, Wp, options, l):
    """Updates one level of the deep NMF model with min-volume regularization."""
    m, n = X.shape
    r = W.shape[1]
    l_idx = l - 1  # Convert from 1-based to 0-based index

    # Load parameters from options dictionary
    lam_l = options["lam"][l_idx]
    nu = options["rho"] / options["lam"][l_idx + 1]

    # Update W using the ADMM procedure
    # TODO: double check this one since both options are the same
    if options["accADMM"]:
        W, _, _, res, _ = update_wl_admm(
            W,
            options["rho"],
            options["maxIterADMM"],
            options["thres"],
            options["innerloop"],
            nu,
            options["epsi"],
            r,
            options["delta"][l_idx],
            options["alpha"][l_idx],
            lam_l,
            X,
            H,
            Wp,
            options["accADMM"],
        )
    else:
        W, Z, U, res, tcpu = update_wl_admm(
            W,
            options["rho"],
            options["maxIterADMM"],
            options["thres"],
            options["innerloop"],
            nu,
            options["epsi"],
            r,
            options["delta"][l_idx],
            options["alpha"][l_idx],
            lam_l,
            X,
            H,
            Wp,
        )
    # Update H using standard KL multiplicative update
    prod = W @ H + EPS_STABILITY
    Wt = W.T
    WtE = np.sum(W, axis=0).reshape(-1, 1)
    H = H * (Wt @ ((X / prod))) / (WtE + EPS_STABILITY)
    H = np.maximum(H, EPS_STABILITY)
    return W, H, res


# ==============================================================================
# Main Deep NMF Algorithm
# ==============================================================================


def deep_kl_nmf(X, r, init, params: DeepNMFParams):
    """
    Performs Deep KL-NMF.
    X ≈ W{1}H{1}, W{1} ≈ W{2}H{2}, ..., W{L-1} ≈ W{L}H{L}
    """
    if params is None:
        raise ValueError("no deep nmf parameters/options passed")
    L = len(r)

    # Initialization using the multilayer method from the other file
    if params.Wl is None or params.Hl is None:
        if params.display:
            print("Initialization with multi-layer beta-NMF")
        params.normalize = 2 if params.min_vol is False else 3
        W, H, e_init = multilayerNMF.multilayer_klnmf(X, r, init, asdict(params))
        e = np.zeros((params.outerit + 1, L + 1))
        e[0, :L] = e_init
    else:
        W, H = params.Wl, params.Hl
        e = np.zeros((params.outerit + 1, L + 1))
    inWH = {"W": [w.copy() for w in W], "H": [h.copy() for h in H]}
    output = {}
    # Initialize lambda if not provided
    if params.lam is None:
        params.lam = 1.0 / (e[0, :L] + EPS_STABILITY)
    e[0, L] = params.lam @ e[0, :L]

    # Alternating optimization loop
    if params.display:
        print("Alternating optimization of the L layers...")
    for it in range(params.outerit):
        for i in range(L):  # 0 to L-1
            current_X = X if i == 0 else W[i - 1]
            if i < L - 1:
                Wp = W[i + 1] @ H[i + 1]

            # --- Select update rule based on layer and options ---
            if params.min_vol is not False and params.beta == 1:
                if i < L - 1:
                    W[i], H[i], _ = level_update_deep_minvol_klnmf(
                        H[i], current_X, W[i], Wp, asdict(params), i + 1
                    )
                else:  # Last layer (no Wp)
                    H[i], _ = multilayerNMF.mu_beta(current_X, W[i], H[i], params.beta)
                    W_T, _ = multilayerNMF.mu_beta(
                        current_X.T, H[i].T, W[i].T, params.beta
                    )
                    W[i] = W_T.T
            else:
                # Standard deep NMF update
                if i < L - 1:
                    lam = params.lam[i + 1] / params.lam[i]
                    W[i], H[i] = level_update_deep_klnmf(
                        H[i],
                        current_X,
                        W[i],
                        Wp,
                        lam,
                        params.epsi,
                        params.beta,
                        params.HnormType,
                    )
                else:  # Last layer
                    H[i], _ = multilayerNMF.mu_beta(current_X, W[i], H[i], params.beta)
                    W_T, _ = multilayerNMF.mu_beta(
                        current_X.T, H[i].T, W[i].T, params.beta
                    )
                    W[i] = W_T.T

            # Error computation for the current layer
            e[it + 1, i] = multilayerNMF.betadiv(current_X, W[i] @ H[i], params.beta)

        # Compute global weighted error
        e[it + 1, L] = params.lam @ e[it + 1, :L]

        if params.display and (it + 1) % 10 == 0:
            print(f"Iter {it+1:3d}: Global Error = {e[it+1, L]:.4f}")

    # Save final outputs
    output["e"] = e
    output["W"] = W
    output["H"] = H
    output["inWH"] = inWH
    return W, H, e, inWH, output


if __name__ == "__main__":
    # 1. Create a synthetic dataset
    m, n = 100, 80
    r_true = [15, 8]
    np.random.seed(42)
    W1_true = np.abs(np.random.randn(m, r_true[0]))
    W2_true = np.abs(np.random.randn(r_true[0], r_true[1]))
    H2_true = np.abs(np.random.randn(r_true[1], n))
    H1_true = W2_true @ H2_true + 0.1 * np.random.rand(r_true[0], n)
    X = W1_true @ H1_true + 0.1 * np.random.rand(m, n)

    r_model = [15, 8]
    options = DeepNMFParams(
        L=len(r_model),
        maxiter=200,
        outerit=100,
        beta=1,
        display=True,
        min_vol=True,
        rho=2,
        accADMM=False,
    )
    W, H, e, inWH, output = deep_kl_nmf(X, r_model, options)

    # 2. Print results
    print("\nDeep NMF complete.")
    print(f"Final global error: {e[-1, -1]:.4f}")
    for i in range(len(r_model)):
        print(f"Layer {i+1}: W shape={W[i].shape}, H shape={H[i].shape}")
