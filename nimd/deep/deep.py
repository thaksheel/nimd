# deepnmf_torch.py
import time
import torch
from scipy.special import lambertw
import time
import warnings
from typing import Literal

from .utils import DeepNMFParams, ensure_tensor
from .multilayer import MultilayerKLNMF
from ..core.initiliazation import perturb_init


class DeepNMF:
    def __init__(
        self, X, params, device: Literal["cuda", "cpu"], multilayer_nmf, dtype, eps_stab
    ):
        self.X = X
        self.params: DeepNMFParams = params
        self.device = torch.device(device)
        self.multilayer_obj: MultilayerKLNMF = multilayer_nmf
        self.dtype = dtype
        self.durations = [time.time()]
        self.time_info = {"items": ["init"], "times": [time.time()], "k": [0]}
        self.EPS_STABILITY: float = eps_stab

    def lambertw_tensor(self, x: torch.Tensor):
        """
        Compute Lambert W for a torch tensor x.
        Use torch.special.lambertw if available; otherwise convert to numpy, call scipy, convert back.
        """
        x_np = x.detach().cpu().numpy()
        w_np = lambertw(x_np)
        return torch.from_numpy(w_np.real).to(device=self.device, dtype=x.dtype)

    def updatemu_hrows(self, C, D, H, mu, epsi):
        """
        Newton-Raphson for mu for row-wise constraints. All inputs are torch tensors.
        """
        C = ensure_tensor(C, device=self.device, dtype=self.dtype)
        D = ensure_tensor(D, device=self.device, dtype=self.dtype)
        H = ensure_tensor(H, device=self.device, dtype=self.dtype)
        mu = ensure_tensor(mu, device=self.device, dtype=self.dtype)

        K, T = C.shape
        JN1 = torch.ones((T, 1), dtype=C.dtype, device=C.device)
        delta = 1.0
        maxitermu = 10000
        k = 1
        if mu.ndim == 1:
            mu = mu.reshape(-1, 1)
        doLoop = True
        while doLoop and k <= maxitermu:
            mu_prev = mu.clone()
            denom = D - mu @ JN1.t() + self.EPS_STABILITY
            Mat = H * (C / denom)
            xi = torch.sum(Mat, dim=1, keepdim=True) - delta
            Matp = H * C / (denom**2)
            xip = torch.sum(Matp, dim=1, keepdim=True)
            mu = mu - xi / xip
            if torch.max(torch.abs(mu - mu_prev)) <= epsi:
                doLoop = False
            k += 1
        self.time_recording(f"updatemu_hrows", k=k)
        if k == maxitermu:
            warnings.warn(
                "updatemu_hrows: Newton-Raphson reached max iterations. Change ephis for better results."
            )
        return mu

    def updatemu(self, Phi, Eta, Psi, W, mu, epsi):
        """
        Newton-Raphson for mu. All inputs are expected to be torch tensors.
        """
        # ensure tensors
        Phi = ensure_tensor(Phi, device=self.device, dtype=self.dtype)
        Eta = ensure_tensor(Eta, device=self.device, dtype=self.dtype)
        Psi = ensure_tensor(Psi, device=self.device, dtype=self.dtype)
        W = ensure_tensor(W, device=self.device, dtype=self.dtype)
        mu = ensure_tensor(mu, device=self.device, dtype=self.dtype)

        if mu.ndim == 1:
            mu = mu.unsqueeze(1)
        F, K = W.shape
        JF1 = torch.ones((F, 1), device=self.device, dtype=self.dtype)
        k = 0
        max_iter_mu = 1000
        while k < max_iter_mu:
            mu_prev = mu.clone()
            Phi_mu = Phi + JF1 @ mu.T  # broadcasting
            sqrt_term = torch.sqrt(Phi_mu * Phi_mu + Eta)
            Mat = W * (sqrt_term - Phi_mu) / (Psi + self.EPS_STABILITY)
            xi = (torch.sum(Mat, dim=0) - 1.0).unsqueeze(1)
            Matp = (W / (Psi + self.EPS_STABILITY)) * (Phi_mu / sqrt_term - 1.0)
            xip = torch.sum(Matp, dim=0).unsqueeze(1)
            mu = mu - xi / (xip + self.EPS_STABILITY)
            if torch.max(torch.abs(mu - mu_prev)) <= epsi:
                break
            k += 1
        self.time_recording(f"updatemu_minvol={self.params.min_vol}", k=k)
        if k == max_iter_mu:
            warnings.warn(
                "updatemu: Newton-Raphson reached max iterations. Change ephis for better results"
            )
        return mu

    def update_wl_admm(
        self,
        W0,
        nu,
        rl,
        delta,
        alpha_l,
        lam_l,
        Wlminus1,
        Hl,
        Wp,
    ):
        """
        ADMM update for W_l. All matrix inputs should be torch tensors or convertible.
        """
        # determine device from W0
        W0 = ensure_tensor(W0, device=self.device, dtype=self.dtype)
        W = W0.clone()
        Z = W.clone()
        U = torch.zeros_like(W0, device=self.device, dtype=self.dtype)
        mu = torch.zeros((rl, 1), device=self.device, dtype=self.dtype)
        m, n = Wlminus1.shape
        residual = torch.zeros(
            self.params.maxIterADMM + 1, device=self.device, dtype=self.dtype
        )
        residual[0] = torch.linalg.norm(W - Z, ord="fro")

        # Pre-computations
        Y = torch.linalg.inv(
            W.T @ W + delta * torch.eye(rl, device=self.device, dtype=self.dtype)
        )
        Y_plus = torch.maximum(
            Y, torch.zeros_like(Y, device=self.device, dtype=self.dtype)
        )
        Y_minus = torch.maximum(
            -Y, torch.zeros_like(Y, device=self.device, dtype=self.dtype)
        )
        Eta_term1 = 8 * alpha_l / lam_l * (W0 @ (Y_plus + Y_minus))
        Eta_term2 = (
            4
            * self.params.rho
            / lam_l
            * torch.ones((m, rl), device=self.device, dtype=self.dtype)
        )
        Eta_term3 = (Wlminus1 / (W0 @ Hl + self.EPS_STABILITY)) @ Hl.T
        Eta = (Eta_term1 + Eta_term2) * Eta_term3
        Psi = 4 * alpha_l / lam_l * W0 @ (
            Y_plus + Y_minus
        ) + 2 * self.params.rho / lam_l * torch.ones(
            (m, rl), device=self.device, dtype=self.dtype
        )
        if self.params.accADMM:
            Uchap, Zchap = U.clone(), Z.clone()
            Z_prev, U_prev = Z.clone(), U.clone()
            r_accel = 10
        k = 0
        res = float("inf")
        while k < self.params.maxIterADMM and res > self.params.thres:
            if self.params.accADMM:
                gammak = k / (k + r_accel)
                Uchap = U + gammak * (U - U_prev)
                Zchap = Z + gammak * (Z - Z_prev)
            else:
                Uchap, Zchap = U, Z
            for _ in range(self.params.innerloop):
                V = Zchap - Uchap
                Phi_term1 = (
                    torch.ones((1, n), device=self.device, dtype=self.dtype) @ Hl.T
                )
                Phi = (
                    Phi_term1.repeat(m, 1)
                    - 4 * alpha_l / lam_l * (W0 @ Y_minus)
                    - self.params.rho / lam_l * V
                )
                mu = self.updatemu(Phi, Eta, Psi, W, mu, self.params.epsi)
                Phi_mu = Phi + mu.T
                W = (
                    W0
                    * (torch.sqrt(Phi_mu * Phi_mu + Eta) - Phi_mu)
                    / (Psi + self.EPS_STABILITY)
                )
                W = torch.maximum(
                    W,
                    torch.tensor(
                        self.EPS_STABILITY, device=self.device, dtype=self.dtype
                    ),
                )
            V = W + Uchap
            b = -torch.log(Wp + self.EPS_STABILITY) - nu * V
            b_clipped = torch.clamp(-b, min=-100.0, max=100.0)
            # lambertw on tensor
            Z = self.lambertw_tensor(torch.exp(-b_clipped) * nu).real / (
                nu + self.EPS_STABILITY
            )
            if self.params.accADMM:
                U = Uchap + (W - Z)
                Z_prev, U_prev = Z.clone(), U.clone()
            else:
                U = U + W - Z
            res = torch.linalg.norm(W - Z, ord="fro").item()
            residual[k + 1] = res
            k += 1
        return W, residual[:k]

    def level_update_deep_klnmf(self, H, X, W, Wp, lam, epsi, beta, HnormType):
        """
        Update one level without min-vol. Inputs can be numpy or torch; converted to torch.
        Returns W, H as torch tensors.
        """
        W = ensure_tensor(W, device=self.device, dtype=self.dtype)
        H = ensure_tensor(H, device=self.device, dtype=self.dtype)
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        Wp = ensure_tensor(Wp, device=self.device, dtype=self.dtype)
        lam = float(lam)
        # self.time_recording("ensuring_tensor")
        m, n = X.shape
        r1 = W.shape[1]
        e = torch.ones((m, n), dtype=X.dtype, device=X.device)

        # --- Update H ---
        prod = W @ H
        JN1 = torch.ones((n, 1), dtype=X.dtype, device=X.device)
        Jr1 = torch.ones((r1, 1), dtype=X.dtype, device=X.device)
        Wt = W.t()
        if beta == 1:
            C = Wt @ (X / prod)
            D = Wt @ e
        elif beta == 3 / 2:
            C = Wt @ ((prod ** (beta - 2)) * X)
            D = Wt @ (prod ** (beta - 1))
        else:
            raise ValueError("beta must be 1 or 3/2")
        if HnormType == "rows":
            D_min_vals, I = torch.min(D, dim=1)
            row_idx = torch.arange(r1, device=X.device)
            mu_0_H = (D[row_idx, I] - C[row_idx, I] * H[row_idx, I]).reshape(-1, 1)
            mu_H = self.updatemu_hrows(C, D, H, mu_0_H, epsi)
            denom = D - mu_H @ JN1.t() + self.EPS_STABILITY
            H = H * (C / denom)
        elif HnormType == "cols":
            D_min_vals, I = torch.min(D, dim=0)
            col_idx = torch.arange(n, device=X.device)
            mu_0_H = (D[I, col_idx] - C[I, col_idx] * H[I, col_idx]).reshape(-1, 1)
            mu_H = self.multilayer_obj.updatemu_hcols(C, D, H, mu_0_H, epsi)
            denom = D - Jr1 @ mu_H.t() + self.EPS_STABILITY
            H = H * (C / denom)
        else:
            raise ValueError("HnormType must be 'rows' or 'cols'")
        H = torch.clamp(H, min=self.EPS_STABILITY)

        # --- Update W ---
        Ht = H.t()
        if beta == 1:
            a = e @ Ht - lam * torch.log(Wp)
            b = W * ((X / (W @ H)) @ Ht)
            z = (1.0 / lam) * b * torch.exp(a / lam)
            w_val = self.lambertw_tensor(z)
            W = (1.0 / lam) * b / w_val
            W = torch.clamp(W, min=self.EPS_STABILITY)

        elif beta == 3 / 2:
            prod_pow = torch.sqrt(W @ H)
            W_pow = torch.sqrt(W)
            A = (1.0 / W_pow) * (prod_pow @ Ht) + 2 * lam
            B = W_pow * ((X / prod_pow) @ Ht)
            C = 2 * lam * torch.sqrt(Wp)
            inner = C + torch.sqrt(C**2 + 4 * A * B)
            W = 0.25 * (inner / A) ** 2
            W = torch.clamp(W, min=self.EPS_STABILITY)
        else:
            raise ValueError("beta must be 1 or 3/2")

        return W, H

    def level_update_deep_minvol_klnmf(self, H, X, W, Wp, l: int):
        """
        Update one level with min-vol regularization.
        All numeric arrays in options are converted to tensors on the device of W.
        """
        r = W.shape[1]
        l_idx = l - 1
        lam_l = float(self.params.lam[l_idx])
        nu = self.params.rho / self.params.lam[l_idx + 1]
        W, res = self.update_wl_admm(
            W0=W,
            nu=nu,
            rl=r,
            delta=self.params.delta[l_idx],
            alpha_l=self.params.alpha[l_idx],
            lam_l=lam_l,
            Wlminus1=X,
            Hl=H,
            Wp=Wp,
        )
        prod = W @ H + self.EPS_STABILITY
        Wt = W.T
        WtE = torch.sum(W, dim=0).reshape(-1, 1)
        H = H * (Wt @ ((X / prod))) / (WtE + self.EPS_STABILITY)
        H = torch.maximum(
            H, torch.tensor(self.EPS_STABILITY, device=self.device, dtype=self.dtype)
        )

        return W, H, res

    def time_recording(self, task_name: str, **kwargs):
        self.time_info["items"].append(task_name)
        self.time_info["times"].append(time.time() - self.durations[-1])
        if "k" in kwargs:
            self.time_info["k"].append(kwargs["k"])
        self.durations.append(time.time())
        return self

    def run(
        self,
        init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"],
        perturb: bool = False,
        nl: float = None,
    ):
        """
        Deep KL-NMF adapted to torch tensors. Accepts numpy arrays or torch tensors.
        Returns W, H as lists of torch tensors and an output dict.
        """
        layers_len = len(self.params.layers_rank)
        X_t = ensure_tensor(self.X, device=self.device, dtype=self.dtype)

        # NOTE: Initialization using multilayerNMF
        if self.params.Wl is None or self.params.Hl is None:
            if self.params.min_vol:
                if self.params.normalize != 3:
                    raise ValueError(
                        f"min_vol requires self.params.normalize=3 instead got self.params.normalize={self.params.normalize}"
                    )
            self.multilayer_obj.eps_stab = self.EPS_STABILITY
            W, H, e_init = self.multilayer_obj.run(init, perturb, nl)
            e = torch.zeros(
                (self.params.outerit + 1, layers_len + 1),
                device=self.device,
                dtype=self.dtype,
            )
            e[0, :layers_len] = ensure_tensor(
                e_init, device=self.device, dtype=self.dtype
            ).reshape(-1)
        else:
            W, H = self.params.Wl, self.params.Hl
            e = torch.zeros(
                (self.params.outerit + 1, layers_len + 1),
                device=self.device,
                dtype=self.dtype,
            )
        if False:
            if perturb:
                arr = [perturb_init(w, h, eta=nl) for w, h in zip(W, H)]
                W = [a[0] for a in arr]
                H = [a[1] for a in arr]
        output = {}
        if self.params.lam is None:
            e0 = e[0, :layers_len]
            self.params.lam = ensure_tensor(
                (1.0 / (e0 + self.EPS_STABILITY)).detach().cpu().tolist(),
                self.device,
                self.dtype,
            )
        else:
            self.params.lam = ensure_tensor(self.params.lam, self.device, self.dtype)
        e[0, layers_len] = torch.dot(self.params.lam, e[0, :layers_len])

        # min-vol initialization
        if self.params.min_vol:
            logdetSave = torch.zeros(
                (self.params.outerit + 1, layers_len),
                device=self.device,
                dtype=self.dtype,
            )
            output["ratio"] = torch.zeros(
                (2, layers_len), device=self.device, dtype=self.dtype
            )
            cumsum = self.params.lam @ e[0, :layers_len]
            for i in range(layers_len):
                if i == 0:
                    df = self.multilayer_obj.betadiv(X_t, W[i] @ H[i], self.params.beta)
                else:
                    df = self.multilayer_obj.betadiv(
                        W[i - 1], W[i] @ H[i], self.params.beta
                    )
                # compute logdet using torch if W[i] is tensor
                Wi = W[i]
                mv = torch.log10(
                    torch.linalg.det(
                        Wi.T @ Wi
                        + self.params.delta[i]
                        * torch.eye(Wi.shape[1], device=self.device, dtype=self.dtype)
                    )
                ).item()
                self.params.alpha[i] = self.params.alpha_tilde[i] * df / abs(mv)
                logdetSave[0, i] = mv
                output["ratio"][0, i] = self.params.alpha[i] * logdetSave[0, i] / df
                cumsum = cumsum + self.params.alpha[i] * mv
            e[0, layers_len] = cumsum
            mu = torch.zeros(
                (self.params.layers_rank[layers_len - 1], 1),
                device=self.device,
                dtype=self.dtype,
            )
            E = torch.ones_like(
                W[layers_len - 1] @ H[layers_len - 1],
                device=self.device,
                dtype=self.dtype,
            )

        # Alternating optimization
        self.time_recording("before_deepnmf_optimization")
        if self.params.display:
            print("Alternating optimization of the L layers...")
        if self.params.min_vol:
            e_m = torch.zeros(
                (self.params.outerit + 1, 1), device=self.device, dtype=self.dtype
            )
            e_m[0, 0] = e[0, layers_len]
            res = {}

        # FIXME: temp resolve for H > eps_stab
        H = [torch.clamp(h, min=self.EPS_STABILITY) for h in H]
        W = [torch.clamp(w, min=self.EPS_STABILITY) for w in W]
        for it in range(self.params.outerit):
            for i in range(layers_len):
                if i == 0 and layers_len > 1:
                    current_X = X_t
                    if not self.params.min_vol:
                        lam_ratio = self.params.lam[1] / self.params.lam[0]
                        Wp = W[1] @ H[1]
                        W[0], H[0] = self.level_update_deep_klnmf(
                            H=H[0],
                            X=current_X,
                            W=W[0],
                            Wp=Wp,
                            lam=lam_ratio,
                            epsi=self.params.epsi,
                            beta=self.params.beta,
                            HnormType=self.params.HnormType,
                        )
                        # self.time_recording(f"lvl_up_deep_outerit={it}")
                    elif self.params.beta == 1:
                        Wp = W[1] @ H[1]
                        W[0], H[0], res[0] = self.level_update_deep_minvol_klnmf(
                            H[0], current_X, W[0], Wp, 0
                        )
                        self.time_recording(f"lvl_up_deep_minvol_outerit={it}")
                    e[it + 1, 0] = self.multilayer_obj.betadiv(
                        current_X, W[0] @ H[0], self.params.beta
                    )
                    if self.params.min_vol:
                        Wi = W[0]
                        logdetSave[it + 1, 0] = torch.log10(
                            torch.linalg.det(
                                Wi.T @ Wi
                                + self.params.delta[0]
                                * torch.eye(
                                    self.params.layers_rank[0],
                                    dtype=self.dtype,
                                    device=self.device,
                                )
                            )
                        )
                elif i == layers_len - 1:
                    current_X = W[layers_len - 2] if layers_len > 1 else X_t
                    if not self.params.min_vol:
                        if self.params.HnormType == "rows":
                            H[i], _ = self.multilayer_obj.mu_beta(
                                current_X, W[i], H[i], self.params.beta
                            )
                            W_T, _ = self.multilayer_obj.mu_beta(
                                current_X.T, H[i].T, W[i].T, self.params.beta
                            )
                            W[i] = W_T.T
                            W[i], H[i] = self.multilayer_obj.normalize_wh(
                                W[i], H[i], sumtoone=2
                            )
                        elif self.params.HnormType == "cols":
                            prod = W[i] @ H[i]
                            Wt = W[i].T
                            C = Wt @ (((prod) ** (self.params.beta - 2)) * current_X)
                            D = Wt @ (prod ** (self.params.beta - 1))
                            m, n = current_X.shape
                            I = torch.argmin(D, axis=1)
                            idx = (
                                I,
                                torch.arange(n, device=self.device, dtype=self.dtype),
                            )
                            mu_0_H = D[idx] - C[idx] * H[i][idx]
                            mu_H = self.multilayer_obj.updatemu_hcols(
                                C, D, H[i], mu_0_H, self.params.epsi, self.EPS_STABILITY
                            )
                            H[i] = H[i] * (
                                C
                                / (
                                    D
                                    - torch.ones(
                                        (self.params.layers_rank[i], 1),
                                        device=self.device,
                                        dtype=self.dtype,
                                    )
                                    @ mu_H
                                    + self.EPS_STABILITY
                                )
                            )
                            H[i] = torch.maximum(H[i], self.EPS_STABILITY)
                            W_T, _ = self.multilayer_obj.mu_beta(
                                current_X.T, H[i].T, W[i].T, self.params.beta
                            )
                            W[i] = W_T.T
                        e[it + 1, i] = self.multilayer_obj.betadiv(
                            current_X, W[i] @ H[i], self.params.beta
                        )
                    elif self.params.beta == 1:
                        Wprev = X_t if i == 0 else W[i - 1]
                        m, n = Wprev.shape
                        Wi = W[i]
                        Y = torch.linalg.inv(
                            Wi.T @ Wi
                            + self.params.delta[i]
                            * torch.eye(
                                self.params.layers_rank[i],
                                device=self.device,
                                dtype=self.dtype,
                            )
                        )
                        Y_plus = torch.maximum(
                            Y, torch.zeros_like(Y, device=self.device, dtype=self.dtype)
                        )
                        Y_minus = torch.maximum(
                            -Y,
                            torch.zeros_like(Y, device=self.device, dtype=self.dtype),
                        )
                        Phi = (
                            torch.tile(
                                torch.ones((1, n), device=self.device, dtype=self.dtype)
                                @ H[i].T,
                                (m, 1),
                            )
                            - 4 * self.params.alpha[i] * Wi @ Y_minus
                        )
                        Eta = (
                            8
                            * self.params.alpha[i]
                            * (Wi @ (Y_plus + Y_minus))
                            * ((Wprev / (Wi @ H[i] + self.EPS_STABILITY)) @ H[i].T)
                        )
                        Psi = 4 * self.params.alpha[i] * Wi @ (Y_plus + Y_minus)
                        mu = self.updatemu(Phi, Eta, Psi, Wi, mu, self.params.epsi)
                        Phi_mu = (
                            Phi
                            + torch.ones((m, 1), device=self.device, dtype=self.dtype)
                            @ mu.T
                        )
                        W[i] = W[i] * (
                            (torch.sqrt(Phi_mu * Phi_mu + Eta) - Phi_mu)
                            / (Psi + self.EPS_STABILITY)
                        )
                        W[i] = torch.maximum(
                            W[i],
                            torch.tensor(
                                self.EPS_STABILITY, device=self.device, dtype=self.dtype
                            ),
                        )
                        prod = W[i] @ H[i]
                        W_t = W[i].T
                        H[i] = H[i] * (
                            W_t @ ((E / prod) * Wprev) / (W_t @ E + self.EPS_STABILITY)
                        )
                        H[i] = torch.maximum(
                            H[i],
                            torch.tensor(
                                self.EPS_STABILITY, device=self.device, dtype=self.dtype
                            ),
                        )
                        e[it + 1, i] = self.multilayer_obj.betadiv(
                            Wprev, W[i] @ H[i], self.params.beta
                        )
                        logdetSave[it + 1, i] = torch.log10(
                            torch.linalg.det(
                                W[i].T @ W[i]
                                + self.params.delta[i]
                                * torch.eye(
                                    self.params.layers_rank[i],
                                    device=self.device,
                                    dtype=self.dtype,
                                )
                            )
                        )
                else:
                    current_X = W[i - 1]
                    if not self.params.min_vol:
                        lam_ratio = self.params.lam[i + 1] / self.params.lam[i]
                        Wp = W[i + 1] @ H[i + 1]
                        W[i], H[i] = self.level_update_deep_klnmf(
                            H[i],
                            current_X,
                            W[i],
                            Wp,
                            lam_ratio,
                            self.params.epsi,
                            self.params.beta,
                            self.params.HnormType,
                        )
                    elif self.params.beta == 1:
                        Wp = W[i + 1] @ H[i + 1]
                        W[i], H[i], res[i] = self.level_update_deep_minvol_klnmf(
                            H[i], current_X, W[i], Wp, i
                        )
                    e[it + 1, i] = self.multilayer_obj.betadiv(
                        current_X, W[i] @ H[i], self.params.beta
                    )
                    if self.params.min_vol:
                        Wi = W[i]
                        logdetSave[it + 1, i] = torch.log10(
                            torch.linalg.det(
                                Wi.T @ Wi
                                + self.params.delta[i]
                                * torch.eye(
                                    self.params.layers_rank[i],
                                    device=self.device,
                                    dtype=self.dtype,
                                )
                            )
                        )
            e[it + 1, layers_len] = self.params.lam @ e[it + 1, :layers_len]
            if self.params.min_vol:
                e_m[it + 1, 0] = (
                    self.params.lam @ e[it + 1, :layers_len]
                    + ensure_tensor(self.params.alpha, self.device, dtype=self.dtype)
                    @ logdetSave[it + 1, :layers_len]
                )
            if self.params.display:
                if (it + 1) % 10 == 0:
                    print(f"{it + 1:2.0f} : {e[it + 1, layers_len]:1.2f} - ", end="")
                if (it + 1) % 50 == 0:
                    print()
        output["e"] = e
        if self.params.min_vol:
            output["logdetEvol"] = logdetSave
            output["e_m"] = e_m
            output["alpha"] = self.params.alpha
            for i in range(layers_len):
                df = e[-1, i]
                output["ratio"][1, i] = self.params.alpha[i] * logdetSave[-1, i] / df
        self.time_recording(f"end")
        return W, H, output
