# nmf_beta_divergence.py

import numpy as np
import scipy.sparse as sp
from numpy.linalg import norm
import time
from typing import Literal, Optional, List, Dict
from sklearn.utils.validation import check_array, check_non_negative, check_random_state
from sklearn.utils.extmath import randomized_svd
from nmfs.nnsvdlrc import NNSVDLRC


# A small constant for numerical stability to avoid division by zero.
EPS_STABILITY = 1e-9

# ==============================================================================
# Helper and Projection Functions
# ==============================================================================


def initialize_nmf(
    X,
    n_components,
    init: str = Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"],
    eps=1e-6,
    random_state=None,
):
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : int
        The number of components desired in the approximation.

    init :  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - None: 'nndsvda' if n_components <= min(n_samples, n_features),
            otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    eps : float, default=1e-6
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like of shape (n_samples, n_components)
        Initial guesses for solving X ~= WH.

    H : array-like of shape (n_components, n_features)
        Initial guesses for solving X ~= WH.

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape

    if (
        init is not None
        and init != "random"
        and n_components > min(n_samples, n_features)
    ):
        raise ValueError(
            "init = '{}' can only be used when "
            "n_components <= min(n_samples, n_features)".format(init)
        )

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = "nndsvda"
        else:
            init = "random"

    # NNSVDLRC initialization
    if init == "nnsvdlrc":
        W, H, *_ = NNSVDLRC(X, n_components)
        return W, H

    # Random initialization
    if init == "random":
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.standard_normal(size=(n_components, n_features)).astype(
            X.dtype, copy=False
        )
        W = avg * rng.standard_normal(size=(n_samples, n_components)).astype(
            X.dtype, copy=False
        )
        np.abs(H, out=H)
        np.abs(W, out=W)
        return W, H

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.standard_normal(size=len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.standard_normal(size=len(H[H == 0])) / 100)
    else:
        raise ValueError(
            "Invalid init parameter: got %r instead of one of %r"
            % (init, (None, "random", "nndsvd", "nndsvda", "nndsvdar"))
        )

    return W, H


def updatemu_hcols(C, D, H, gamma_beta, mu, epsi):
    """
    Updates the Lagrange multipliers 'mu' using Newton's method.
    This is a helper function used in multilayer_klnmf for the 'cols' option.

    Args:
        C (np.ndarray): The C matrix in the update rule.
        D (np.ndarray): The D matrix in the update rule.
        H (np.ndarray): The current H matrix.
        gamma_beta (float): The gamma_beta parameter (not used in the active code).
        mu (np.ndarray): The current Lagrange multipliers.
        epsi (float): The convergence tolerance.

    Returns:
        np.ndarray: The updated Lagrange multipliers 'mu'.
    """
    K, T = C.shape
    delta = 1.0
    do_loop = True
    max_iter_mu = 10**4
    k = 0

    while do_loop and k < max_iter_mu:
        mu_prev = mu.copy()
        Mat = H * (C / (D - mu.T + EPS_STABILITY))
        xi = (np.sum(Mat, axis=0) - delta).reshape(-1, 1)
        Matp = H * C / (D - mu.T + EPS_STABILITY) ** 2
        xip = np.sum(Matp, axis=0).reshape(-1, 1)
        mu = mu - xi / (xip + EPS_STABILITY)
        if np.max(np.abs(mu - mu_prev)) <= epsi:
            do_loop = False
        k += 1

    return mu


def simplex_col_proj(Y):
    """
    Projects each column vector in the matrix Y onto the probability simplex.

    This is a Python implementation of the algorithm described in:
    "Projection onto the probability simplex: An efficient algorithm with a
    simple proof, and an application" by W. Wang and M.A. Carreira-Perpinan.
    https://arxiv.org/abs/1309.1541

    Args:
        Y (np.ndarray): A D-by-N matrix where each column is a vector to be projected.

    Returns:
        np.ndarray: The projected matrix X.
    """
    Y_orig_shape = Y.shape
    if len(Y_orig_shape) == 1:
        Y = Y.reshape(-1, 1)
    Y_t = Y.T
    N, D = Y_t.shape
    X = np.sort(Y_t, axis=1)[:, ::-1]
    X_tmp = (np.cumsum(X, axis=1) - 1) / np.arange(1, D + 1)
    # Find the value rho for each row
    rho = np.sum(X > X_tmp, axis=1) - 1  # -1 for 0-based indexing
    lambda_vals = X_tmp[np.arange(N), rho]
    X_proj_t = np.maximum(Y_t - lambda_vals[:, np.newaxis], 0)
    X_proj = X_proj_t.T
    if len(Y_orig_shape) == 1:
        return X_proj.flatten()
    return X_proj


def simplex_proj(Y):
    """
    Projects columns of Y onto the L1-ball {x | x >= 0, sum(x) <= 1}.

    This function was not provided in the original MATLAB code. This is a
    plausible implementation: it projects onto the non-negative orthant,
    and for columns whose sum exceeds 1, it then projects them onto the
    probability simplex {x | sum(x) = 1}.

    Args:
        Y (np.ndarray): A matrix where each column is a vector to be projected.

    Returns:
        np.ndarray: The projected matrix.
    """
    Y = np.maximum(Y, 0)
    col_sums = np.sum(Y, axis=0)
    idx_to_project = col_sums > 1
    if np.any(idx_to_project):
        Y[:, idx_to_project] = simplex_col_proj(Y[:, idx_to_project])
    return Y


# ==============================================================================
# Core NMF Algorithm Functions
# ==============================================================================


def orth_nnls(M, U, Mn=None):
    """
    Solves a type of Orthogonal Non-negative Least Squares problem by finding
    the best single column of U to represent each column of M.

    Args:
        M (np.ndarray): The m-by-n target matrix.
        U (np.ndarray): The m-by-r basis matrix.
        Mn (np.ndarray, optional): A column-normalized version of M. If not
                                   provided, it will be computed.

    Returns:
        np.ndarray: The resulting r-by-n coefficient matrix V.
    """
    # If Mn (normalized M) is not provided
    if Mn is None:
        norm2m = np.linalg.norm(M, axis=0)
        Mn = M / (norm2m + 1e-16)

    m, n = Mn.shape
    m_u, r = U.shape
    norm2u = np.linalg.norm(U, axis=0)
    Un = U / (norm2u + 1e-16)
    A = Mn.T @ Un
    best_u_indices = np.argmax(A, axis=1)
    V = np.zeros((r, n))
    U_best = U[:, best_u_indices]
    norm2u_best_sq = norm2u[best_u_indices] ** 2
    weights = np.einsum("ij,ij->j", M, U_best) / (norm2u_best_sq + 1e-16)

    # Assign the weights to the correct positions in V.
    V[best_u_indices, np.arange(n)] = weights

    return V


def nnls_init(X, W, WtW, WtX):
    """
    Initializes H for the Non-negative Least Squares problem min_{H>=0} ||X-WH||_F^2.
    [UPDATED to use the full orth_nnls function]

    Args:
        X (np.ndarray): The m-by-n target matrix.
        W (np.ndarray): The m-by-r basis matrix.
        WtW (np.ndarray): Precomputed W.T @ W.
        WtX (np.ndarray): Precomputed W.T @ X.

    Returns:
        np.ndarray: The initialized r-by-n coefficient matrix H.
    """
    if np.linalg.cond(W) > 1e6:
        # If W is ill-conditioned, use orthogonal NMF initialization.
        H = orth_nnls(X, W)
    else:
        # Otherwise, use the projected Least Squares solution.
        if sp.issparse(X):
            H = np.maximum(0, np.linalg.pinv(W) @ X.toarray())
        else:
            H = np.maximum(0, np.linalg.lstsq(W, X, rcond=None)[0])

        # Scale the result.
        numerator = np.sum(H * WtX)
        denominator = np.sum(WtW * (H @ H.T))

        if denominator > EPS_STABILITY:
            alpha = numerator / denominator
            H = H * alpha

    # Check for any all-zero rows and re-initialize them.
    row_sums = np.sum(H, axis=1)
    zero_rows = np.where(row_sums == 0)[0]

    if len(zero_rows) > 0:
        max_h_val = np.max(H)
        if max_h_val == 0:
            max_h_val = 1.0

        n_cols = H.shape[1]
        H[zero_rows, :] = 0.001 * max_h_val * np.random.rand(len(zero_rows), n_cols)

    return H


def nnls_fpgm(X, W, options=None):
    """
    Solves Non-negative Least Squares (min_{H>=0} ||X-WH||_F^2) using a
    Fast Projected Gradient Method (FPGM).

    Args:
        X (np.ndarray): The m-by-n target matrix.
        W (np.ndarray): The m-by-r basis matrix.
        options (dict, optional): A dictionary of options:
            'delta' (float): Stopping tolerance.
            'inneriter' (int): Maximum number of inner iterations.
            'proj' (int): Type of projection for H (0-3).
            'alpha0' (float): FPGM extrapolation parameter.
            'init' (np.ndarray): Initial guess for H.

    Returns:
        tuple: (H, WtW, WtX) containing the solution matrix H and precomputed terms.
    """
    if options is None:
        options = {}

    # Set default options
    delta = options.get("delta", 1e-6)
    inner_iter = options.get("inneriter", 500)
    proj = options.get("proj", 0)
    alpha0 = options.get("alpha0", 0.05)
    init_H = options.get("init", None)

    m, n = X.shape
    m_w, r = W.shape

    WtW = W.T @ W
    WtX = W.T @ X

    if init_H is None:
        H = nnls_init(X, W, WtW, WtX)
    else:
        H = init_H

    L = norm(WtW, 2)  # Lipschitz constant

    alpha = [alpha0]
    beta = []

    # Initial projection of H
    if proj == 1:
        H = simplex_proj(H)
    elif proj == 0:
        H = np.maximum(H, 0)
    elif proj == 2:
        H = simplex_col_proj(H.T).T
    elif proj == 3:
        H = simplex_col_proj(H)

    Y = H.copy()
    i = 0
    eps0 = 0.0
    eps = 1.0

    while i < inner_iter and (eps >= delta * eps0 if eps0 > 0 else True):
        Hp = H.copy()

        # FGM Coefficients (Nesterov's accelerated method)
        alpha_i = alpha[i]
        alpha_i_plus_1 = (np.sqrt(alpha_i**4 + 4 * alpha_i**2) - alpha_i**2) / 2
        alpha.append(alpha_i_plus_1)
        beta_val = alpha_i * (1 - alpha_i) / (alpha_i**2 + alpha_i_plus_1)
        beta.append(beta_val)

        # Gradient descent and projection step
        H = Y - (WtW @ Y - WtX) / L
        if proj == 1:
            H = simplex_proj(H)
        elif proj == 0:
            H = np.maximum(H, 0)
        elif proj == 2:
            H = simplex_col_proj(H.T).T
        elif proj == 3:
            H = simplex_col_proj(H)

        # Extrapolation step
        Y = H + beta_val * (H - Hp)

        if i == 0:
            eps0 = norm(H - Hp, "fro")
            if eps0 == 0:
                break
        eps = norm(H - Hp, "fro")
        i += 1

    return H, WtW, WtX


def normalize_wh(W, H, sumtoone, X, options_for_nnls=None):
    """
    Normalizes the factor matrices W and H based on the specified model.

    Args:
        W (np.ndarray): The basis matrix.
        H (np.ndarray): The coefficient matrix.
        sumtoone (int): Normalization type (1-4).
        X (np.ndarray): The original data matrix, needed for re-optimizing.
        options_for_nnls (dict, optional): Options for nnls_fpgm if called.

    Returns:
        tuple: The normalized (W, H) matrices.
    """
    if sumtoone == 1:  # Columns of H sum to at most 1
        Hn = simplex_proj(H)
        if norm(Hn - H) > 1e-3 * norm(Hn):
            H = Hn
            opts = (options_for_nnls or {}).copy()
            opts.update({"inneriter": 100, "init": W.T})
            W, _, _ = nnls_fpgm(X.T, H.T, opts)
            W = W.T
        H = Hn
    elif sumtoone == 2:  # Rows of H sum to 1
        scalH = np.sum(H, axis=1)
        scalH[scalH == 0] = 1  # Avoid division by zero
        H = np.diag(1.0 / scalH) @ H
        W = W @ np.diag(scalH)
    elif sumtoone == 3:  # Columns of W sum to 1
        scalW = np.sum(W, axis=0)
        scalW[scalW == 0] = 1
        H = np.diag(scalW) @ H
        W = W @ np.diag(1.0 / scalW)
    elif sumtoone == 4:  # Columns of H sum to 1
        Hn = simplex_col_proj(H)
        if norm(Hn - H) > 1e-3 * norm(Hn):
            H = Hn
            opts = (options_for_nnls or {}).copy()
            opts.update({"inneriter": 100, "init": W.T})
            W, _, _ = nnls_fpgm(X.T, H.T, opts)
            W = W.T
        H = Hn

    return W, H


# ==============================================================================
# Beta-Divergence and Multiplicative Updates
# ==============================================================================


def betadiv(X, Y, beta):
    """
    Computes the beta-divergence between matrices X and Y.

    Args:
        X (np.ndarray): The first matrix.
        Y (np.ndarray): The second matrix.
        beta (float): The beta parameter defining the divergence.
            beta = 2: Frobenius norm
            beta = 1: Kullback-Leibler divergence
            beta = 0: Itakura-Saito distance

    Returns:
        float: The total beta-divergence D_beta(X, Y).
    """
    eps = np.finfo(float).eps
    if beta == 0:  # Itakura-Saito
        ratio = X / (Y + eps)
        Z = ratio - np.log(ratio + eps) - 1
    elif beta == 1:  # Kullback-Leibler
        Z = X * np.log(X / (Y + eps) + eps) - X + Y
    else:  # General beta-divergence
        Y_beta = Y**beta
        Y_beta_1 = Y ** (beta - 1)
        Z = (X**beta + (beta - 1) * Y_beta - beta * X * Y_beta_1) / (
            beta * (beta - 1) + eps
        )
    return np.sum(Z)


def compwiseprodsparselowrank_vectorized(X, W, H, fun):
    """
    Vectorized computation of Y = fun(X, W*H) for a sparse matrix X.
    This avoids forming the dense matrix W*H.

    Args:
        X (sp.spmatrix): A sparse matrix.
        W (np.ndarray): The W factor matrix.
        H (np.ndarray): The H factor matrix.
        fun (callable): A function to apply element-wise, e.g., lambda s, wh: s / wh.

    Returns:
        sp.spmatrix: The resulting sparse matrix Y.
    """
    if not sp.issparse(X):
        raise TypeError("Input matrix X must be a SciPy sparse matrix.")

    i, j, s = sp.find(X)
    wh_vals = np.sum(W[i, :] * H[:, j].T, axis=1)
    new_vals = fun(s, wh_vals)
    return sp.csc_matrix((new_vals, (i, j)), shape=X.shape)


def blockrecursivecompwiseprodsparselowrank(X, W, H, fun, nnzparam=1e4):
    """
    Recursively computes the component-wise product for sparse matrices by
    dividing the matrix into blocks.

    Args:
        X (sp.spmatrix): The sparse matrix.
        W, H (np.ndarray): Factor matrices.
        fun (callable): The function to apply.
        nnzparam (int, optional): The threshold of non-zero elements to switch
                                  to the vectorized base case.

    Returns:
        sp.spmatrix: The resulting sparse matrix Y.
    """
    if not sp.issparse(X) or X.nnz == 0:
        return sp.csc_matrix(X.shape, dtype=float)

    # Base case
    if X.nnz < nnzparam:
        return compwiseprodsparselowrank_vectorized(X, W, H, fun)

    # Recursive step
    Y = sp.lil_matrix(X.shape, dtype=float)
    nblocks = 2
    m, n = X.shape
    mi = int(np.ceil(m / nblocks))
    ni = int(np.ceil(n / nblocks))

    for i in range(nblocks):
        for j in range(nblocks):
            indi = slice(i * mi, min(m, (i + 1) * mi))
            indj = slice(j * ni, min(n, (j + 1) * ni))
            if X[indi, indj].nnz > 0:
                Y[indi, indj] = blockrecursivecompwiseprodsparselowrank(
                    X[indi, indj], W[indi, :], H[:, indj], fun, nnzparam
                )
    return Y.tocsc()


def nd_mubeta(X, W, H, beta):
    """
    Calculates the Numerator (N) and Denominator (D) for the multiplicative
    update rule for H, based on the beta-divergence.

    Args:
        X, W, H: The matrices.
        beta (float): The beta parameter.

    Returns:
        tuple: (N, D, error) for the H update.
    """
    e = None
    WH = W @ H

    if beta == 1:  # Kullback-Leibler
        if sp.issparse(X):

            def xdy_fun(x, y):
                return x / (y + EPS_STABILITY)

            XdWH = blockrecursivecompwiseprodsparselowrank(X, W, H, xdy_fun)
        else:
            XdWH = X / (WH + EPS_STABILITY)
        N = W.T @ XdWH
        # The sum is over columns of W (axis=0), then reshaped for broadcasting
        D = np.sum(W, axis=0).reshape(-1, 1)
        e = betadiv(X, WH, beta)

    elif beta == 2:  # Frobenius
        N = W.T @ X
        D = (W.T @ W) @ H
        e = 0.5 * norm(X - WH, "fro") ** 2

    else:  # Other beta-divergences (assumes dense X)
        WH_beta_2 = WH ** (beta - 2)
        WH_beta_1 = WH ** (beta - 1)
        N = W.T @ (WH_beta_2 * X)
        D = W.T @ WH_beta_1
        e = betadiv(X, WH, beta)

    return N, D, e


def mu_beta(X, W, H, beta, epsilon=None):
    """
    Performs a single multiplicative update (MU) for H to minimize D_beta(X, WH).

    Args:
        X, W, H: The matrices.
        beta (float): The beta parameter.
        epsilon (float, optional): A small lower bound for entries of H.
                                   Defaults to machine epsilon.

    Returns:
        tuple: (H_updated, error)
    """
    if epsilon is None:
        epsilon = np.finfo(float).eps

    N, D, e = nd_mubeta(X, W, H, beta)

    # Update rule with gamma power for convergence
    if 1 <= beta <= 2:
        H = H * (N / (D + EPS_STABILITY))
    else:
        gamma = 1 / (2 - beta) if beta < 1 else 1 / (beta - 1)
        H = H * ((N / (D + EPS_STABILITY)) ** gamma)

    H = np.maximum(epsilon, H)
    return H, e


def beta_nmf(
    X,
    r,
    init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"],
    options=None,
):
    """
    Solves the beta-NMF problem: min_{W,H >= 0} D_beta(X, WH).

    Args:
        X (np.ndarray): The m-by-n non-negative input matrix.
        r (int): The factorization rank.
        options (dict, optional): A dictionary of options.
            'maxiter', 'timemax', 'beta', 'epsilon', 'accuracy', 'display'
            'W', 'H' for initialization.

    Returns:
        tuple: (W, H, e) where W and H are the factors and e is the error history.
    """
    if options is None:
        options = {}

    m, n = X.shape
    max_iter = options.get("maxiter", 500)
    time_max = options.get("timemax", float("inf"))
    beta = options.get("beta", 1.0)
    epsilon = options.get("epsilon", np.finfo(float).eps)
    accuracy = options.get("accuracy", 1e-4)
    display = options.get("display", False)

    # W = np.random.rand(m, r) if options["W0"] is None else options["W0"]
    # H = np.random.rand(r, n) if options["H0"] is None else options["H0"]
    W, H = initialize_nmf(X, r, init, epsilon)
    e = []
    i = 0
    cpu_init = time.process_time()

    # Main optimization loop
    while i < max_iter and time.process_time() <= cpu_init + time_max:
        # Update H, then update W using the transpose trick
        H, _ = mu_beta(X, W, H, beta, epsilon)
        W_T, ei = mu_beta(X.T, H.T, W.T, beta, epsilon)
        W = W_T.T
        e.append(ei)
        # Stopping condition based on error change over 10 iterations
        if i >= 11 and abs(e[-1] - e[-11]) < accuracy * abs(e[-10]):
            if display:
                print("\nConvergence reached.")
            break
        # Scaling: max entry in each column of W is 1
        for k in range(r):
            mxk = np.max(W[:, k])
            if mxk > EPS_STABILITY:
                W[:, k] /= mxk
                H[k, :] *= mxk
        if display:
            if (i + 1) % 10 == 0:
                print(f"{i+1:3d}...", end="", flush=True)
            if (i + 1) % 100 == 0:
                print()
        i += 1
    if display:
        print("\n")
    return W, H, e


def multilayer_klnmf(
    X,
    r_list,
    init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"],
    options=None,
):
    """
    Performs multilayer NMF, where each layer factorizes the W matrix
    from the previous layer. X ≈ W1*H1, W1 ≈ W2*H2, etc.

    Args:
        X (np.ndarray): The input matrix.
        r_list (list of int): A list of ranks for each layer.
        options (dict, optional): A dictionary of options for the NMF solvers.
            'HnormType' can be 'rows' or 'cols' for different update strategies.

    Returns:
        tuple: (W_list, H_list, e) containing lists of factor matrices and
               the error at each layer.
    """
    if options is None:
        options = {}

    L = len(r_list)
    display = options.get("display", False)
    H_norm_type = options.get("HnormType", "rows")
    rng_seed = options.get("rngseed", None)
    beta = options.get("beta", 1.0)
    max_iter = options.get("maxiter", 500)
    normalize_type = options.get("normalize", 2)
    epsi = options.get("epsi", 1e-6)  # For updatemu_hcols

    if rng_seed is not None:
        np.random.seed(rng_seed)

    W_list, H_list = [None] * L, [None] * L
    e = np.zeros(L)

    for i in range(L):
        current_X = X if i == 0 else W_list[i - 1]
        r = r_list[i]

        if H_norm_type == "rows":
            W_i, H_i, ei = beta_nmf(current_X, r, init, options)
            W_i, H_i = normalize_wh(W_i, H_i, normalize_type, current_X)
            e[i] = ei[-1] if ei else betadiv(current_X, W_i @ H_i, beta)
            W_list[i], H_list[i] = W_i, H_i

        elif H_norm_type == "cols":
            m, n = current_X.shape
            # TODO: I might need an init here for this type
            # W_i = np.random.rand(m, r)
            # H_i = np.random.rand(r, n)
            W_i, H_i = initialize_nmf(X, r, init)

            for k in range(max_iter):
                prod = W_i @ H_i + EPS_STABILITY
                Wt = W_i.T
                C = Wt @ (prod ** (beta - 2) * current_X)
                D = Wt @ (prod ** (beta - 1))

                I = np.argmin(D, axis=0)
                idx = (I, np.arange(n))
                mu_0_H = (D[idx] - C[idx] * H_i[idx]).reshape(-1, 1)
                mu_H = updatemu_hcols(C, D, H_i, 1, mu_0_H, epsi)

                H_i = H_i * (C / (D - mu_H.T + EPS_STABILITY))
                H_i = np.maximum(H_i, EPS_STABILITY)
                W_T_i, _ = mu_beta(current_X.T, H_i.T, W_i.T, beta)
                W_i = W_T_i.T

                if display:
                    if (k + 1) % 10 == 0:
                        print(f"{k+1:3d}...", end="", flush=True)
                    if (k + 1) % 100 == 0:
                        print()
            if display and max_iter > 0:
                print()

            W_list[i], H_list[i] = W_i, H_i
            e[i] = betadiv(current_X, W_i @ H_i, beta)

        if display:
            print(f"Layer {i + 1} done.")

    return W_list, H_list, e


if __name__ == "__main__":
    print("--- Running Beta-NMF Example (beta=1, KL-Divergence) ---")

    # 1. Create a synthetic dataset
    m, n, r = 50, 40, 5
    np.random.seed(0)
    W_true = np.random.rand(m, r)
    H_true = np.random.rand(r, n)
    X = W_true @ H_true + 0.1 * np.random.rand(m, n)  # Add some noise

    # 2. Set options for the algorithm
    nmf_options = {
        "maxiter": 200,
        "beta": 1,  # Kullback-Leibler Divergence
        "display": 1,
        "accuracy": 1e-5,
    }

    # 3. Run the beta_nmf algorithm
    W, H, error_history = beta_nmf(X, r, nmf_options)

    # 4. Print results
    final_error = betadiv(X, W @ H, nmf_options["beta"])
    print(f"\nFactorization complete.")
    print(f"Final reconstruction error (beta=1): {final_error:.4f}")
    print(f"Shape of W: {W.shape}, Shape of H: {H.shape}")

    print("\n--- Running Multilayer NMF Example ---")

    # 1. Define ranks for each layer
    layer_ranks = [10, 5]  # X -> W1(m, 10), W1 -> W2(10, 5)

    # 2. Run the multilayer algorithm
    W_layers, H_layers, layer_errors = multilayer_klnmf(X, layer_ranks, nmf_options)

    # 3. Print results
    print(f"\nMultilayer factorization complete for {len(layer_ranks)} layers.")
    for i in range(len(layer_ranks)):
        print(f"Layer {i+1}:")
        print(
            f"  Shape of W[{i}]: {W_layers[i].shape}, Shape of H[{i}]: {H_layers[i].shape}"
        )
        print(f"  Reconstruction error at this layer: {layer_errors[i]:.4f}")
