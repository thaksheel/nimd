from sklearn.utils.validation import check_non_negative, check_random_state
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition._nmf import norm
import numpy as np
from typing import Literal
from scipy.sparse.linalg import eigs


def LRAnnlsHALSupdt(Y, Z, U, V, alphaparam=0.5, delta=0.01):
    """
    Solves min_{V >= 0} ||Y*Z - U*V||_F^2 using block-coordinate descent.

    Parameters:
    Y, Z       : Input matrices such that M = Y @ Z
    U, V       : U defines the NNLS problem, V is the initialization
    alphaparam : Controls number of iterations (default: 0.5)
    delta      : Convergence threshold (default: 0.01)

    Returns:
    V    : Updated matrix minimizing ||Y*Z - U*V||_F^2
    UtU  : U.T @ U
    UtM  : U.T @ (Y @ Z)
    """
    M = Y @ Z
    KX = np.sum(Y > 0) + np.sum(Z > 0)
    n = V.shape[1]
    m, r = U.shape
    maxiter = int(np.floor(1 + alphaparam * (KX + m * r) / (n * r + n)))

    UtU = U.T @ U
    UtM = U.T @ M

    eps0 = 0
    cnt = 1
    eps = 1

    while eps >= (delta**2) * eps0 and cnt <= maxiter:
        nodelta = 0
        for k in range(r):
            deltaV = (UtM[k, :] - UtU[k, :] @ V) / UtU[k, k]
            deltaV = np.maximum(deltaV, -V[k, :])
            V[k, :] += deltaV
            nodelta += np.sum(deltaV**2)
            if np.all(V[k, :] == 0):
                V[k, :] = 1e-16 * np.max(V)

        if cnt == 1:
            eps0 = nodelta
        eps = nodelta
        cnt += 1

    return V, UtU, UtM


def mySVD(X, ReducedDim=0):
    """
    Accelerated SVD approximation using eigen-decomposition.

    Parameters:
    X           : Input matrix (NumPy array)
    ReducedDim  : Target rank for approximation (optional)

    Returns:
    U, S, V     : Matrices such that X ≈ U @ S @ V.T
    """
    MAX_MATRIX_SIZE = 1600
    EIGVECTOR_RATIO = 0.1

    nSmp, mFea = X.shape

    if mFea / nSmp > 1.0713:
        ddata = X @ X.T
        ddata = np.maximum(ddata, ddata.T)

        dimMatrix = ddata.shape[0]
        if (
            ReducedDim > 0
            and dimMatrix > MAX_MATRIX_SIZE
            and ReducedDim < dimMatrix * EIGVECTOR_RATIO
        ):
            eigvalue, U = eigs(ddata, k=ReducedDim, which="LM")
        else:
            if hasattr(ddata, "tocsc"):  # check for sparse
                ddata = ddata.toarray()
            eigvalue, U = np.linalg.eigh(ddata, UPLO="U")
            idx = np.argsort(-eigvalue)
            eigvalue = eigvalue[idx]
            U = U[:, idx]

        maxEigValue = np.max(np.abs(eigvalue))
        eigIdx = np.where(np.abs(eigvalue) / maxEigValue < 1e-10)[0]
        eigvalue = np.delete(eigvalue, eigIdx)
        U = np.delete(U, eigIdx, axis=1)

        if ReducedDim > 0 and ReducedDim < len(eigvalue):
            eigvalue = eigvalue[:ReducedDim]
            U = U[:, :ReducedDim]

        eigvalue_Half = np.sqrt(eigvalue)
        S = np.diag(eigvalue_Half)

        eigvalue_MinusHalf = 1.0 / eigvalue_Half
        V = X.T @ (U * eigvalue_MinusHalf[np.newaxis, :])

    else:
        ddata = X.T @ X
        ddata = np.maximum(ddata, ddata.T)

        dimMatrix = ddata.shape[0]
        if (
            ReducedDim > 0
            and dimMatrix > MAX_MATRIX_SIZE
            and ReducedDim < dimMatrix * EIGVECTOR_RATIO
        ):
            eigvalue, V = eigs(ddata, k=ReducedDim, which="LM")
        else:
            if hasattr(ddata, "tocsc"):
                ddata = ddata.toarray()
            eigvalue, V = np.linalg.eigh(ddata, UPLO="U")
            idx = np.argsort(-eigvalue)
            eigvalue = eigvalue[idx]
            V = V[:, idx]

        maxEigValue = np.max(np.abs(eigvalue))
        eigIdx = np.where(np.abs(eigvalue) / maxEigValue < 1e-10)[0]
        eigvalue = np.delete(eigvalue, eigIdx)
        V = np.delete(V, eigIdx, axis=1)

        if ReducedDim > 0 and ReducedDim < len(eigvalue):
            eigvalue = eigvalue[:ReducedDim]
            V = V[:, :ReducedDim]

        eigvalue_Half = np.sqrt(eigvalue)
        S = np.diag(eigvalue_Half)

        eigvalue_MinusHalf = 1.0 / eigvalue_Half
        U = X @ (V * eigvalue_MinusHalf[np.newaxis, :])

    return U, S, V


def NNSVDLRC(X, r, delta=0.05, maxiter=20):
    """ref: https://github.com/5y3datif/NNSVD-LRC"""

    m, n = X.shape
    p = int(np.floor(r / 2 + 1))

    # Truncated SVD
    if isinstance(X, np.ndarray) and not np.any(np.isnan(X)):
        u, s, vt = mySVD(X, p)
    else:
        raise ValueError("Input matrix X must be a valid NumPy array.")

    Y = u @ np.sqrt(s)
    Z = np.sqrt(s) @ vt.T

    W = np.zeros((m, r))
    H = np.zeros((r, n))

    W[:, 0] = np.abs(Y[:, 0])
    H[0, :] = np.abs(Z[0, :])

    i, j = 1, 1
    while i < r:
        if i % 2 == 1:
            W[:, i] = np.maximum(Y[:, j], 0)
            H[i, :] = np.maximum(Z[j, :], 0)
        else:
            W[:, i] = np.maximum(-Y[:, j], 0)
            H[i, :] = np.maximum(-Z[j, :], 0)
            if i > 1:
                j += 1
        i += 1

    WtYZ = (W.T @ Y) @ Z
    WtW = W.T @ W
    HHt = H @ H.T
    scaling = np.sum(WtYZ * H) / np.sum(WtW * HHt)
    H *= np.sqrt(scaling)
    W *= np.sqrt(scaling)
    WtYZ *= np.sqrt(scaling)
    WtW *= scaling
    HHt *= scaling

    nX = np.sqrt(np.sum((Y.T @ Y) * (Z @ Z.T)))
    # NOTE: appending the absolute of a to prevent sqrt errors wit -ve values
    a = nX**2 - 2 * np.sum(WtYZ * H) + np.sum(WtW * HHt)
    # if a < 0:
    #     print(f"-ve a encountered in nnsdvlrc a={a}")
    e = [np.sqrt(abs(a)) / nX]

    k = 1
    while (k == 1 or e[-2] - e[-1] > delta * e[0]) and k <= maxiter:
        W = LRAnnlsHALSupdt(Z.T, Y.T, H.T, W.T)[0].T
        H, WtW, WtX = LRAnnlsHALSupdt(Y, Z, W, H)
        a = nX**2 - 2 * np.sum(WtX * H) + np.sum(WtW * (H @ H.T))
        # if a < 0:
        #     print(f"-ve a encountered in nnsdvlrc a={a}")
        e.append(np.sqrt(abs(a)) / nX)
        k += 1

    return W, H, Y, Z, e


def initialize_nmf(
    X: np.ndarray,
    n_components,
    init: str = Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"],
    eps=1e-6,
    random_state=None,
    perturb: bool = False,
    noise_level: float = None,
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
        if perturb:
            return perturb_init(W, H, eta=noise_level)
        else:
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
        if perturb:
            return perturb_init(W, H, eta=noise_level)
        else:
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
    if perturb:
        return perturb_init(W, H, eta=noise_level)
    else:
        return W, H


def perturb_init(W, H, eta, rng=0):
    """
    Controlled perturbation of W and H using NumPy.
    eta = noise level (0 = no noise)
    """
    rng = np.random.default_rng(rng)
    sW = np.std(W)
    sH = np.std(H)
    noiseW = rng.normal(loc=0.0, scale=sW * eta, size=W.shape)
    noiseH = rng.normal(loc=0.0, scale=sH * eta, size=H.shape)
    Wp = np.clip(W + noiseW, a_min=0, a_max=None)
    Hp = np.clip(H + noiseH, a_min=0, a_max=None)

    return Wp, Hp
