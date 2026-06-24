import numpy as np
from scipy.sparse.linalg import svds
from numpy.linalg import svd
import pandas as pd

from scipy.sparse.linalg import eigs
import scipy.linalg as la


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
        if ReducedDim > 0 and dimMatrix > MAX_MATRIX_SIZE and ReducedDim < dimMatrix * EIGVECTOR_RATIO:
            eigvalue, U = eigs(ddata, k=ReducedDim, which='LM')
        else:
            if hasattr(ddata, "tocsc"):  # check for sparse
                ddata = ddata.toarray()
            eigvalue, U = np.linalg.eigh(ddata, UPLO='U')
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
        if ReducedDim > 0 and dimMatrix > MAX_MATRIX_SIZE and ReducedDim < dimMatrix * EIGVECTOR_RATIO:
            eigvalue, V = eigs(ddata, k=ReducedDim, which='LM')
        else:
            if hasattr(ddata, "tocsc"):
                ddata = ddata.toarray()
            eigvalue, V = np.linalg.eigh(ddata, UPLO='U')
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
    m, n = X.shape
    p = int(np.floor(r / 2 + 1))

    # Truncated SVD
    if isinstance(X, np.ndarray) and not np.any(np.isnan(X)):
        # if np.count_nonzero(X) / X.size < 0.5:
        #     u, s, vt = svd(X, hermitian=False, full_matrices=True)
        #     u = u[:, :r]
        #     s = s[:r]
        #     vt = vt[:r,:]
        # else:
        #     u, s, vt = mySVD(X, p)
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
    e = [np.sqrt(nX**2 - 2 * np.sum(WtYZ * H) + np.sum(WtW * HHt)) / nX]

    k = 1
    while (k == 1 or e[-2] - e[-1] > delta * e[0]) and k <= maxiter:
        W = LRAnnlsHALSupdt(Z.T, Y.T, H.T, W.T)[0].T
        H, WtW, WtX = LRAnnlsHALSupdt(Y, Z, W, H)
        e.append(np.sqrt(nX**2 - 2 * np.sum(WtX * H) + np.sum(WtW * (H @ H.T))) / nX)
        k += 1

    return W, H, Y, Z, e
