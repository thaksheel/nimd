import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import issparse
from scipy.linalg import svd
from scipy.optimize import linprog
from nmfs.NNLS import *


def LPinitSemiNMF(M, r, algolp=1):
    """
    Initialization for semi-NMF based on the SVD and linear programming.

    Parameters:
    M : ndarray
        m-by-n matrix
    r : int
        Factorization rank
    algolp : int, optional
        Solver used to solve the linear systems (default is 1 for linprog)

    Returns:
    Vlp : ndarray
        r-by-n nonnegative matrix such that U*Vlp approx M for some U
    yopt : ndarray
        Optimal solution vector
    epsiopt : float
        Optimal epsilon value
    """

    def linsys_semiNMF(V, epsi, algolp):
        r, n = V.shape
        if algolp == 1:
            c = np.zeros(r)
            A_ub = -(V + epsi).T
            b_ub = -np.ones(n)
            options = {"disp": False}
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, options=options)
            y = res.x
            eflag = 1 if res.success else -1
        else:
            raise NotImplementedError("CVX method is not implemented in this example.")
        return y, eflag

    U, S, V = svd(M)
    V = V[:r, :]
    I = np.sum(np.abs(V), axis=0)
    I = np.where(I > 1e-6 * np.mean(I))[0]
    V = V[:, I]

    for k in range(r):
        if np.abs(np.min(V[k, :])) > np.max(V[k, :]):
            V[k, :] = -V[k, :]

    y, eflag = linsys_semiNMF(V, 0, algolp)
    if eflag == 1:
        epsiopt = 0
        yopt = y
    else:
        epsimin = 0
        epsimax = np.max(np.abs(V))
        epsiopt = epsimax
        epsimax0 = epsimax
        epsi = (epsimin + epsimax) / 2
        while (epsimax - epsimin) / epsimax0 > 1e-3 or eflag == -1:
            y, eflag = linsys_semiNMF(V, epsi, algolp)
            if eflag == -1:
                epsimin = epsi
            else:
                epsimax = epsi
                epsiopt = epsi
                yopt = y
            epsi = (epsimin + epsimax) / 2

    x = (V + epsiopt).T @ yopt
    alpha = np.zeros((r, 1))
    for i in range(r):
        denom = x if np.all(x != 0) else x + 1e-12
        alpha[i, 0] = max(0, np.max(-V[i, :].T / denom))
    x = np.array([np.array([i]) for i in x])
    V = np.maximum(0, V + alpha @ x.T)
    Vlp = np.zeros((r, M.shape[1]))
    Vlp[:, I] = V

    return Vlp


def sign_flip(loads, X):
    """
    Adjusts the signs of the loadings to maximize the minimum entry.

    Parameters:
    loads : list of ndarrays
        List of loadings
    X : ndarray
        Data array

    Returns:
    sgns : ndarray
        Sign matrix
    loads : list of ndarrays
        Adjusted loadings
    """

    def handle_oddnumbers(Bcon):
        sgns = np.sign(Bcon)
        nb_neg = np.where(Bcon < 0)[0]
        min_val, index = np.min(np.abs(Bcon)), np.argmin(np.abs(Bcon))
        if Bcon[index] < 0:
            sgns[index] = -sgns[index]
        elif Bcon[index] > 0 and len(nb_neg) > 0:
            sgns[index] = -sgns[index]
        return sgns

    def subtract_otherfactors(X, loads, mode, factor):
        order = len(X.shape)
        x = np.moveaxis(X, mode, 0)
        loads = [loads[mode]] + loads[:mode] + loads[mode + 1 :]

        for m in range(order):
            loads[m] = np.delete(loads[m], factor, axis=1)

        M = outerm(loads)
        x = x - M
        return x

    def outerm(facts, lo=0, vect=0):
        order = len(facts)
        mwasize = [facts[i].shape[0] for i in range(order) if i != lo]
        nofac = facts[0].shape[1]
        mwa = np.zeros((np.prod(mwasize), nofac))

        for j in range(nofac):
            mwvect = facts[0][:, j]
            for i in range(1, order):
                if i != lo:
                    mwvect = np.outer(mwvect, facts[i][:, j]).flatten()
            mwa[:, j] = mwvect

        if vect != 1:
            mwa = np.sum(mwa, axis=1).reshape(mwasize)

        return mwa

    order = len(X.shape)
    F = [loads[i].shape[1] for i in range(order)]

    S = np.zeros((order, max(F)))
    for m in range(order):
        for f in range(F[m]):
            a = loads[m][:, f]
            a = a / np.dot(a, a)
            x = subtract_otherfactors(X, loads, m, f)
            s = np.array(
                [
                    np.sign(np.dot(a, x[:, i])) * (np.dot(a, x[:, i]) ** 2)
                    for i in range(x.shape[1])
                ]
            )
            S[m, f] = np.sum(s)

    sgns = np.sign(S)

    for f in range(F[0]):
        for i in range(sgns.shape[0]):
            se = np.sum(sgns[:, f] == -1)
            if se % 2 == 0:
                loads[i][:, f] *= sgns[i, f]
            else:
                sgns[:, f] = handle_oddnumbers(S[:, f])
                se = np.sum(sgns[:, f] == -1)
                if se % 2 == 0:
                    loads[i][:, f] *= sgns[i, f]
                else:
                    print("Something Wrong!!!")

    return sgns, loads


def SVDinitSemiNMF(M, r, init=1):
    """
    SVD-based initialization for semi-NMF.

    Parameters:
    M : ndarray
        m-by-n matrix
    r : int
        Factorization rank
    init : int, optional
        Different ways to flip the signs of the factors of the truncated SVD (default is 1)

    Returns:
    U : ndarray
        m-by-r matrix
    V : ndarray
        r-by-n nonnegative matrix
    """

    if r == 1:
        A, S, B = svds(M, r)
        B = B.T
        if np.sum(B > 0) < np.sum(B < 0):
            B = -B
        V = np.maximum(B, 0)
        U = np.dot(M, np.linalg.pinv(V))
    else:
        # A, S, B = svds(M, r - 1)
        A, S, B = svd(M)
        A = A[:, :r-1]
        S = S[:r-1]
        A = np.dot(A, np.diag(S))
        B = B[:r-1,:]

        # B = B.T
        m, n = M.shape
        if init == 1:  # Flip sign to maximize minimum entry
            for i in range(r - 1):
                if np.min(B[i, :]) < np.min(-B[i, :]):
                    B[i, :] = -B[i, :]
                    A[:, i] = -A[:, i]
        else:  # Using Bro et al. sign flip
            loads = [A, B.T]
            sgns, loads = sign_flip(loads, M)
            A = loads[0]
            B = loads[1].T

        if r == 2:
            U = np.hstack([A, -A])
        else:
            U = np.hstack([A, -np.sum(A, axis=1).reshape(-1, 1)])

        V = np.vstack([B, np.zeros((1, n))])
        if r >= 3:
            V = V - np.min(B, axis=0) * np.ones((r, 1))
        else:
            # TODO: incorrect dimension here for r<3 not sure how to fix this
            V = V - np.ones((r, 1)) * np.minimum(B, 0)

    return U, V


def linsys_semiNMF(V, epsi=0, algolp=1):
    """
    Check whether the following linear system: (V+epsi)'*y >= 1 is feasible.

    Parameters:
    V : ndarray
        Input matrix
    epsi : float, optional
        Small epsilon value to avoid division by zero (default is 0)
    algolp : int, optional
        Algorithm to use: 1 for linprog (default), other values for CVX (not implemented)

    Returns:
    y : ndarray
        Solution vector if the system is feasible
    eflag : int
        1 if the system is feasible, -1 otherwise
    """
    V = V.T
    r, n = V.shape

    if algolp == 1:
        # Using linprog
        c = np.zeros(r)
        A_ub = -(V + epsi).T
        b_ub = -np.ones(n)
        options = {"disp": False}
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, options=options)
        y = res.x
        eflag = 1 if res.success else -1
    else:
        # Using CVX (not implemented)
        raise NotImplementedError("CVX method is not implemented in this example.")
    return y, eflag


def seminonnegativerank(M):
    """
    Computes the semi-nonnegative rank and a corresponding factorization of matrix M: M = UV,
    where U has r columns and V >= 0 has r rows and r = semi-nonnegative rank of M.

    Parameters:
    M : ndarray
        m-by-n matrix

    Returns:
    seminnrank : int
        The (numerical) semi-nonnegative rank of M
    U : ndarray
        m-by-r matrix
    V : ndarray
        r-by-n matrix, V >= 0
    """

    r = np.linalg.matrix_rank(M)
    if not issparse(M):
        U, S, Vt = svd(M)
        V = Vt[:r, :].T
    else:
        U, S, Vt = svds(M, r)
        V = Vt.T
    y, eflag = linsys_semiNMF(V, 0)
    if eflag == 1:
        seminnrank = r
        V = LPinitSemiNMF(M, r)
        U = np.dot(M, np.linalg.pinv(V))
    else:
        seminnrank = r + 1
        U, V = SVDinitSemiNMF(M, r + 1)
    return seminnrank, U, V


def semiNMF(X, r, maxiter=100):
    """
    Solves semi-NMF: min_{W, H} ||X-WH||_F such that H >= 0.

    Parameters:
    X : ndarray
        m-by-n matrix
    r : int
        Rank for the approximation
    maxiter : int, optional
        Maximum number of iterations (default is 100)

    Returns:
    W : ndarray
        m-by-r matrix
    H : ndarray
        r-by-n matrix, H >= 0
    e : list
        Evolution of the error ||X-Xr||_F at each iteration
    """

    # Step 1: truncated SVD and check whether it is semi-nonnegative
    if np.prod(X.shape) < 1e8 and not issparse(X):
        u, s, v = svds(X, r)
        Xr = np.dot(u, np.dot(np.diag(s), v))
        seminnrank, W, H = seminonnegativerank(Xr)
    else:
        seminnrank = 0

    # Main step
    if seminnrank == r:
        e = np.linalg.norm(X - Xr, "fro")
        return W, H, e
    else:
        W, H = SVDinitSemiNMF(X, r)
        nX2 = np.linalg.norm(X, "fro") ** 2
        e = []
        for i in range(maxiter):
            W = np.dot(X, np.linalg.pinv(H))
            options = {"init": H}
            H, WTW, WTX = NNLS(W, X, options)
            # TODO: removed nargout since it was not doing anything
            error = nX2 - 2 * np.sum(WTX * H) + np.sum(WTW * np.dot(H, H.T))
            e.append(np.sqrt(max(0, error)))
        return W, H, e
