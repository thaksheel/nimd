import numpy as np
import time
from scipy import sparse


def compwiseprodsparselowrank(X, W, H, fun):
    """
    Given the sparse matrix X, and the low-rank factors (W,H),
    Compute Y = fun(X, W*H) only at the nonzero entries of X.
    """
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    i, j = X.nonzero()
    s = X.data
    m, n = X.shape
    Y = sparse.lil_matrix((m, n))
    N = len(s)
    for t in range(N):
        val = fun(X[i[t], j[t]], np.dot(W[i[t], :], H[:, j[t]]))
        Y[i[t], j[t]] = val
    return Y.tocsr()


def blockcompwiseprodsparselowrank(X, W, H, beta, nblocks):
    """
    Perform Y = compwiseprodsparselowrank(X,W,H,beta) by decomposing X into blocks.
    """
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    m, n = X.shape
    mi = int(np.ceil(m / nblocks))
    ni = int(np.ceil(n / nblocks))
    Y = sparse.lil_matrix((m, n))
    for i in range(nblocks):
        for j in range(nblocks):
            indi = slice(i * mi, min((i + 1) * mi, m))
            indj = slice(j * ni, min((j + 1) * ni, n))
            # Define the function for the component-wise operation
            fun = lambda x, y: x * (y**beta)
            Y_block = compwiseprodsparselowrank(
                X[indi, indj], W[indi, :], H[:, indj], fun
            )
            Y[indi, indj] = Y_block
    return Y.tocsr()


def blockrecursivecompwiseprodsparselowrank(X, W, H, fun, nnzparam=1e3):
    """
    Perform Y = compwiseprodsparselowrank(X,W,H,fun) by decomposing X in blocks.
    """
    # Ensure X is a scipy sparse matrix
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    m, n = X.shape
    Y = sparse.csr_matrix((m, n))

    # Base step
    if X.nnz < nnzparam:
        Y = compwiseprodsparselowrank(X, W, H, fun)  # Placeholder
    else:
        nblocks = 2
        mi = int(np.ceil(m / nblocks))
        ni = int(np.ceil(n / nblocks))
        for i in range(nblocks):
            for j in range(nblocks):
                indi = slice(i * mi, min((i + 1) * mi, m))
                indj = slice(j * ni, min((j + 1) * ni, n))
                Y_block = blockrecursivecompwiseprodsparselowrank(
                    X[indi, indj], W[indi, :], H[:, indj], fun, nnzparam
                )
                Y[indi, indj] = Y_block
    return Y


def betadiv(X, Y, beta):

    eps = np.finfo(float).eps

    if beta == 0:  # Itakura-Saito distance
        XsY = X / (Y + eps)
        Z = XsY - np.log(XsY + eps) - 1
        dist = np.sum(Z)
    elif beta == 1:  # Kullback-Leibler divergence
        Z = X * np.log(X / (Y + eps) + eps) - X + Y
        dist = np.sum(Z)
    else:  # Other beta divergences
        Z = (
            (X + eps) ** beta + (beta - 1) * (Y**beta) - beta * X * (Y ** (beta - 1))
        ) / (beta * (beta - 1))
        dist = np.sum(Z)
    return dist, Z 


def ND_MUbeta(X, W, H, beta, return_error=True):
    """
    Numerator and denominator in the MU for the beta-divergence for the factor H.
    Returns N, D, and optionally e (the error before the update of H).
    """
    epsilon = 2 ** (-52)
    e = None

    if beta == 1:  # Kullback-Leibler divergence
        if sparse.issparse(X):
            # Define the xdy function for KL: x / y
            def xdy(x, y):
                return x / (y + epsilon)

            XdWH = blockrecursivecompwiseprodsparselowrank(X, W, H, xdy)
        else:
            XdWH = X / (np.dot(W, H) + epsilon)
        N = np.dot(W.T, XdWH)
        D = np.tile(np.sum(W, axis=0)[:, np.newaxis], (1, X.shape[1])) + epsilon
        if return_error:
            Xnnz = X[X > 0]
            XdWHnnz = XdWH[X > 0]
            e = np.sum(Xnnz * np.log(XdWHnnz + epsilon)) - np.sum(X) + np.sum(D * H)
    elif beta == 2:  # Frobenius norm
        N = np.dot(W.T, X)
        WtW = np.dot(W.T, W)
        D = np.dot(WtW, H) + epsilon
        if return_error:
            e = 0.5 * (
                np.linalg.norm(X, "fro") ** 2
                - 2 * np.sum(N * H)
                + np.sum(WtW * np.dot(H, H.T))
            )
    else:  # Other beta divergences -- cannot handle sparse
        WH = np.dot(W, H) + epsilon
        N = np.dot(W.T, ((WH + epsilon) ** (beta - 2)) * X)
        D = np.dot(W.T, (WH + epsilon) ** (beta - 1))
        if return_error:
            e, _ = betadiv(X, WH, beta)
    if return_error:
        return N, D, e
    else:
        return N, D


def MUbeta(X, W, H, beta, epsilon=None, return_error=False):

    if epsilon is None:
        epsilon = 2 ** (-52)
    eps = np.finfo(float).eps

    # Call ND_MUbeta (placeholder)
    if return_error:
        N, D, e = ND_MUbeta(X, W, H, beta, return_error)
    else:
        N, D = ND_MUbeta(X, W, H, beta, return_error)
        e = None

    # Use the gamma power to ensure monotonicity
    if 1 <= beta <= 2:
        H = np.maximum(epsilon, H * (N / (D + eps)))
    else:
        if beta < 1:
            gamma = 1 / (2 - beta)
        else:
            gamma = 1 / (beta - 1)
        H = np.maximum(epsilon, H * ((N / (D + eps)) ** gamma))

    if return_error:
        return H, e
    else:
        return H


def betaNMF(X, r, options=None):
    # Start timer
    time0 = time.process_time()
    if np.min(X) < 0:
        raise ValueError("X should be nonnegative.")
    m, n = X.shape

    # Handle options and defaults
    if options is None:
        options = {}
    options.setdefault("maxiter", 500)
    options.setdefault("timemax", 60)
    options.setdefault("beta", 1)
    options.setdefault(
        "extrapol", "nesterov" if 1 <= options["beta"] <= 2 else "noextrap"
    )
    options.setdefault("epsilon", np.finfo(float).eps)
    options.setdefault("accuracy", 1e-4)
    options.setdefault("W", np.random.rand(m, r))
    options.setdefault("H", np.random.rand(r, n))
    options.setdefault("display", 0)

    if options["beta"] == 2:
        print(
            "Warning: Since beta=2, you might want to use more efficient algorithms; see FroNMF.py"
        )

    W = np.maximum(options["epsilon"], options["W"])
    H = np.maximum(options["epsilon"], options["H"])

    i = 1
    cpuinit = time.process_time()
    # Scaling: the maximum entry in each column of W is 1
    for k in range(r):
        mxk = np.max(W[:, k])
        W[:, k] = W[:, k] / mxk
        H[k, :] = H[k, :] * mxk

    # Keep previous iterate in memory for extrapolation
    Hp = H.copy()
    Wp = W.copy()
    cparam = 1e30
    nutprev = 1
    e = []
    t = []
    if options["display"] == 1:
        print("Iterations:")
        cntdis = 0
        mintime = 0.1  # display parameters

    timei = time.process_time()
    while (
        i <= options["maxiter"]
        and time.process_time() <= cpuinit + options["timemax"]
        and (
            i <= 12
            or len(e) <= 2
            or abs(e[-1][0] - e[-11][0]) > options["accuracy"] * abs(e[-1][0])
            if len(e) > 11
            else True
        )
    ):
        # Compute the extrapolated points
        stepH = np.maximum(0, H - Hp)
        if options["extrapol"] == "nesterov":
            nut = 0.5 * (1 + np.sqrt(1 + 4 * nutprev**2))
            extrapolparam = (nutprev - 1) / nut
            nutprev = nut
        elif options["extrapol"] == "ptsengv1":
            extrapolparam = (i - 1) / i
        elif options["extrapol"] == "ptsengv2":
            extrapolparam = i / (i + 1)
        elif options["extrapol"] == "noextrap":
            extrapolparam = 0
        else:
            extrapolparam = 0

        normstepH = np.linalg.norm(stepH, "fro")
        if normstepH == 0:
            extrapH = extrapolparam
        else:
            extrapH = min(extrapolparam, cparam / i ** (1.5 / 2) / normstepH)
        He = H + extrapH * stepH

        stepW = np.maximum(0, W - Wp)
        normstepW = np.linalg.norm(stepW, "fro")
        if normstepW == 0:
            extrapW = extrapolparam
        else:
            extrapW = min(extrapolparam, cparam / i ** (1.5 / 2) / normstepW)
        We = W + extrapW * stepW

        # Keep previous iterate in memory for extrapolation
        Hp = H.copy()
        Wp = W.copy()

        # Update of W and H with MU with extrapolation
        H = MUbeta(X, W, He, options["beta"], options["epsilon"])  # Placeholder
        W = MUbeta(X.T, H.T, We.T, options["beta"], options["epsilon"]).T  # Placeholder

        # Error: this time should not be taken into account in the computational cost of the method
        timeei = time.process_time()
        e.append(betadiv(X, np.dot(W, H), options["beta"]))  # Placeholder
        timeei = time.process_time() - timeei
        if i == 1:
            t.append(time.process_time() - time0 - timeei)
        else:
            t.append(t[-1] + (time.process_time() - timei) - timeei)

        # Display evolution of the iterations
        if options["display"] == 1:
            if time.process_time() - time0 >= mintime:
                print(f"{i}...", end="", flush=True)
                mintime *= 2
                cntdis += 1
                if cntdis % 10 == 0:
                    print()
        i += 1
        timei = time.process_time()

    if options["display"] == 1:
        print()
    return W, H, e, t
