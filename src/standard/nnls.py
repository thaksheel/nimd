import numpy as np
from scipy.sparse import spdiags


def solve_normal_eq_comb(AtA, AtB, PassSet=None):
    """
    Solve normal equations using combinatorial grouping.
    Converted from MATLAB implementation.
    """
    iter_count = 0

    if PassSet is None or PassSet.size == 0 or np.all(PassSet):
        Z = np.linalg.solve(AtA, AtB)
        iter_count += 1
    else:
        Z = np.zeros_like(AtB)
        n, k1 = PassSet.shape

        if k1 == 1:
            # Handle single column case
            idx = PassSet[:, 0]
            Z[idx, 0] = np.linalg.solve(AtA[np.ix_(idx, idx)], AtB[idx, 0])
            iter_count += 1
        else:
            # Transpose and sort rows of PassSet
            sorted_pass_set = PassSet.T[np.lexsort(PassSet[::-1, :])]
            sort_ix = np.lexsort(PassSet[::-1, :])
            diff_rows = np.any(np.diff(sorted_pass_set, axis=0), axis=1)
            break_ix = np.concatenate(([0], np.where(diff_rows)[0] + 1, [k1]))

            for k in range(len(break_ix) - 1):
                cols = sort_ix[break_ix[k]:break_ix[k+1]]
                vars_ = PassSet[:, sort_ix[break_ix[k]]]
                idx = np.where(vars_)[0]
                Z[np.ix_(idx, cols)] = np.linalg.solve(AtA[np.ix_(idx, idx)], AtB[np.ix_(idx, cols)])
                iter_count += 1

    return Z, iter_count


def nnlsm_activeset(A, B, overwrite=False, is_input_prod=False, init=None):
    if is_input_prod:
        AtA, AtB = A, B
    else:
        AtA, AtB = A.T @ A, A.T @ B

    n, k = AtB.shape
    MAX_ITER = n * 5

    if overwrite:
        X, iter_count = solve_normal_eq_comb(AtA, AtB)
        PassSet = X > 0
        NotOptSet = np.any(X < 0, axis=0)
    else:
        if init is None:
            X = np.zeros((n, k))
            PassSet = np.zeros((n, k), dtype=bool)
            NotOptSet = np.ones(k, dtype=bool)
        else:
            X = init
            PassSet = X > 0
            NotOptSet = np.ones(k, dtype=bool)
        iter_count = 0

    Y = np.zeros((n, k))
    Y[:, ~NotOptSet] = AtA @ X[:, ~NotOptSet] - AtB[:, ~NotOptSet]
    NotOptCols = np.where(NotOptSet)[0]

    bigIter = 0
    success = 1

    while NotOptCols.size > 0:
        bigIter += 1
        if MAX_ITER > 0 and bigIter > MAX_ITER:
            success = 0
            break

        Z, subiter = solve_normal_eq_comb(
            AtA, AtB[:, NotOptCols], PassSet[:, NotOptCols]
        )
        iter_count += subiter

        InfeaSubSet = Z < 0
        InfeaSubCols = np.where(np.any(InfeaSubSet, axis=0))[0]
        FeaSubCols = np.where(np.all(~InfeaSubSet, axis=0))[0]

        if InfeaSubCols.size > 0:
            ZInfea = Z[:, InfeaSubCols]
            InfeaCols = NotOptCols[InfeaSubCols]
            Alpha = np.full((n, len(InfeaSubCols)), np.inf)

            i, j = np.where(InfeaSubSet[:, InfeaSubCols])
            Alpha[i, j] = X[i, InfeaCols[j]] / (X[i, InfeaCols[j]] - ZInfea[i, j])

            minVal = np.min(Alpha, axis=0)
            X[:, InfeaCols] += minVal * (ZInfea - X[:, InfeaCols])
            minIx = np.argmin(Alpha, axis=0)

            X[minIx, InfeaCols] = 0
            PassSet[minIx, InfeaCols] = False

        if FeaSubCols.size > 0:
            FeaCols = NotOptCols[FeaSubCols]
            X[:, FeaCols] = Z[:, FeaSubCols]
            Y[:, FeaCols] = AtA @ X[:, FeaCols] - AtB[:, FeaCols]

            NotOptSubSet = (Y[:, FeaCols] < 0) & ~PassSet[:, FeaCols]
            NewOptCols = FeaCols[np.all(~NotOptSubSet, axis=0)]
            UpdateNotOptCols = FeaCols[np.any(NotOptSubSet, axis=0)]

            if UpdateNotOptCols.size > 0:
                minIx = np.argmin(
                    Y[:, UpdateNotOptCols] * ~PassSet[:, UpdateNotOptCols], axis=0
                )
                PassSet[minIx, UpdateNotOptCols] = True

            NotOptSet[NewOptCols] = False
            NotOptCols = np.where(NotOptSet)[0]

    return X, Y, AtA, AtB, iter_count, success


def nnlsm_blockpivot(A, B, isInputProd=False, init=None):
    """
    Non-Negative Least Squares using Block Principal Pivoting.
    Converted from MATLAB implementation.
    """
    if isInputProd:
        AtA, AtB = A, B
    else:
        AtA, AtB = A.T @ A, A.T @ B

    n, k = AtB.shape
    MAX_ITER = n * 5
    X = np.zeros((n, k))

    if init is None or init.size == 0:
        Y = -AtB
        PassiveSet = np.zeros((n, k), dtype=bool)
        iter_count = 0
    else:
        PassiveSet = init > 0
        X, iter_count = solve_normal_eq_comb(AtA, AtB, PassiveSet)
        Y = AtA @ X - AtB

    pbar = 3
    P = np.full(k, pbar)
    Ninf = np.full(k, n + 1)
    iter_count = 0

    NonOptSet = (Y < 0) & ~PassiveSet
    InfeaSet = (X < 0) & PassiveSet
    NotGood = NonOptSet.sum(axis=0) + InfeaSet.sum(axis=0)
    NotOptCols = NotGood > 0

    bigIter = 0
    success = True

    while np.any(NotOptCols):
        bigIter += 1
        if MAX_ITER > 0 and bigIter > MAX_ITER:
            success = False
            break

        Cols1 = NotOptCols & (NotGood < Ninf)
        Cols2 = NotOptCols & (NotGood >= Ninf) & (P >= 1)
        Cols3Ix = np.where(NotOptCols & ~Cols1 & ~Cols2)[0]

        if np.any(Cols1):
            P[Cols1] = pbar
            Ninf[Cols1] = NotGood[Cols1]
            PassiveSet[:, Cols1] |= NonOptSet[:, Cols1]
            PassiveSet[:, Cols1] &= ~InfeaSet[:, Cols1]

        if np.any(Cols2):
            P[Cols2] -= 1
            PassiveSet[:, Cols2] |= NonOptSet[:, Cols2]
            PassiveSet[:, Cols2] &= ~InfeaSet[:, Cols2]

        for Ix in Cols3Ix:
            toChange = np.max(np.where(NonOptSet[:, Ix] | InfeaSet[:, Ix]))
            PassiveSet[toChange, Ix] = not PassiveSet[toChange, Ix]

        NotOptMask = np.tile(NotOptCols, (n, 1))
        X[:, NotOptCols], subiter = solve_normal_eq_comb(
            AtA, AtB[:, NotOptCols], PassiveSet[:, NotOptCols]
        )
        iter_count += subiter
        X[np.abs(X) < 1e-12] = 0  # Numerical stability
        Y[:, NotOptCols] = AtA @ X[:, NotOptCols] - AtB[:, NotOptCols]
        Y[np.abs(Y) < 1e-12] = 0  # Numerical stability

        NonOptSet = NotOptMask & (Y < 0) & ~PassiveSet
        InfeaSet = NotOptMask & (X < 0) & PassiveSet
        NotGood = NonOptSet.sum(axis=0) + InfeaSet.sum(axis=0)
        NotOptCols = NotGood > 0

    return X, Y, AtA, AtB


def simplex_proj(H):
    """Projects the columns of H onto the simplex {x | x >= 0, sum(x) <= 1}"""
    return np.maximum(H, 0) / np.maximum(1, np.sum(H, axis=0))


def simplex_col_proj(H):
    """Projects the columns of H onto the simplex {x | x >= 0, sum(x) = 1}"""
    return H / np.sum(H, axis=0, keepdims=True)


def nnls_fpgm(X, W, options=None):
    """
    Computes an approximate solution to the nonnegative least squares problem using a fast gradient method.
    """
    if options is None:
        options = {}

    delta = options.get("delta", 1e-6)
    inneriter = options.get("inneriter", 500)
    proj = options.get("proj", 0)
    alpha0 = options.get("alpha0", 0.05)

    W = np.array(W, dtype=float)
    WtW = W.T @ W
    WtX = W.T @ X

    H = options.get("init", np.maximum(np.linalg.lstsq(W, X, rcond=None)[0], 0))

    L = np.linalg.norm(WtW, 2)
    Y = H.copy()
    alpha = [alpha0]

    if proj == 1:
        H = simplex_proj(H)
    elif proj == 0:
        H = np.maximum(H, 0)
    elif proj == 2:
        H = simplex_col_proj(H.T).T
    elif proj == 3:
        H = simplex_col_proj(H)

    eps0, eps = 0, 1
    i = 0

    while i < inneriter and eps >= delta * eps0:
        Hp = H.copy()

        alpha.append(
            (np.sqrt(alpha[-1] ** 4 + 4 * alpha[-1] ** 2) - alpha[-1] ** 2) / 2
        )
        beta = alpha[-2] * (1 - alpha[-2]) / (alpha[-2] ** 2 + alpha[-1])

        H = Y - (WtW @ Y - WtX) / L

        if proj == 1:
            H = simplex_proj(H)
        elif proj == 0:
            H = np.maximum(H, 0)
        elif proj == 2:
            H = simplex_col_proj(H.T).T
        elif proj == 3:
            H = simplex_col_proj(H)

        Y = H + beta * (H - Hp)

        if i == 0:
            eps0 = np.linalg.norm(H - Hp, "fro")
        eps = np.linalg.norm(H - Hp, "fro")

        i += 1

    return H, WtW, WtX


def nnls_MU(X, W, options=None):
    """
    Computes an approximate solution of the following nonnegative least squares problem (NNLS)

            min_{H >= 0} ||X-WH||_F^2

    using multiplicative updates.

    Parameters:
    X : numpy array
        Input matrix.
    W : numpy array
        Basis matrix.
    options : dict, optional
        Dictionary containing options for the algorithm.

    Returns:
    H : numpy array
        Coefficient matrix.
    WtW : numpy array
        Precomputed W.T @ W.
    WtX : numpy array
        Precomputed W.T @ X.
    """

    if options is None:
        options = {}
    if "delta" not in options:
        options["delta"] = 1e-6  # Stopping condition
    if "inneriter" not in options:
        options["inneriter"] = 500
    if "epsilon" not in options:
        options["epsilon"] = np.finfo(float).eps

    W = np.array(W, dtype=float)
    X = np.array(X, dtype=float)
    m, n = X.shape
    m, r = W.shape
    WtW = W.T @ W
    WtX = W.T @ X

    if np.min(X) < 0:
        print("Warning: The matrix X should be nonnegative. Zero entries set to 0.")
        X = np.maximum(X, 0)
    if np.min(W) < 0:
        print("Warning: The matrix W should be nonnegative. Zero entries set to 0.")
        W = np.maximum(W, 0)
    if "init" not in options or options["init"] is None:
        H = nnls_init(X, W, WtW, WtX)
    else:
        H = np.array(options["init"], dtype=float)
        if np.min(H) == 0:
            H[H == 0] = 0.001 * np.max(H)
    if np.min(H) < 0:
        print("Warning: The matrix H should be nonnegative. Zero entries set to 0.001.")
        H = np.maximum(H, 0.001)

    eps0 = 0
    cnt = 1
    eps = 1
    while eps >= options["delta"] * eps0 and cnt <= options["inneriter"]:
        Hp = H.copy()
        H = np.maximum(options["epsilon"], H * (WtX / (WtW @ H)))
        if cnt == 1:
            eps0 = np.linalg.norm(Hp - H, "fro")
        eps = np.linalg.norm(Hp - H, "fro")
        cnt += 1

    return H, WtW, WtX


def pinv(A, tol=None):
    """
    Pseudoinverse.

    Parameters:
    A : numpy array
        Input matrix.
    tol : float, optional
        Tolerance for singular values. Singular values less than or equal to tol are treated as zero.

    Returns:
    X : numpy array
        Pseudoinverse of the input matrix A.
    """

    if A.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")

    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    if np.isnan(s[0]):
        return np.full((A.shape[1], A.shape[0]), np.nan, dtype=A.dtype)

    if tol is None:
        tol = max(A.shape) * np.finfo(A.dtype).eps * np.linalg.norm(A, ord=2)
    elif not (isinstance(tol, (int, float)) and np.isreal(tol) and np.isscalar(tol)):
        raise ValueError("Invalid second argument for tolerance.")

    r = np.sum(s > tol)
    if r == 0:
        return np.zeros((A.shape[1], A.shape[0]), dtype=A.dtype)

    s_inv = np.zeros_like(s)
    s_inv[:r] = 1 / s[:r]

    X = (Vt.T[:, :r] * s_inv[:r]) @ U.T[:r, :]

    return X


def SNPA(X, r, options=None):
    """
    Successive Nonnegative Projection Algorithm (variant with f(.) = ||.||^2)

    Parameters:
    X : numpy array
        Input m-by-n matrix.
    r : int
        Number of columns to be extracted.
    options : dict, optional
        Dictionary containing options for the algorithm.

    Returns:
    K : list
        Index set of the extracted columns.
    H : numpy array
        Optimal weights.
    """

    m, n = X.shape

    # Options
    if options is None:
        options = {}
    if "normalize" not in options:
        options["normalize"] = 0
    if "display" not in options:
        options["display"] = 1
    if "maxitn" not in options:
        options["maxitn"] = 200
    if "relerr" not in options:
        options["relerr"] = 1e-6
    if "proj" not in options:
        options["proj"] = 0

    if options["normalize"] == 1:
        # Normalization of the columns of X so that they sum to one
        D = spdiags((np.sum(X, axis=0) + 1e-16) ** -1, 0, n, n)
        X = X @ D

    # Initialization
    normX0 = np.sum(X**2, axis=0)
    nXmax = np.max(normX0)
    i = 1
    normR = normX0
    XtUK = []
    UKtUK = []
    K = []
    U = np.zeros((m, r))

    if options["display"] == 1:
        print("Extraction of the indices by SNPA:")

    # Main loop
    while i <= r and np.sqrt(np.max(normR) / nXmax) > options["relerr"]:
        # Select the column of the residual R with largest l2-norm
        b = np.argmax(normR)
        # Check ties up to 1e-6 precision
        b = np.where((np.max(normR) - normR) / np.max(normR) <= 1e-6)[0]
        # In case of a tie, select column with largest norm of the input matrix X
        if len(b) > 1:
            b = b[np.argmax(normX0[b])]

        # Update the index set, and extracted column
        K.append(b)
        U[:, i - 1] = X[:, b]

        # Update XtUK
        XtUK.append(X.T @ U[:, i - 1])

        # Update UKtUK
        if i == 1:
            UtUi = []
        else:
            UtUi = U[:, : i - 1].T @ U[:, i - 1]
        UKtUK.append(np.hstack([UtUi, U[:, i - 1].T @ U[:, i - 1]]))

        # Update residual
        if i == 1:
            # Fast gradient method for min_{y in Delta} ||M(:,i)-M(:,J)y||
            H = nnls_fpgm(X, X[:, K], options)
        else:
            H[:, K[i - 1]] = 0
            h = np.zeros((1, n))
            h[0, K[i - 1]] = 1
            H = np.vstack([H, h])
            options["init"] = H
            H = nnls_fpgm(X, X[:, K], options)

        # Update the norm of the columns of the residual without computing it explicitly
        if i == 1:
            normR = normX0 - 2 * (np.array(XtUK).T * H) + (H * (np.array(UKtUK) @ H))
        else:
            normR = (
                normX0
                - 2 * np.sum(np.array(XtUK).T * H, axis=1)
                + np.sum(H * (np.array(UKtUK) @ H), axis=1)
            )

        if options["display"] == 1:
            print(f"{i}...", end="")
            if i % 10 == 0:
                print("\n")

        i += 1

    if options["display"] == 1:
        print("\n")

    return K, H


def alternatingONMF(X, r, options=None):
    """
    Solves ONMF based on the two-block coordinate descent (2-BCD) method to solve the NMF problem:

    min_{W and H} || X - WH ||_F^2 such that H >= 0 and HH^T = I_r.

    Parameters:
    X : numpy array
        Input m-by-n matrix.
    r : int
        Factorization rank.
    options : dict, optional
        Dictionary containing options for the algorithm.

    Returns:
    W : numpy array
        Basis matrix.
    H : numpy array
        Coefficient matrix.
    e : list
        Relative error of the iterates (W, H).
    """

    if options is None:
        options = {}
    if "display" not in options:
        options["display"] = 1
    if "maxiter" not in options:
        options["maxiter"] = 100

    # Initializations
    m, n = X.shape
    if "init" in options:
        if options["init"].shape == (m, r):
            if options["display"] == 1:
                print("Initialization provided by the user.")
            W = options["init"]
        elif options["init"] == 1:
            if options["display"] == 1:
                print("Randomized initialization.")
            H = np.hstack([np.eye(r), np.random.rand(r, n - r)])
            H = H[:, np.random.permutation(n)]
            a = np.max(H, axis=0)
            H = (H >= a).astype(float)
            norm2h = np.sqrt(np.sum(H.T**2, axis=1)) + 1e-16
            H = (1.0 / norm2h[:, np.newaxis]) * H
            W = X @ H.T
    else:
        if options["display"] == 1:
            print("Initialization by SNPA:")
        K = SNPA(X, r, {"display": options["display"]})
        W = X[:, K]
        if len(K) < r:
            raise ValueError(
                "SNPA was not able to extract r indices. This means that your data set does not even have r extreme rays."
            )

    if "delta" not in options:
        options["delta"] = 1e-6

    normX2 = np.sum(X**2)
    e = []
    k = 1
    if options["display"] == 1:
        print("Iteration number and relative error of ONMF iterates:")

    while k <= options["maxiter"] and (k <= 3 or abs(e[-1] - e[-2]) > options["delta"]):
        # H = argmin_H ||X-WH||_F, H >= 0, rows H orthogonal up to a scaling of the rows of H
        H, _ = orthNNLS(X, W)
        # Normalize rows of H
        norm2h = np.sqrt(np.sum(H.T**2, axis=1)) + 1e-16
        H = (1.0 / norm2h[:, np.newaxis]) * H
        # W = argmin_W ||X-WH||_F = X*H'; This is a more efficient implementation than before
        for i in range(r):
            Ki = np.where(H[i, :] > 0)[0]
            W[:, i] = X[:, Ki] @ H[i, Ki].T

        # Compute relative error
        e.append(np.sqrt((normX2 - np.sum(W**2)) / normX2))
        if options["display"] == 1:
            if e[-1] < 1e-4:
                print(f"{k}: {100 * e[-1]:.1e}...", end="")
            else:
                print(f"{k}: {100 * e[-1]:.2f}...", end="")
            if k % 10 == 0:
                print("\n")
        k += 1

    if options["display"] == 1:
        print("\n")

    return W, H, e


def orthNNLS(X, W):
    """
    Solves the orthogonal nonnegative least squares problem:

    min_{H >= 0 and HH^T is a diagonal matrix} ||X-WH||_F^2

    Parameters:
    X : numpy array
        Input matrix.
    W : numpy array
        Basis matrix.

    Returns:
    H : numpy array
        Coefficient matrix.
    norm2v : numpy array
        Norm of the columns of W.
    """

    m, n = X.shape
    m, r = W.shape

    # Normalize columns of W
    norm2w = np.sqrt(np.sum(W**2, axis=0))
    Wn = W / (norm2w + 1e-16)

    A = X.T @ Wn  # n by r matrix of "angles" between the columns of W and X
    b = np.argmax(A, axis=1)  # best column of W to approx. each column of X
    H = np.zeros((r, n))

    # Assign the optimal weights to H(b[i],i) > 0
    for k in range(r):
        Kk = np.where(b == k)[0]
        sizeclustk = len(Kk)
        H[k, Kk] = (Wn[:, k].T @ X[:, Kk]) / norm2w[k]

    # Deal with empty clusters (happens relatively rarely)
    emptyclust = np.where(np.sum(H, axis=1) == 0)[0]
    for k in emptyclust:
        # split largest cluster in two using ONMF itself
        maxclus = np.argmax(np.sum(H, axis=1))
        print("Warning: Empty cluster --> largest cluster split in two")
        optionssplit = {"display": 0}
        Ws, Hs = alternatingONMF(X[:, Kk[maxclus]], 2, optionssplit)

        # First cluster
        W[:, maxclus] = Ws[:, 0]
        H[maxclus, Kk[maxclus]] = Hs[0, :]

        # Second cluster
        W[:, k] = Ws[:, 1]
        H[k, Kk[maxclus]] = Hs[1, :]

        # Update clusters
        Kk[maxclus] = np.where(H[maxclus, :] > 0)[0]
        Kk[k] = np.where(H[k, :] > 0)[0]

    return H, norm2w


def nnls_init(X, W, WtW, WtX):
    """
    Initialize H for NNLS.

    Parameters:
    X : numpy array
        Input matrix.
    W : numpy array
        Basis matrix.
    WtW : numpy array
        Precomputed W.T @ W.
    WtX : numpy array
        Precomputed W.T @ X.

    Returns:
    H : numpy array
        Initialized coefficient matrix.
    """

    if np.linalg.cond(W) > 1e6:
        # Assign each column of X to the closest column of W in terms of angle
        H, _ = orthNNLS(X, W)
    else:
        # Projected LS solution + scaling
        if isinstance(X, np.ndarray) and np.any(X):
            H = np.maximum(0, pinv(W) @ X)
        else:
            H = np.maximum(0, np.linalg.lstsq(W, X, rcond=None)[0])

        # Scale
        alpha = np.sum(H * WtX) / np.sum(WtW * (H @ H.T))
        H *= alpha

    # Check that no rows of H are zeros
    zerow = np.where(np.sum(H, axis=1) == 0)[0]
    H[zerow, :] = 0.001 * np.max(H) * np.random.rand(len(zerow), H.shape[1])

    return H


def nnls_ADMM(X, W, options=None):
    """
    Computes an approximate solution of the following nonnegative least squares problem (NNLS):

            min_{H >= 0} ||X-WH||_F^2

    with ADMM, tackling the Lagrangian:

    L(H,Y,Z) = ||X-WH||_F^2 + rho/2 ||H-Y||_F^2 + <Z, H-Y>

    via alternating optimization.

    Parameters:
    X : numpy array
        Input matrix.
    W : numpy array
        Basis matrix.
    options : dict, optional
        Dictionary containing options for the algorithm.

    Returns:
    H : numpy array
        Coefficient matrix.
    WtW : numpy array
        Precomputed W.T @ W.
    WtX : numpy array
        Precomputed W.T @ X.
    """

    if options is None:
        options = {}
    if "delta" not in options:
        options["delta"] = 1e-6  # Stopping condition
    if "inneriter" not in options:
        options["inneriter"] = 500

    W = np.array(W, dtype=float)
    X = np.array(X, dtype=float)
    m, n = X.shape
    m, r = W.shape
    WtW = W.T @ W
    WtX = W.T @ X

    # If no initial matrices are provided, H is initialized as follows:
    if "init" not in options or options["init"] is None:
        H = nnls_init(X, W, WtW, WtX)
    else:
        H = options["init"]

    # ADMM parameter
    if "rho" not in options:
        rho = np.trace(WtW) / r
    else:
        rho = options["rho"]

    # Precompute the inverse
    invWtWrhoI = np.linalg.inv(WtW + rho * np.eye(r))
    invWtWrhoIWtX = invWtWrhoI @ WtX

    # Initialize Y and Z to zero
    Y = np.zeros((r, n))
    Z = np.zeros((r, n))
    cnt = 1
    Hp = np.inf

    while (
        np.linalg.norm(H - Y, "fro") > options["delta"] * np.linalg.norm(H, "fro")
        or np.linalg.norm(H - Hp, "fro") > options["delta"] * np.linalg.norm(H, "fro")
    ) and cnt <= options["inneriter"]:
        Hp = H.copy()
        Y = np.maximum(0, H + Z / rho)
        H = invWtWrhoIWtX + invWtWrhoI @ (rho * Y - Z)
        Z = Z + rho * (H - Y)
        cnt += 1

    # Final projection to have H feasible
    H = np.maximum(H, 0)

    return H, WtW, WtX


def nnls_HALSupdt(X, W, options=None):
    """
    Computes an approximate solution of the following nonnegative least squares problem (NNLS):

            min_{H >= 0} ||X-WH||_F^2

    with an exact block-coordinate descent scheme.

    Parameters:
    X : numpy array
        Input matrix.
    W : numpy array
        Basis matrix.
    options : dict, optional
        Dictionary containing options for the algorithm.

    Returns:
    H : numpy array
        Coefficient matrix.
    WtW : numpy array
        Precomputed W.T @ W.
    WtX : numpy array
        Precomputed W.T @ X.
    """

    if options is None:
        options = {}
    if "delta" not in options:
        options["delta"] = 1e-6  # Stopping condition
    if "inneriter" not in options:
        options["inneriter"] = 500

    W = np.array(W, dtype=float)
    X = np.array(X, dtype=float)
    m, n = X.shape
    m, r = W.shape
    WtW = W.T @ W
    WtX = W.T @ X

    # If no initial matrices are provided, H is initialized as follows:
    if "init" not in options or options["init"] is None:
        H = nnls_init(X, W, WtW, WtX)
    else:
        H = options["init"]

    eps0 = 0
    cnt = 1
    epsi = 1
    while epsi >= (options["delta"]) ** 2 * eps0 and cnt <= options["inneriter"]:
        nodelta = 0
        for k in range(r):
            deltaH = np.maximum(
                (WtX[k, :] - WtW[k, :] @ H) / (WtW[k, k] + 1e-16), -H[k, :]
            )
            H[k, :] += deltaH
            nodelta += np.dot(deltaH, deltaH)
            if np.all(H[k, :] == 0):  # safety procedure
                H[k, :] = 1e-16
        if cnt == 1:
            eps0 = nodelta
        epsi = nodelta
        cnt += 1

    return H, WtW, WtX


def nnls(W, X, options=None):
    if options is None:
        options = {}
    
    options.setdefault('algo', 'HALS')
    options.setdefault('init', None) 
    options.setdefault('delta', 1e-6) 
    options.setdefault('inneriter', 500) 
    
    if 'alpha' in options and options['algo'] != 'ADMM':
        m, n = X.shape
        m, r = W.shape
        alphainneriter = 1 + int(np.ceil(options['alpha'] * (np.count_nonzero(X) + m * r) / (n * r)))
        # NOTE: disable this for now since it changes solver iterations number 
        # options['inneriter'] = min(alphainneriter, options['inneriter'])
    
    if options['algo'] == 'HALS':
        H, WTW, WTX = nnls_HALSupdt(X, W, options)
    elif options['algo'] == 'ASET':
        Z = np.maximum(0, options.get('init', 0))
        H, _, WTW, WTX = nnlsm_blockpivot(W, X, 0, )
        if np.isnan(H).sum() > 0:
            H, _, WTW, WTX = nnlsm_activeset(W, X, 0, 0, max(0, options.get('init', 0)))
    # filter prunnign via geometric median 
    elif options['algo'] == 'FPGM': 
        H, WTW, WTX = nnls_fpgm(X, W, options)
    elif options['algo'] == 'MUUP':
        H, WTW, WTX = nnls_MU(X, W, options)
    elif options['algo'] == 'ADMM':
        H, WTW, WTX = nnls_ADMM(X, W, options)
    elif options['algo'] == 'ALSH':
        WTW = W.T @ W
        WTX = W.T @ X
        r = W.shape[1]
        if np.linalg.cond(WTW) > 1e6:
            delta = np.trace(WTW) / r
            WTW = WTW + 1e-6 * delta * np.eye(WTW.shape[0])
            X, iter_count = solve_normal_eq_comb(WTW, WTX)
            H = np.maximum(0, X)
        else:
            X, iter_count = solve_normal_eq_comb(WTW, WTX)
            H = np.maximum(0, X)
    
    return H, WTW, WTX
