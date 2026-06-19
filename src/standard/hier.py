import numpy as np
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.linalg import svd
import warnings


def anls_entry_rank2_binary(left, right):
    n = right.shape[0]
    solve_either = np.zeros((n, 2))
    solve_either[:, 0] = np.maximum(0, right[:, 0] / left[0, 0])
    solve_either[:, 1] = np.maximum(0, right[:, 1] / left[1, 1])

    cosine_either = solve_either * np.array([np.sqrt(left[0, 0]), np.sqrt(left[1, 1])])

    choose_first = cosine_either[:, 0] >= cosine_either[:, 1]
    solve_either[choose_first, 1] = 0
    solve_either[~choose_first, 0] = 0

    return solve_either


def anls_entry_rank2_precompute_opt(left, right):
    left = np.array(left)
    right = np.array(right)
    
    if left.size == 1:
        H = np.maximum(0, right / left)
    else:
        # Solve left * H.T = right.T for H.T, then transpose back
        H = np.linalg.lstsq(left, right.T, rcond=None)[0].T
        use_either = ~(np.all(H >= 0, axis=1))
        if np.any(use_either):
            H[use_either, :] = anls_entry_rank2_binary(left, right[use_either, :])
        H = H.T
    return H


def SPA(X, r, options=None):
    """
    Successive projection algorithm for separable NMF

    Parameters:
    X (numpy.ndarray): an m-by-n matrix X.
    r (int): number of columns to be extracted.
    options (dict): options for the algorithm.

    Returns:
    K (list): index set of the extracted columns.
    """
    m, n = X.shape

    # Options
    if options is None:
        options = {}
    if "normalize" not in options:
        options["normalize"] = 0
    if "display" not in options:
        options["display"] = 1
    if "precision" not in options:
        options["precision"] = 1e-6

    if options["normalize"] == 1:
        # Normalization of the columns of X so that they sum to one
        D = diags((np.sum(X, axis=0) ** (-1)), 0)
        X = X @ D

    normX0 = np.sum(X**2, axis=0)
    nXmax = np.max(normX0)
    normR = normX0.copy()

    i = 1
    K = []
    U = np.zeros((m, r))

    # Perform r recursion steps (unless the relative approximation error is smaller than 10^-9)
    if options["display"] == 1:
        print("Extraction of the indices by SPA:")

    while i <= r and np.sqrt(np.max(normR) / nXmax) > options["precision"]:
        # Select the column of X with largest l2-norm
        a = np.max(normR)
        b = np.where((a - normR) / a <= 1e-6)[0]
        if len(b) > 1:
            c = np.max(normX0[b])
            b = b[np.argmax(normX0[b])]
        else:
            b = b[0]

        # Update the index set, and extracted column
        K.append(b)
        U[:, i - 1] = X[:, b]

        # Compute (I - u_{i-1}u_{i-1}^T) ... (I - u_1u_1^T) U[:, i], that is, R^(i)(:, J(i))
        for j in range(i - 1):
            U[:, i - 1] -= U[:, j] * (U[:, j].T @ U[:, i - 1])

        # Normalize U[:, i]
        U[:, i - 1] /= np.linalg.norm(U[:, i - 1])

        # Update the norm of the columns of X after orthogonal projection
        normR -= (U[:, i - 1].T @ X) ** 2

        if options["display"] == 1:
            print(f"{i}...", end="")
            if i % 10 == 0:
                print()

        i += 1

    if options["display"] == 1:
        print()

    return K


def fastsvds(M, r):
    M = np.array(M)
    m, n = M.shape
    rationmn = 10

    if m < rationmn * n:
        MMt = M @ M.T
        u, s, _ = svd(MMt)
        u = u[:, :r]
        s = s[:r]
        v = M.T @ u
        v = v / (np.sqrt(np.sum(v**2, axis=0, keepdims=True)) + 1e-16)
        s = np.sqrt(s)
    elif n < rationmn * m:
        MtM = M.T @ M
        v, s, _ = svd(MtM)
        v = v[:, :r]
        s = s[:r]
        u = M @ v
        u = u / (np.sqrt(np.sum(u**2, axis=0, keepdims=True)) + 1e-16)
        s = np.sqrt(s)
    else:
        u, s, v = svd(M)
        u = u[:, :r]
        s = s[:r]
        v = v[:r, :]

    return u, s, v


def rank2nmf(M):
    m, n = M.shape

    # Best rank-two approximation of M
    if min(m, n) == 1:
        U, S, Vt = fastsvds(M, 1)
        U = np.abs(U)
        V = np.abs(Vt.T)
        s1 = S[0]
    else:
        u, s, vt = fastsvds(M, 2)
        s1 = s[0]
        options = {"display": 0}
        s = np.diag(s)
        z = s @ vt.T
        K = SPA(z, 2, options)
        U = np.zeros((M.shape[0], 2))
        if len(K) >= 1:
            U[:, 0] = np.maximum(u @ s @ vt[K[0], :].T, 0)
        if len(K) >= 2:
            U[:, 1] = np.maximum(u @ s @ vt[K[1], :].T, 0)
        # Compute corresponding optimal V
        # TODO: different from master, need 'anls_entry_rank2_precompute_opt'
        V = anls_entry_rank2_precompute_opt(U.T @ U, M.T @ U)

    return U, V, s1


def fquad(x, s=0.01):
    delta, fdel, fdelp, finter, gs = fdelta(x, s)
    # fdel is the percentage of values smaller than delta
    # finter is the number of points in a small interval around delta

    epsilon = 1e-10  # A small number to avoid log(0)
    fdel = np.clip(fdel, epsilon, 1 - epsilon)
    fobj = -np.log(fdel * (1 - fdel)) + np.exp(finter)
    # fobj = -np.log(fdel * (1 - fdel)) + np.exp(finter) #NOTE: remove since divide by zero

    # Can potentially use other objectives:
    # fobj = -np.log(fdel * (1 - fdel)) + 2**finter
    # fobj = (2 * (fdel - 0.5))**2 + finter**2
    # fobj = -np.log(fdel * (1 - fdel)) + finter**2
    # fobj = (2 * (fdel - 0.5))**2 + finter**2

    b = np.argmin(fobj)
    thres = delta[b]

    return thres, delta, fobj


def fdelta(x, s=0.01):
    n = len(x)
    delta = np.arange(0, 1 + s, s)
    lD = len(delta)

    gs = 0.05  # Other values could be used, in [0, 0.5]

    fdel = np.zeros(lD)
    fdelp = np.zeros(lD)
    finter = np.zeros(lD)

    for i in range(lD):
        fdel[i] = np.sum(x <= delta[i]) / n
        if i == 1:  # use only next point to evaluate fdelp[0]
            fdelp[0] = (fdel[1] - fdel[0]) / s
        elif i >= 2:  # use next and previous point to evaluate fdelp[i-1]
            fdelp[i - 1] = (fdel[i] - fdel[i - 2]) / (2 * s)
            if i == lD - 1:  # use only previous point to evaluate fdelp[lD-1]
                fdelp[lD - 1] = (fdel[lD - 1] - fdel[lD - 2]) / s

        deltahigh = min(1, delta[i] + gs)
        deltalow = max(0, delta[i] - gs)
        finter[i] = (
            (np.sum(x <= deltahigh) - np.sum(x < deltalow)) / n / (deltahigh - deltalow)
        )

    return delta, fdel, fdelp, finter, gs


def spkmeans(X, init):
    """
    Perform spherical k-means clustering.

    Parameters:
    X (numpy.ndarray): d x n data matrix
    init (int or numpy.ndarray): k (1 x 1) or label (1 x n, 1<=label[i]<=k) or center (d x k)

    Returns:
    label (numpy.ndarray): cluster labels
    m (numpy.ndarray): cluster centers
    energy (float): clustering energy
    """
    d, n = X.shape

    if n <= init:
        label = np.arange(1, init + 1)
        m = X
        energy = 0
    else:
        # Normalize the columns of X
        X = X * (np.sum(X**2, axis=0) + 1e-16) ** (-0.5)

        if isinstance(init, int):
            idx = np.random.choice(n, init, replace=False)
            m = X[:, idx]
            label = np.argmax(m.T @ X, axis=0)
        elif init.shape[0] == 1 and init.shape[1] == n:
            label = init
        elif init.shape[0] == d:
            m = init * (np.sum(init**2, axis=0) + 1e-16) ** (-0.5)
            label = np.argmax(m.T @ X, axis=0)
        else:
            raise ValueError("ERROR: init is not valid.")

        # Main algorithm: final version
        last = np.zeros(n)
        while np.any(label != last):
            u, indices = np.unique(label, return_inverse=True)
            k = len(u)
            E = csr_matrix((np.ones(n), (np.arange(n), indices)), shape=(n, k))
            m = X @ E.toarray()
            m = m * (np.sum(m**2, axis=0) + 1e-16) ** (-0.5)
            last = label
            label = np.argmax(m.T @ X, axis=0)

        u, indices = np.unique(label, return_inverse=True)
        energy = np.sum(np.max(m.T @ X, axis=0))

    return label, m, energy


def splitclust(M, algo=1):
    if algo == 1:  # rank-2 NMF
        U, V, s = rank2nmf(M)
        # Normalize columns of V to sum to one
        V = V * (np.sum(V, axis=0) + 1e-16) ** (-1)
        x = V[0, :]
        # Compute threshold to split cluster
        threshold, _, _ = fquad(x)
        K = [np.where(x >= threshold)[0], np.where(x < threshold)[0]]

    # NOTE: FastSepNMF not present in master
    # elif algo == 2:  # k-means
    #     u, s, vt = svds(M, k=2)  # Initialization: SVD+SPA
    #     Kf = FastSepNMF(s @ vt, 2, 0)
    #     U0 = u @ s @ vt[Kf, :].T

    #     kmeans = KMeans(n_clusters=2, init=U0.T, n_init=1)
    #     IDX = kmeans.fit_predict(M.T)
    #     U = kmeans.cluster_centers_.T
    #     K = [np.where(IDX == 0)[0], np.where(IDX == 1)[0]]
    #     s = s[0]

    # elif algo == 3:  # spherical k-means
    #     u, s, vt = svds(M, k=2)  # Initialization: SVD+SPA
    #     Kf = FastSepNMF(s @ vt, 2, 0)
    #     U0 = u @ s @ vt[Kf, :].T

    #     IDX, U = spkmeans(M, U0)
    #     # Alternatively, you can use:
    #     # kmeans = KMeans(n_clusters=2, init=U0.T, n_init=1, algorithm='full')
    #     # IDX = kmeans.fit_predict(M.T)
    #     # U = kmeans.cluster_centers_.T
    #     K = [np.where(IDX == 0)[0], np.where(IDX == 1)[0]]
    #     s = s[0]

    return K, U, s


def reprvec(M):
    u, s, vt = svd(M)
    u = u[:, 0]
    s = s[:1]
    u = np.abs(u)
    m, n = M.shape

    # Extract the column of M approximating u the best (up to a translation and scaling)
    u = u - np.mean(u)
    Mm = M - np.mean(M, axis=0)

    norm_u = np.linalg.norm(u)
    dot_products = Mm.T @ u / norm_u
    col_norms = np.sqrt(np.sum(Mm**2, axis=0))
    err = np.arccos(np.clip(dot_products / (col_norms + 1e-16), -1, 1))

    # err = np.arccos((Mm.T @ u / np.linalg.norm(u)) / np.sqrt(np.sum(Mm**2, axis=0)))
    b = np.argmin(err)
    u = M[:, b]

    return u, s, b


def clu2vec(K, m=None, r=None):
    if r is None:
        r = len(K)
    if m is None:
        # Compute max entry in K
        m = 0
        for i in range(r):
            m = max(m, max(K[i]))

    IDX = np.zeros(m, dtype=int)
    for i in range(r):
        IDX[K[i]] = i + 1  # MATLAB is 1-based, Python is 0-based, so we add 1

    return IDX


def vectoind(IDX, r=None):
    m = len(IDX)
    if r is None:
        r = np.max(IDX)

    V = np.zeros((m, r), dtype=int)
    for i in range(1, r + 1):
        V[np.where(IDX == i), i - 1] = (
            1  # MATLAB is 1-based, Python is 0-based, so we adjust the index
        )

    return V


def hierclust2nmf(M, r: int, algo: int = 1, sol=None):
    M = np.array(M)
    if np.min(M) < 0:
        warnings.warn(
            "The input matrix contains negative entries which have been set to zero"
        )
        M = np.maximum(M, 0)

    # If input is a tensor, convert to matrix
    if M.ndim == 3:
        H, L, m_ = M.shape
        n = H * L
        A = np.zeros((m_, n))
        for i in range(m_):
            A[i, :] = M[:, :, i].reshape(-1)
        M = A
        m = m_
    else:
        m, n = M.shape

    if sol is None:
        sol = {}
        sol["K"] = [np.arange(n)]
        sol["allnodes"] = [1]
        sol["maxnode"] = 1
        sol["parents"] = [[0, 0]]
        sol["childs"] = {}
        sol["leafnodes"] = [0]
        sol["e"] = [[-1]]
        sol["U"] = [np.ones(m)]
        sol["count"] = 1
        sol["firstsv"] = [0]

    manualsplit = 0

    while sol["count"] < r:
        # Update: split leaf nodes added at previous iteration
        for k in range(len(sol["leafnodes"])):
            leaf_idx = sol["leafnodes"][k]
            if sol["e"][leaf_idx][0] == -1 and len(sol["K"][leaf_idx]) > 1:
                # Split the cluster
                Kc, Uc, sc = splitclust(M[:, sol["K"][leaf_idx]], algo)

                # Add the two leaf nodes, child of nodes(sol.leafnodes(k))
                sol["allnodes"].extend([sol["maxnode"] + 1, sol["maxnode"] + 2])
                while len(sol["parents"]) <= sol["maxnode"] + 2:
                    sol["parents"].append([0, 0])
                sol["parents"][sol["maxnode"] + 1] = [leaf_idx, 0]
                sol["parents"][sol["maxnode"] + 2] = [leaf_idx, 0]
                sol["childs"][leaf_idx] = [sol["maxnode"] + 1, sol["maxnode"] + 2]
                sol["childs"][sol["maxnode"] + 1] = []
                sol["childs"][sol["maxnode"] + 2] = []

                # Assign clusters
                sol["K"].append(sol["K"][leaf_idx][Kc[0]])
                sol["K"].append(sol["K"][leaf_idx][Kc[1]])

                # Compute centroids and firstsv
                U1, sv1, _ = reprvec(M[:, sol["K"][len(sol["K"]) - 2]])
                U2, sv2, _ = reprvec(M[:, sol["K"][len(sol["K"]) - 1]])
                sol["U"].append(U1)
                sol["U"].append(U2)
                sol["firstsv"].append(sv1)
                sol["firstsv"].append(sv2)
                # Update criterion 
                sol["e"].extend([[-1], [-1]])
                sol["e"][leaf_idx] = sv1**2 + sv2**2 - sol["firstsv"][leaf_idx] ** 2
                sol["maxnode"] += 2
        # Choose the cluster to split, split it, and update leaf nodes
        if sol["count"] == 1:
            b = 0
        elif manualsplit == 0:
            e_vals = [sol["e"][idx] for idx in sol["leafnodes"]]
            b = int(np.argmax(e_vals))
        if b > -1:
            leaf_to_split = sol["leafnodes"][b]
            sol["leafnodes"].extend([i-1 for i in sol["childs"][leaf_to_split]])
            sol["leafnodes"] = sol["leafnodes"][:b] + sol["leafnodes"][b + 1 :]
            sol["count"] += 1

    # Convert clusters to indicator vector and centroids matrix
    IDX = clu2vec([sol["K"][i] for i in sol["leafnodes"]], n)
    C = np.column_stack([sol["U"][i] for i in sol["leafnodes"]])

    return IDX, C, sol
