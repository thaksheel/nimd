import numpy as np
import time
from .nnls import nnls
import copy
from typing import Literal, Dict


def fro_nmf(X, r, algo: Literal["MUUP", "ADMM", "HALS", "FPGM", "ALSH"], W0, H0, options=None):
    start_time = time.time()
    m, n = X.shape

    if options is None:
        options = {}

    # Set default options
    options.setdefault("display", 0)
    options.setdefault(
        "init", {"W": W0, "H": H0})
    options.setdefault("maxiter", 500)
    options.setdefault("algo", algo)
    options.setdefault("timemax", 60)  # NOTE: changed to 120s
    options.setdefault("accuracy", 1e-4)
    options.setdefault("delta", 1e-1)  # TODO: had to change back to 1e-1
    options.setdefault("alpha", 0.5)
    options.setdefault("inneriter", 100)
    options.setdefault("extrapolprojH", 3)
    # NOTE: this was missing with the MUUP beta which caused most of the issues
    options.setdefault("beta0", 0 if algo == 'MUUP' else 0.5)
    options.setdefault("eta", 1.5)
    options.setdefault("gammabeta", 1.1 if algo == "ANLS" else 1.01)
    options.setdefault(
        "gammabetabar", 1.05 if algo == "ANLS" else 1.005)

    if (
        options["eta"] < options["gammabeta"]
        or options["gammabeta"] < options["gammabetabar"]
    ):
        raise ValueError("You should choose eta > gamma > gammabar.")

    if not (0 <= options["beta0"] <= 1):
        raise ValueError("beta0 must be in the interval [0,1].")

    W, H = options["init"]["W"], options["init"]["H"]

    # Scale initialization so that argmin_a ||a * WH - X||_F = 1
    XHt = X @ H.T
    HHt = H @ H.T
    scaling = np.sum(XHt * W) / np.sum(HHt * (W.T @ W))
    W *= scaling

    # Normalize W and H so that columns/rows have the same norm, that is,  ||W(:,k)|| = ||H(k,:)|| for all k.
    normW = np.sqrt(np.sum(W**2, axis=0)) + 1e-16
    normH = np.sqrt(np.sum(H**2, axis=1)) + 1e-16
    d = np.sqrt(normW) / np.sqrt(normH)
    H *= d[:, None]
    for k in range(r):
        W[:, k] = W[:, k] / np.sqrt(normW[k]) * np.sqrt(normH[k])

    # Extrapolation variables
    Wy, Hy = W.copy(), H.copy()
    beta = [options["beta0"]]
    betamax = 1
    nX = np.linalg.norm(X, "fro")
    e = [np.linalg.norm(X - W @ H, "fro") / nX]
    etrue = [e[0]]
    emin = e[0]
    Wbest, Hbest = W.copy(), H.copy()
    t = [time.time() - start_time]
    i = 1
    while (
        i <= options["maxiter"]
        and (i <= 12 or abs(e[-2] - e[-12]) >= options["accuracy"])
    ):
        options["init"] = Hy
        Hn, _, _ = nnls(Wy, X, options)
        if options["extrapolprojH"] >= 2:
            Hy = Hn + beta[i-1] * (Hn - H)
        else:
            Hy = Hn
        if options["extrapolprojH"] == 3:
            Hy = np.maximum(0, Hy)
        options["init"] = Wy.T
        Wn, HyHyT, XHyT = nnls(Hy.T, X.T, options)
        Wn = Wn.T
        XHyT = XHyT.T
        HyHyT = HyHyT.T
        # Wy = Wn + beta[i - 1] * (Wn - W)
        Wy = Wn + beta[i - 1] * (Wn - W)
        if options["extrapolprojH"] == 1:
            Hy = Hn + beta[i - 1] * (Hn - H)
        e.append(np.sqrt(max(0, nX**2 - 2 * np.sum(XHyT * Wn) +
                 np.sum(HyHyT * (Wn.T @ Wn)))) / nX)
        t.append(time.time() - start_time)
        if e[-1] > e[i-1]:
            # If ADMM --> reduce delta
            if options["algo"] == "ADMM":
                options["delta"] /= 10
                options["inneriter"] = int(np.ceil(1.5 * options["inneriter"]))
                if options["display"] == 1:
                    print(
                        "ADMM reduces the parameter delta and increases the number of inner iterations."
                    )
            # Scale (W, H)
            normW = np.sqrt(np.sum(W**2, axis=0)) + 1e-16
            normH = np.sqrt(np.sum(H**2, axis=1)) + 1e-16
            for k in range(r):
                W[:, k] = W[:, k] / np.sqrt(normW[k]) * np.sqrt(normH[k])
                H[k, :] = H[k, :] / np.sqrt(normH[k]) * np.sqrt(normW[k])
            # Restart the scheme
            Wy = W
            Hy = H
            if i == 1:
                betamax = beta[-1]
            else:
                betamax = beta[i - 1]
            beta.append(beta[i-1] / options["eta"])
        else:
            W = Wn
            H = Hn
            beta.append(min(betamax, beta[-1] * options["gammabeta"]))
            betamax = min(1, betamax * options["gammabetabar"])
        if e[-1] <= emin:
            Wbest = copy.deepcopy(Wn)
            Hbest = np.maximum(Hy, 0)
            emin = e[-1]
        i += 1
    return Wbest, Hbest, e, t, etrue, i, options
