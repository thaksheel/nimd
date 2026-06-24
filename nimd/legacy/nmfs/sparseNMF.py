import numpy as np
import time
from nmfs.NNLS import *
import inspect
import copy


def sp_col(X, w=None):
    # Compute average Hoyer sparsity of columns of W
    m, r = X.shape
    x = [X[:, i] for i in range(r)]

    if w is None:
        w = [np.ones(m) for _ in range(r)]

    spx = sp(x, w)
    return spx


def sp(x, w=None):
    # Compute Hoyer sparsity of x
    r = len(x)
    spx = 0

    for i in range(r):
        if np.all(x[i] == 0):
            spx = 1
        else:
            ni = len(x[i])
            if w is None:
                spx += (
                    np.sqrt(ni) - np.linalg.norm(x[i], 1) / np.linalg.norm(x[i], 2)
                ) / (np.sqrt(ni) - 1)
            else:
                nw = np.linalg.norm(w[i], 2)
                spx += (nw - np.dot(w[i], np.abs(x[i])) / np.linalg.norm(x[i], 2)) / (
                    nw - np.min(w[i])
                )

    spx /= r
    return spx


def wcheckcrit(x, w, precision=1e-6):
    indi = np.where(w > 0)[0]
    xi = x[indi]
    wi = w[indi]

    xiwi = xi / wi
    maxx = np.max(xiwi)

    indi = np.where(np.abs(xiwi - maxx) < precision)[0]
    if len(indi) > 1:
        xcrit = maxx
    else:
        xcrit = []

    return xcrit, maxx


def nargout(func):
    sig = inspect.signature(func)
    return (
        len(sig.return_annotation)
        if sig.return_annotation != inspect.Signature.empty
        else 1
    )


def wgmu(x, w, mu):
    vgmu = 0
    gradg = 0
    xp = []

    for i in range(len(x)):
        ni = len(x[i])
        betai = 1 / (np.linalg.norm(w[i]) - np.min(w[i]))
        xpi = x[i] - mu * betai * w[i]
        indtp = np.where(xpi > 0)[0]

        if indtp.size > 0:
            xpi = np.maximum(0, xpi)
            f2 = np.linalg.norm(xpi)
            if nargout >= 3:
                nip = np.dot(w[i][indtp], w[i][indtp])
                gradg += betai**2 * (
                    -nip * f2 ** (-1)
                    + (np.dot(w[i][indtp], xpi[indtp])) ** 2 * f2 ** (-3)
                )
            xpi /= np.linalg.norm(xpi, 2)
            vgmu += betai * np.sum(xpi * w[i])
        else:
            im = np.argmax(xpi)
            xpi = np.zeros(ni)
            xpi[im] = 1
            vgmu += betai * w[i][im]

        xp.append(xpi)

    return vgmu, xp, gradg


def weightedgroupedsparseproj_col(X, s, options=None):
    if options is None:
        options = {}

    m, r = X.shape
    x = [X[:, i] for i in range(r)]

    xp, gxpmu, numiter, newmu = weightedgroupedsparseproj(x, s, options)

    Xp = np.column_stack(xp)

    return Xp, numiter


def weightedgroupedsparseproj(x, s, options=None):
    if options is None:
        options = {}

    if "w" not in options:
        options["w"] = [np.ones(len(xi)) for xi in x]
    if "precision" not in options:
        options["precision"] = 1e-3
    if "linrat" not in options:
        options["linrat"] = 0.9

    if s < 0 or s > 1:
        raise ValueError("The sparsity parameter has to be in [0,1].")

    k = 0
    muup0 = 0
    r = len(x)
    critmu = []

    sx = [np.sign(xi) for xi in x]
    x = [np.abs(xi) for xi in x]

    for i in range(r):
        nwi = np.linalg.norm(options["w"][i])
        betaim1 = nwi - np.min(options["w"][i])
        k += nwi / betaim1
        critxi, maxxi = wcheckcrit(x[i], options["w"][i])
        muup0 = max(muup0, maxxi * betaim1)
        critmu.extend(critxi * betaim1)

    k -= r * s
    vgmu, xnew, gradg = wgmu(x, options["w"], 0)

    if vgmu < k:
        return x, vgmu, 0, 0

    numiter = 0
    mulow = 0
    glow = vgmu
    muup = muup0
    newmu = 0
    gnew = glow
    gpnew = gradg
    Delta = muup - mulow

    while abs(gnew - k) > options["precision"] * r and numiter < 100:
        oldmu = newmu
        newmu = oldmu + (k - gnew) / gpnew

        if newmu >= muup or newmu <= mulow:
            newmu = (mulow + muup) / 2

        gnew, xnew, gpnew = wgmu(x, options["w"], newmu)

        if gnew < k:
            gup = gnew
            xup = xnew
            muup = newmu
        else:
            glow = gnew
            mulow = newmu

        if (
            muup - mulow > options["linrat"] * Delta
            and abs(oldmu - newmu) < (1 - options["linrat"]) * Delta
        ):
            newmu = (mulow + muup) / 2
            gnew, xnew, gpnew = wgmu(x, options["w"], newmu)
            if gnew < k:
                gup = gnew
                xup = xnew
                muup = newmu
            else:
                glow = gnew
                mulow = newmu

        numiter += 1

        if (
            critmu
            and abs(mulow - muup) < abs(newmu) * options["precision"]
            and min(abs(newmu - np.array(critmu))) < options["precision"] * newmu
        ):
            print("Warning: The objective function is discontinuous around mu^*.")
            return xnew, gnew, numiter, newmu

    xp = xnew
    gxpmu = gnew

    for i in range(r):
        alpha = np.dot(xp[i], x[i])
        xp[i] = alpha * (sx[i] * xp[i])

    return xp, gxpmu, numiter, newmu


def fastgradsparseNNLS(X, H, W, options):
    if "delta" not in options:
        options["delta"] = 0.1
    if "inneriter" not in options:
        options["inneriter"] = 100

    XHt = np.dot(X, H.T)
    HHt = np.dot(H, H.T)

    i = 1
    alpha0 = 0.1
    alpha = [alpha0]
    Yw = W
    L = np.linalg.norm(HHt, 2)
    Wn = W
    normWWn0 = -1
    while (
        i <= options["inneriter"]
        and np.linalg.norm(Wn - W) >= options["delta"] * normWWn0
    ):
        W = Wn
        gradYw = np.dot(Yw, HHt) - XHt
        Wn = np.maximum(0, Yw - 1 / L * gradYw)
        if "s" in options and options["s"] is not None:
            if options["colproj"] == 0:
                Wn = weightedgroupedsparseproj_col(Wn, options["s"], options)
            else:
                for k in range(Wn.shape[1]):
                    Wn[:, k] = weightedgroupedsparseproj_col(
                        Wn[:, k], options["s"], options
                    )
        alpha.append(
            (np.sqrt(alpha[i - 1] ** 4 + 4 * alpha[i - 1] ** 2) - alpha[i - 1] ** 2) / 2
        )
        beta = alpha[i - 1] * (1 - alpha[i - 1]) / (alpha[i - 1] ** 2 + alpha[i])
        Yw = Wn + beta * (Wn - W)
        if i == 1:
            normWWn0 = np.linalg.norm(W - Wn, "fro")
        i += 1

    return Wn, XHt, HHt


def sparseNMF(X, r, options=None):
    start_time = time.time()
    m, n = X.shape

    if options is None:
        options = {}

    options.setdefault("sW", None)
    options.setdefault("sH", None)
    options.setdefault("maxiter", 500)
    options.setdefault("timemax", 5)
    options.setdefault("delta", 0.1)
    options.setdefault("inneriter", 10)
    options.setdefault("FPGM", 0)
    options.setdefault("colproj", 0)
    options.setdefault("display", 1)

    if "W" in options and "H" in options:
        W = options["W"]
        H = options["H"]
    else:
        W = np.random.rand(m, r)
        H = np.random.rand(r, n)
        alpha = np.sum(W.T @ X * H) / np.sum((W.T @ W) * (H @ H.T))
        W *= alpha

    if options["sW"] is not None:
        W = weightedgroupedsparseproj_col(W, options["sW"], options)
    if options["sH"] is not None:
        H = weightedgroupedsparseproj_col(H.T, options["sH"], options).T

    itercount = 1
    nX2 = np.sum(X**2)
    nX = np.sqrt(nX2)
    e = []
    t = []
    Wbest = copy.deepcopy(W)
    Hbest = copy.deepcopy(H)
    ndisplay = 10

    if options["display"] == 1:
        print("Iteration number and relative error in percent:")

    while (
        itercount
        <= options["maxiter"]
        # TODO: stopped this for testing
        # and (time.time() - start_time) <= options["timemax"]
    ):
        normW = np.sqrt(np.sum(W**2, axis=0)) + 1e-16
        normH = np.sqrt(np.sum(H**2, axis=1)) + 1e-16
        for k in range(r):
            W[:, k] = W[:, k] / np.sqrt(normW[k]) * np.sqrt(normH[k])
            H[k, :] = H[k, :] / np.sqrt(normH[k]) * np.sqrt(normW[k])

        if options["FPGM"] == 0 and options["sH"] is None:
            options["init"] = H
            H, WTW, WTX = NNLS(W, X, options)
        else:
            options["s"] = options["sH"]
            H, WTW, WTX = fastgradsparseNNLS(X.T, W.T, H.T, options)
            H = H.T

        if options["sW"] is None:
            options["init"] = W.T
            W, HHt, XHt = NNLS(H.T, X.T, options)
            W = W.T
            XHt = XHt.T
        else:
            options["s"] = options["sW"]
            W, XHt, HHt = fastgradsparseNNLS(X, H, W, options)
        e.append(
            np.sqrt(max(0, (nX2 - 2 * np.sum(W * XHt) + np.sum(HHt * (W.T @ W)))) / nX)
        )
        t.append(time.time() - start_time)

        if itercount >= 2 and e[-1] <= e[-2]:
            Wbest = W
            Hbest = H

        if options["display"] == 1:
            if itercount % ndisplay == 0:
                print(f"{itercount}:{100 * e[-1]:.3f} - ", end="")
            if itercount % (ndisplay * 10) == 0:
                print()

        itercount += 1

    if itercount % (ndisplay * 10) > 0:
        print()

    W = Wbest
    H = Hbest

    return W, H, e, t, options, itercount
