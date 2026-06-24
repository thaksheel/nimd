import numpy as np
from nmfs._nnls import *

def NNLS(W, X, options=None):
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
