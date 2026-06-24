import numpy as np
from sklearn.model_selection import train_test_split
from typing import Literal
import pandas as pd

from ..legacy.nmfs.ssnmf import SSNMF

# TODO: move core.utils to standard.utils
from ..core.utils import SSNMFParam, get_s
from ..core.initiliazation import initialize_nmf


def correct_matrix(W: np.ndarray) -> np.ndarray:
    """Corrects for zeros and negative values to machine epsillon using np.finfo(float).eps"""
    machine_epsilon = np.finfo(float).eps
    W[W == 0] = machine_epsilon
    W = np.maximum(W, machine_epsilon)
    return W


def ss_nmf(
    X: np.ndarray,
    W0: np.ndarray,
    H0: np.ndarray,
    y: np.ndarray,
    init: str,
    rank: int,
    params: SSNMFParam,
    task: Literal["regression", "classification"],
    rng: int = 42,
) -> tuple:
    np.random.seed(rng)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params.split, random_state=rng
    )
    W0, H0 = initialize_nmf(X_train, rank, init=init, random_state=rng)
    if task == "regression":
        y = np.array([y_train.copy().T])
    elif task == "classification":
        y = pd.get_dummies(y_train).T.to_numpy().astype(int)
    A0, S0, B0 = (
        H0.T,
        W0.T,
        correct_matrix(np.random.rand(y.shape[0], rank)),
    )
    model = SSNMF(
        X_train.T,
        rank,
        Y=y,
        lam=params.lam,
        tol=params.tol,
        A=A0,
        B=B0,
        S=S0,
        modelNum=params.model_num,
    )
    model.mult(
        numiters=params.numiters,
        saveerrs=params.saveerrs,
        eps=1e-14,
    )
    S_train, A_train = model.S, model.A
    S_test = get_s(
        model,
        X_test.T,
        A_train,
        params.model_num,
        iter_s=params.iter_s,
    )
    return (S_train.T, S_test.T, y_train.T, y_test.T)
