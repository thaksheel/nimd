import torch
import numpy as np
import re
from dataclasses import dataclass
from typing import Literal, Optional, List
from scipy.optimize._nnls import nnls
import pandas as pd
from scipy.stats import ttest_ind

from .. import DeepNMFParams, MultilayerParams


@dataclass
class SSNMFParam:
    model_num: Literal[3, 4, 5, 6] = 4
    split: float = 0.2
    tol: float = 1e-4
    numiters: int = 400
    saveerrs: bool = False
    iter_s: int =100
    lam: float = 1e-3
    seed: int = 42


@dataclass
class RunnerParams:
    """
    eps_stab=1e-6 for minvol=True with best performance
    eps_stab=1e-7 for minvol=False
    """

    dtype: torch.dtype
    device: torch.device
    rank: int
    val_train_test_splits: List
    model: Literal["deep", "multilayer", "fronorm", "beta", "hier", "pca", "ssnmf"]
    task: Literal["regression", "classification"]
    deep_params: DeepNMFParams
    multi_params: MultilayerParams
    ssnmf_params: SSNMFParam

    get_shap: bool = False
    end: int = 10
    convert_type: Literal["division_base", "linspace"] = "division_base"
    fronorm_algo: Literal["MUUP", "ADMM", "HALS", "FPGM", "ALSH"] = "HALS"
    eval_type: Literal["feature", "full"] = "feature"
    init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"] = "nndsvd"
    modelname: Literal["rf", "linreg", "logit", "mlp"] = "rf"
    norm_X: Optional[Literal["minmax", "standard"]] = None
    norm_init: Optional[Literal["scaling", "feature_norm"]] = None
    rng: int = 42
    cv: bool = False
    return_tensor: bool = False
    display: bool = False
    eps_stab: float = 1e-4
    perturb: bool = False
    noise_level: Optional[float] = None


@dataclass
class Score:
    r2: float
    rmse: float
    mae: float
    accuracy: float
    f1_macro: float
    test_size: int
    train_size: int
    f1: np.ndarray
    true: np.ndarray
    pred: np.ndarray


@dataclass
class ResultData:
    nmf_model: str
    ml_model: str
    rank: int
    init: str
    min_vol: bool
    depths: int
    ranks: List[int]

    base_score: Score
    score: Score
    feature_importance: np.ndarray
    runtime: float

    W: np.ndarray
    H: np.ndarray
    Hl: List[np.ndarray]
    Wl: List[np.ndarray]
    error: List[np.ndarray]
    eps: float

    noise: float = None
    seed: int = None
    fronorm_algo:str = None


@dataclass
class HierarchyAbundance:
    component_num: int
    abundance: float
    rank: int
    layer: int
    threshold: float
    previous_component_num: int = 0
    element_num: int = None
    element_name: str = None
    metal_class: str = None


def find_sparsity(X: np.ndarray):
    return np.count_nonzero(X == 0) / X.size


def get_s(
    model,
    X_test: np.ndarray,
    A_train: np.ndarray,
    model_num: int,
    iter_s: int = 20,
):
    """
    Given a trained (S)SNMF model i.e. learned data dictionary, A, the function applies the (S)SNMF model
    to the test data to produce the representation of the test data.

    Returns:
        S_test (ndarray): representation matrix of the test data, shape(#topics, #test features)
    """
    r = A_train.shape[1]
    n1 = X_test.shape[0]
    n2 = X_test.shape[1]
    # Frobenius measure on features data (use nonnegative least squares)
    if model_num == 3 or model_num == 4:
        S_test = np.zeros([r, n2])
        err = []
        for i in range(n2):
            S_test[:, i], e = nnls(A_train, X_test[:, i])
            # err.append(e/np.linalg.norm(X_test[:, i]))
    # I-divergence measure on features data (use mult. upd. I-div)
    np.random.seed(41)
    W_test = np.ones((n1, n2))
    if model_num == 5 or model_num == 6:
        S_test = np.random.rand(r, X_test.shape[1])
        for i in range(iter_s):
            S_test = np.transpose(
                model.dictupdateIdiv(
                    np.transpose(X_test),
                    np.transpose(S_test),
                    np.transpose(A_train),
                    np.transpose(W_test),
                    eps=1e-10,
                )
            )
    return S_test


def correct_matrix(W: np.ndarray) -> np.ndarray:
    """Corrects for zeros and negative values to machine epsillon using np.finfo(float).eps"""
    machine_epsilon = np.finfo(float).eps
    W[W == 0] = machine_epsilon
    W = np.maximum(W, machine_epsilon)
    return W


def quick_load_data(filenmae: str, type: Literal["regression", "classification"], return_distribution: bool = False):
    dropcols = ["Material", "a", "b", "Composition", "Stability"]
    df = pd.read_csv(filenmae)
    if type == "classification":
        df["Stability"] = df["Stability"].map({"unstable": 0, "stable": 1}).astype(int)
        X = df.drop(columns=dropcols, axis=1).copy().to_numpy()
        y = df["Stability"].copy().to_numpy()
    elif type == "regression":
        X = df.drop(dropcols, axis=1).copy().to_numpy()
        y = df["a"].copy().to_numpy()

    feature_names = df.drop(dropcols, axis=1).columns.tolist()
    if return_distribution:
        g_classes = np.array(df[feature_names[:-9]].sum(0) / df[feature_names[:-9]].sum(0).sum())
        return X, y, df, feature_names, g_classes
    return X, y, df, feature_names


def _table(M: str, full: pd.DataFrame, ot: Literal["mean", "std"]) -> pd.DataFrame:
    if ot == "mean":
        summary = full.groupby([M, "init"])["score"].mean().reset_index()
    if ot == "std":
        summary = full.groupby([M, "init"])["score"].std().reset_index()
    table = summary.pivot(index=M, columns="init", values="score")
    return table


def scores_summary_table(files: List[str]) -> List[pd.DataFrame]:
    models = [re.search(r"results_([a-zA-Z0-9_]+)\d\.xlsx$", s).group(1) for s in files]
    datasets = dict(zip(models, files))
    dfs = []
    for model_name, path in datasets.items():
        df = pd.read_excel(path)
        df["model"] = model_name
        dfs.append(df)
    full: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    tables = [
        _table(m, full, ot) for m in ["model", "fronorm_algo"] for ot in ["mean", "std"]
    ]
    return tables


class AnalyzeResults:
    def __init__(self):
        pass

    def means_table(self, df: pd.DataFrame, metrics: List[str], models: List[str]):
        means = []
        for model in models:
            means.append(
                np.array([df[df.method == model][metric].mean() for metric in metrics])
            )
        df = pd.DataFrame(columns=["model"], data=models)
        df[metrics] = means
        return df

    def stddev_table(self, df: pd.DataFrame, metrics: List[str], models: List[str]):
        means = []
        for model in models:
            means.append(
                np.array([df[df.method == model][metric].std() for metric in metrics])
            )
        df = pd.DataFrame(columns=["model"], data=models)
        df[metrics] = means
        return df

    def ttest(self, df: pd.DataFrame, pval_threshold: float = 0.05):
        models = df.method.unique()
        results = np.array(
            [df[df.method == method].accuracy.to_numpy() for method in models]
        )
        results_counts = self.pairwise_ttest(
            models, results, pval_threshold=pval_threshold
        )
        df_pvals = pd.DataFrame(columns=models, index=models, data=results_counts)
        return df_pvals

    def pairwise_ttest(
        self, models: List, results: np.ndarray, pval_threshold: float = 0.05
    ):
        pvals = []
        for i in range(len(models)):
            pvals.append(
                np.array(
                    [
                        ttest_ind(results[i], result, equal_var=False)[1]
                        for result in results
                    ]
                )
            )
        pvals = np.array(pvals)
        eval_pvals = pvals <= pval_threshold
        return eval_pvals
