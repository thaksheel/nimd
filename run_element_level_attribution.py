import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src import (
    Runner,
    RunnerParams,
    DeepNMFParams,
    quick_load_data,
)
from src.interpretation import Interpretation


X, y, df, feature_names, g_classes = quick_load_data(
    filenmae="./data/data.csv", type="regression", return_distribution=True
)

params = DeepNMFParams(
    layers_depths=4,
    division_base=1.2,
    layers_rank=None,
    lam=None,
    outerit=70,
    maxiter=70,
    rngseed=42,
    rho=10,
    display=False,
    accADMM=True,
    beta=1,
    min_vol=False,
    normalize=2,
    innerloop=1,
    maxIterADMM=70,
)
runner_params = RunnerParams(
    end=10,
    convert_type="division_base",
    init="nnsvdlrc",
    norm_X=None,
    rank=None,
    model="beta",
    fronorm_algo="HALS",
    task="regression",
    score_type="r2",
    deep_params=params,
    multi_params=params,
    dtype=torch.float64,
    device=torch.device("cpu"),
    val_train_test_splits=[0, 0.8, 0.2],
    display=True,
    rng=42,
    eval_type="feature",  # does not work for PCA
    eps_stab=1e-7,
    return_tensor=False,
)
runner = Runner(X=X, y=y)
interpret = Interpretation()
nmf_models = ["deep", "multilayer", "fronorm", "beta", "hier", "deep_minvol"]
nmf_models = ["deep"]
nmf_models = ["deep", "multilayer", "fronorm", "beta", "hier"]
ranks = [i for i in range(20, 70, 10)]

df_atr = runner.exp_element_level_attribution(
    runner_params, ranks, nmf_models, interpret
)
df_results = interpret.evaluate_attribution(
    df=df_atr,
    feature="element_name",
    g_classes=g_classes,
)
df_atr.to_excel("./exports/el_atr_top.xlsx")
df_results.to_excel("./exports/el_atr_results.xlsx")

print(
    df_results.sort_values(
        ["coherence", "diversity"], ascending=[False, True]
    )
)  # insight

print("END")
