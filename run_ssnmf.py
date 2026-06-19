import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
import torch

from src.core.selection import NMFSelection
from src import *


@dataclass
class SSNMFResults:
    task: str
    init: str
    rank: int
    score: float
    test_split: float
    lam: float
    model_num: int


task = "classification"
task = "regression"
X, y, df, feature_names = quick_load_data(filenmae="./data/data.csv", type=task)

deep_params = DeepNMFParams(
    layers_rank=[10, 5, 2],
    lam=None,
    outerit=100,
    maxiter=100,
    rngseed=42,
    rho=10,
    display=False,
    accADMM=True,
    beta=1,
    min_vol=True,
    innerloop=1,
    maxIterADMM=100,
    normalize=3,
)
ssnmf_params = SSNMFParam(
    model_num=4,
    split=0.4,
    numiters=400,
    iter_s=100,
    lam=1e-3,
)
params = RunnerParams(
    rank=10,
    model="ssnmf",
    task=task,
    init="nnsvdlrc",
    deep_params=deep_params,
    multi_params=deep_params,
    ssnmf_params=ssnmf_params,
    dtype=torch.float64,
    device=torch.device("cpu"),
    val_train_test_splits=[0, 0.8, 0.2],
    display=True,
    rng=42,
    eval_type="full",  # does not work for PCA
    eps_stab=1e-7,
    get_shap=False,
    norm_X='minmax',   
)
runner = Runner(X, y)

ranks = [i for i in range(20, 70, 1)]
lams = [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
nums = [3, 4, 5, 6]
inits = ["nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc", "random"]
dfs = []
for num in nums:
    for init in inits:
        params.init = init
        result_data = runner.run_evaluation(params, ranks)
        df_ = runner.convert_results_to_df(result_data)
        df_['model_num'] = [num] * len(df_)
        dfs.append(df_)
        print(f"---> completed {init} \n model_num{num}")

df_results = pd.concat(dfs)
df_results.to_excel("./exports/results_ssnmf2.xlsx")

df_ssnmf = (
    df.query("20 <= rank <= 70")
    .groupby(["lam", "model_num", "init"])
    .mean(numeric_only=True)
    .reset_index()
)
best_by_init = df.loc[df.groupby("init")["score"].idxmax()].reset_index(drop=True)
best_by_init.to_excel("./exports/ssnmf_scores.xlsx")

print("END")
