import pandas as pd
import torch
import warnings

from src import *

warnings.filterwarnings("ignore", category=Warning)

deep_params = DeepNMFParams(
    layers_rank=[10, 5, 2],
    lam=None,
    outerit=50,
    maxiter=50,
    rngseed=42,
    rho=10,
    display=False,
    accADMM=True,
    beta=1,
    min_vol=True,
    innerloop=1,
    maxIterADMM=50,
    normalize=3,
)
params = RunnerParams(
    rank=10,
    model="deep",
    task="regression",
    deep_params=deep_params,
    multi_params=deep_params,
    ssnmf_params=SSNMFParam(),
    dtype=torch.float64,
    device=torch.device("cpu"),
    val_train_test_splits=[0, 0.8, 0.2],
    display=True,
    rng=42,
    eval_type="full",  # does not work for PCA
    eps_stab=1e-7,
    get_shap=False,
)

ranks = [r for r in range(20, 71, 1)]
algos = ["FPGM", "MUUP", "ADMM", "HALS", "ALSH"]
models = ["pca", "beta", "hier", "multilayer", "fronorm", "deep"]
models = ["minvol"]  # minvol is done in run_depths.py instead
inits = ["nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc", "random"]

# ranks = [20, 40, 60]
# algos = ["FPGM", "MUUP"]
# models = ["pca", "hier", "beta"]
# inits = ["nndsvd", "random"]

dfs = []
tasks = ["regression", "classification"]
tasks = ["classification"]
for task in tasks:
    for model in models:
        X, y, df, feature_names = quick_load_data(filenmae="./data/data.csv", type=task)
        runner = Runner(X, y)
        params.task = task
        results = runner.exp_supervised_results(
            ranks=ranks,
            models=[model],
            inits=inits,
            algos=algos,
            params=params,
        )
        df_ = runner.convert_results_to_df(results)
        df_["task"] = [task] * len(df_)
        dfs.append(df_)
        df = pd.concat(dfs)
        df.to_excel("./exports/minvol_classification_results1.xlsx")

print(df.head())
print("END")
