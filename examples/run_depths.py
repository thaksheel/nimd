import torch
import pandas as pd
import warnings
import numpy as np
from matplotlib import pyplot as plt

from nimd import (
    Runner,
    RunnerParams,
    DeepNMFParams,
    quick_load_data,
    SSNMFParam
)
from nimd.interpretation import Interpretation


warnings.filterwarnings("ignore")

X, y, df, feature_names = quick_load_data(filenmae="./data/data.csv", type="regression")

params = DeepNMFParams(
    layers_depths=20,
    division_base=1.2,
    layers_rank=None,  # FIXME: this is used throughout deep.py so limits the flexibility of layers_depths
    lam=None,
    outerit=100,
    maxiter=100,
    rngseed=42,
    rho=10,
    display=False,
    accADMM=True,
    beta=1,
    min_vol=False,
    innerloop=1,
    maxIterADMM=70,
    normalize=2,
)
runner_params = RunnerParams(
    end=10,
    convert_type="division_base",
    init="nnsvdlrc",
    norm_X=None,
    rank=None,
    model="deep",
    task="regression",
    deep_params=params,
    multi_params=params,
    ssnmf_params=SSNMFParam(),
    dtype=torch.float64,
    device=torch.device("cpu"), # cuda is really unstable for performance
    val_train_test_splits=[0, 0.8, 0.2],
    display=True,
    rng=42,
    eval_type="full",  # does not work for PCA
    eps_stab=1e-8,
    return_tensor=False,
)
runner = Runner(X=X, y=y)
interpret = Interpretation()

# deepnmf minvol=False
# ranks, inits = [50], ["nndsvd", 'random']
ranks = [i for i in range(20, 70, 1)]
inits = ["nnsvdlrc", "random", "nndsvd", "nndsvda", "nndsvdar"]
results_d = []
results = []
for init in inits:
    runner_params.init = init
    depths_results, results_data = interpret.depths_tunner(
        ranks,
        runner,
        runner_params,
        depth_limit=10,
        weight=[0.7, 0.3],
    )
    results_d.extend(depths_results)
    df_depths = pd.DataFrame([r.__dict__ for r in results_d])
    results.append(runner.convert_results_to_df(results_data))
    df_results = pd.concat(results)
    df_depths.to_excel("./exports/depths_results3.xlsx")
    df_results.to_excel("./exports/super_results_deep3.xlsx")

# deepnmf minvol=True
runner_params.deep_params.min_vol = True
runner_params.deep_params.normalize = 3
runner_params.deep_params.outerit = 70
runner_params.deep_params.maxiter = 70
runner_params.eps_stab = 1e-8

print("==========> minvol strated")
results_d = []
results = []
for init in inits:
    runner_params.init = init
    depths_results, results_data = interpret.depths_tunner(
        ranks,
        runner,
        runner_params,
        depth_limit=10,
        weight=[0.7, 0.3],
    )
    results_d.extend(depths_results)
    df_depths = pd.DataFrame([r.__dict__ for r in results_d])
    results.append(runner.convert_results_to_df(results_data))
    df_results = pd.concat(results)
    df_depths.to_excel("./exports/depths_results_minvol1.xlsx")
    df_results.to_excel("./exports/super_results_deep_minvol1.xlsx")

print("END")
