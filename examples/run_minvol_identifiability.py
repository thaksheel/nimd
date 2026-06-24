import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from nimd import (
    Runner,
    RunnerParams,
    DeepNMFParams,
    quick_load_data,
)
from nimd.interpretation import Interpretation
from nimd import *

X, y, df, feature_names = quick_load_data(filenmae="./data/data.csv", type="regression")

params = DeepNMFParams(
    layers_rank=[20, 10, 5],
    lam=None,
    outerit=50,
    maxiter=100,
    rngseed=42,
    rho=10,
    display=False,
    accADMM=True,
    beta=1,
    min_vol=True,
    innerloop=1,
    maxIterADMM=60,
    normalize=3,
)
runner_params = RunnerParams(
    rank=20,
    model="deep",
    task="regression",
    init="nndsvda",
    ssnmf_params=SSNMFParam(),
    deep_params=params,
    multi_params=params,
    dtype=torch.float64,
    device=torch.device("cpu"),
    val_train_test_splits=[0, 0.8, 0.2],
    display=True,
    rng=42,
    eval_type="feature",  # does not work for PCA
    eps_stab=1e-6,
    perturb=True,
    noise_level=0.1,
)
runner = Runner(X=X, y=y)

ranks = [r for r in range(45, 50, 1)]
seeds = [r for r in range(42, 52, 1)]
noise_levels = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4, 8, 16]
# ranks = [45, 46]
# seeds = [42, 43]
# noise_levels = [0, 1, 8, 16, 32, 64]

# NOTE: ----> Running
results_un = runner.exp_minvol_identifiability(
    runner_params, seeds, noise_levels, ranks, method="unconstraint"
)  # FIXME: for some reason the effect of nl is not showing up
results = runner.exp_minvol_identifiability(
    runner_params, seeds, noise_levels, ranks, method="minvol"
)
# shape=(seed, noise, rank)

# NOTE: factor mapping influence
interpret = Interpretation()
df_importance = interpret.map_factors(
    results=results.reshape(-1),
    lookup_file="./data/lookup.xlsx",
)
df_importance_un = interpret.map_factors(
    results=results_un.reshape(-1),
    lookup_file="./data/lookup.xlsx",
)

# NOTE: scores impact
scores_minvol = interpret.performance_deviation(results)
scores_un = interpret.performance_deviation(results_un)
df_minvol = interpret.results_tensor_to_df(results)
df_un = interpret.results_tensor_to_df(results_un)
df_minvol.to_excel("./exports/noise_scores_results_minvol1.xlsx")
df_un.to_excel("./exports/noise_scores_results_un1.xlsx")

interpret.plot_noise_score_influence(results, results_un, noise_levels)

# NOTE: entropy by noise_levels
interpret.plot_entropy_noise(
    interpret.compute_entropy(df_importance, ranks, noise_levels, seeds),
    interpret.compute_entropy(df_importance_un, ranks, noise_levels, seeds),
    noise_lvl=noise_levels,
)


print("END")
