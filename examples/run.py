import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import List

from nimd import Runner, RunnerParams, DeepNMFParams, quick_load_data, ResultData

# warnings.filterwarnings("ignore", category=Warning)


def plot_errors(
    errors: np.ndarray,
    layer: int,
    depth: int,
    rank: int,
):
    plt.figure()
    plt.plot(errors, marker="o")
    plt.xlabel("iterations")
    plt.ylabel("errors")
    plt.title(f"layer={layer} rank={rank}, depth={depth}")
    plt.show()
    return True


def export_data(results: List[ResultData], outfile: str):
    data = {
        "nmf_model": [r.nmf_model for r in results],
        "init": [r.init for r in results],
        "rank": [r.rank for r in results],
        "depths": [r.depths for r in results],
        "eps": [r.eps for r in results],
        "score": [r.score for r in results],
        "runtime": [r.runtime for r in results],
    }
    df = pd.DataFrame(data)
    if outfile:
        df.to_excel(outfile)
    return df


X, y, df, feature_names = quick_load_data(filenmae="./data/data.csv", type="regression")

params = DeepNMFParams(
    layers_depths=20,
    division_base=1.15,
    layers_rank=None,  # FIXME: this is used throughout deep.py so limits the flexibility of layers_depths
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
runner_params = RunnerParams(
    end=10,
    convert_type="linspace",
    init="nnsvdlrc",
    norm_X=None,
    rank=None,
    model="deep",
    task="regression",
    score_type="r2",
    deep_params=params,
    multi_params=params,
    dtype=torch.float64,
    device=torch.device("cpu"),
    val_train_test_splits=[0, 0.8, 0.2],
    display=True,
    rng=42,
    eval_type="full",  # does not work for PCA
    eps_stab=1e-7,  # will result in -ve R2 and controls R2 performance! used to be 1e-9
)
runner = Runner(X=X, y=y)

depths = [2, 3, 4]
ranks = [20, 40, 60]
epss = [1e-7, 1e-10, 1e-12]
inits = ["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"]
inits = ["nnsvdlrc"]
for i, eps in enumerate(epss):
    print(f"-----> i={i} eps={eps}")
    for init in inits:
        runner_params.init = init
        runner_params.eps_stab = eps
        for i, depth in enumerate(depths):
            runner_params.deep_params.set_layer_depth(depth)
            results = runner.run_evaluation(params=runner_params, ranks=ranks)
            # FIXME: need to reset lam each time depth is updated. lam should be a class attribute in DeepNMF instead of DeepNMFParams
            runner_params.deep_params.lam = None
            # df = export_data(results, outfile="./exports/results_eps2.xlsx")
# [
#     plot_errors(errors=results[0].error[:, i], layer=i, depth=6, rank=60)
#     for i in range(6)
# ]

print("END")
