import torch
import warnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Literal

from src import (
    Runner,
    RunnerParams,
    DeepNMFParams,
    quick_load_data,
)
from src.interpretation import Interpretation
from src import ResultData

warnings.filterwarnings("ignore", category=Warning)


def plot_performance_by_depths(results: List[ResultData], runner_params, params):
    s = np.array([r.depths for r in results])
    y = np.array([r.score for r in results])
    x = np.array([r.rank for r in results])

    plt.figure()
    plt.scatter(x, y, s=s * 25, edgecolors="black", alpha=0.7)
    plt.xlabel("ranks")
    plt.ylabel("R^2")
    plt.title(
        f"ranks=[20, 40, 50, 70] for depths=[2...10] init={runner_params.init} \n division_base={params.division_base}"
    )
    plt.show()


def collect_data(
    results: List[ResultData],
    score_nmi: np.ndarray,
    outfile: str,
    convert_type: str,
    end_rank: int,
    division_base: float,
):
    data = {
        "nmf_model": [r.nmf_model for r in results],
        "init": [r.init for r in results],
        "convert_type": [convert_type for _ in results],
        "rank": [r.rank for r in results],
        "depths": [r.depths for r in results],
        "score": [r.score for r in results],
        "runtime": [r.runtime for r in results],
        "nmi_last": [interpret._nmi_scores(r.Hl).tolist()[-1] for r in results],
        "depths_score": score_nmi,
    }
    if end_rank:
        data["end_rank"] = [end_rank for _ in results]
    if division_base:
        data["division_base"] = [division_base for _ in results]

    df = pd.DataFrame(data)
    if outfile:
        df.to_excel(outfile, index=True)
    return df


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
    maxIterADMM=100,
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
    score_type="r2",
    deep_params=params,
    multi_params=params,
    dtype=torch.float64,
    device=torch.device("cpu"),
    val_train_test_splits=[0, 0.8, 0.2],
    display=True,
    rng=42,
    eval_type="full",  # does not work for PCA
    eps_stab=1e-7,
    return_tensor=False,
)
runner = Runner(X=X, y=y)

inits = ["nnsvdlrc", "random", "nndsvd", "nndsvda", "nndsvdar"]
ranks = [i for i in range(10, 61, 10)]
ranks, inits, depths = [70], ["nnsvdlrc"], [30]
depths = [i for i in range(2, 14, 1)]
for init in inits:
    runner_params.init = init
    for i, depth in enumerate(depths):
        runner_params.deep_params.set_layer_depth(depth)
        results = runner.run_evaluation(params=runner_params, ranks=ranks)
        # FIXME: need to reset lam each time depth is updated. lam should be a class attribute in DeepNMF instead of DeepNMFParams
        runner_params.deep_params.lam = None
        interpret = Interpretation()
        sc = interpret.depths_ratios(results=results, out_type="score")
        er = interpret.depths_ratios(results=results, out_type="errors")
        sc_nmi = interpret.depths_scores(
            results=results,
            weight=[0.7, 0.3],
            eps=1e-9,
            penalty=0.1,
        )
        df = collect_data(
            results,
            score_nmi=sc_nmi,
            convert_type=runner_params.convert_type,
            # outfile="./exports/results_depths5.xlsx",
            outfile=None,
            division_base=(
                params.division_base
                if runner_params.convert_type == "division_base"
                else None
            ),
            end_rank=(
                runner_params.end if runner_params.convert_type == "linspace" else None
            ),
        )

nmis = [interpret._nmi_scores(r.Hl) for r in results]
depths_thresholds = [np.where(nmi > 0.5)[0] for nmi in nmis]
depths_ranks = df.loc[df.groupby("rank")["depths_score"].idxmax()]
print(df.sort_values(by=["depths_score"], ascending=[False]).head())

# NOTE: Interpretation verification
dfs_attribution = interpret.topk_element_attribution(
    Hs=results[2].Hl,
    ranks=results[2].ranks,
    layers=[1, 2, 3, 4],
    lookup=pd.read_excel("./data/lookup.xlsx"),
    threshold_by_layers=[0.05 for _ in range(4)],
    outpath="./exports/",
    nmf="deep",
    type_t="threshold",
    top_k=3,
    task="reg",
)

count_tables = []
for i, df in enumerate(dfs_attribution):
    counts = df["metal_class"].value_counts()
    counts.name = f"df_{i}"  # name the column
    count_tables.append(counts)
df_counts = pd.concat(count_tables, axis=1).fillna(0).astype(int)


print("END")
