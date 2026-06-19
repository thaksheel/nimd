import pandas as pd
import torch

from src import (
    Runner,
    RunnerParams,
    DeepNMFParams,
    ResultData,
    quick_load_data,
)



def export_results_toexcel(outpath: str, results: list[list[ResultData]], **kwargs):
    if "fronorm_algo" in kwargs:
        algos = kwargs["fronorm_algo"]
    else:
        algos = None
    data = [r.__dict__ for result in results for r in result]
    if algos:
        # TODO: fix the logic here for export for fronorm 
        current_init = kwargs["init"]
        algo_col = []
        for init in kwargs["inits"]:
            for algo in algos:
                for _ in kwargs["ranks"]:
                    algo_col.append(algo)
                if current_init == init:
                    break
    df = pd.DataFrame(data)
    df.drop(columns=["W", "H", "Wl", "Hl"], inplace=True)
    if results[0][0].nmf_model == "fronorm":
        df["fronorm_algo"] = algo_col
    if outpath:
        df.to_excel(outpath, index=False)
    return df


X, y, df, feature_names = quick_load_data(filenmae="./data/data.csv", type="regression")

params = DeepNMFParams(
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
runner_params = RunnerParams(
    rank=10,
    model="deep",
    task="regression",
    deep_params=params,
    multi_params=params,
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
ranks = [20, 40, 60]
algos = ["FPGM", "MUUP", "ADMM", "HALS", "ALSH"]
models = ["pca", "beta", "hier", "multilayer", "fronorm", "deep"]
models = ["pca"]
inits = ["nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc", "random"]
inits = ["random"]
for model in models:
    results = []
    runner_params.eval_type = (
        "feature" if model == "pca" else "full"
    )  # NOTE: only deepnmf needs full for better scores
    for init in inits:
        runner_params.init = init
        runner_params.model = model
        for algo in algos:
            runner_params.fronorm_algo = algo
            runner = Runner(X=X, y=y)
            results.append(runner.run_evaluation(params=runner_params, ranks=ranks))
            if model != "fronorm":
                break
        outfile = f"./exports/bin-results/results_{model}_minvol0.xlsx"
        df_results = export_results_toexcel(
            results=results,
            outpath=outfile,
            fronorm_algo=algos,
            ranks=ranks,
            inits=inits,
            init=init,
        )
        if model == "pca":
            break
print(df_results.head())

print("END")
