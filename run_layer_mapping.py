import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch


from src import (
    Runner,
    RunnerParams,
    DeepNMFParams,
    quick_load_data,
)
from src.interpretation import Interpretation


X, y, df, feature_names = quick_load_data(filenmae="./data/data.csv", type="regression")

params = DeepNMFParams(
    layers_rank=[20, 10, 5],
    division_base=1.2,
    layers_depths=4,
    lam=None,
    outerit=50,
    maxiter=100,
    rngseed=42,
    rho=10,
    display=False,
    accADMM=True,
    beta=1,
    min_vol=False,
    innerloop=1,
    maxIterADMM=60,
    normalize=2,
)
runner_params = RunnerParams(
    rank=20,
    model="deep",
    task="regression",
    init="nnsvdlrc",
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
)
runner = Runner(X=X, y=y)
interpret = Interpretation()
results = runner.run_evaluation(runner_params, ranks=[10])
Hl = results[0].Hl

df_attribution = interpret.topk_element_attribution(
    Hs=Hl,
    threshold_by_layers=[0.01, 0.01, 0.01, 0.01],
    type_t="top_k",
    lookup=pd.read_excel("./data/lookup.xlsx"),
    top_k=3,
)
df_attribution.to_excel("./exports/atr_ly_map.xlsx")  # use to check Jaccard insights

# NOTE: Jaccard Matrix
Js = []
for i in range(len(Hl) - 1):
    Js.append(interpret.jaccard_matrix(Hl[i], Hl[i + 1], top_k=1, min_weight=0.0))

# NOTE: plotting Jaccard Matrix
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
[
    interpret.plot_hierarchy_map(
        J,
        type="jaccard",
        ax=axes[i],
        l1_name=f"Layer {i+1}",
        l2_name=f"Layer {i+2}",
    )
    for i, J in enumerate(Js)
]
axes[0].set_title("Layer 1 to 2")
axes[1].set_title("Layer 2 to 3")
axes[2].set_title("Layer 3 to 4")

plt.tight_layout()
plt.savefig("./exports/ly_map.png")
plt.show()


print("END")
