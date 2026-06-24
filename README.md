# Deep NMF for Materials Discovery

This repository contains the research code accompanying the manuscript
**"Harnessing Nonnegative Matrix Factorization for Advanced Computational
Materials Modeling"** by Thaksheel Alleck, Chinedu Ekuma, and Akwum Onwunta.

## Authors

Thaksheel Alleck <tna324@lehigh.edu><br>
Akwum Onwunta <onwuntajunior@gmail.com><br>
Chinedu Ekuma <cekuma1@gmail.com>

The code is organized around the algorithmic development of interpretable
Nonnegative Matrix Factorization (NMF) methods for scientific and materials
informatics data. The included Heuslerene data are a benchmark example used in
the manuscript, but the main purpose of the repository is broader: to compare,
extend, and interpret NMF-based representations for high-dimensional,
nonnegative materials descriptors.

## What This Repository Enables

The code supports an end-to-end workflow for developing and evaluating NMF
models:

1. Build a nonnegative feature matrix from scientific descriptors.
2. Initialize factor matrices with random or SVD-based initializers.
3. Fit standard, multilayer, deep, min-volume, or semi-supervised NMF models.
4. Compare NMF representations with a PCA baseline.
5. Evaluate learned latent factors on supervised regression or classification
   tasks.
6. Interpret factors through top-k feature attribution, layer mapping,
   Normalized Mutual Information (NMI), Jaccard overlap, entropy, and SHAP-based
   downstream importance.
7. Study robustness and identifiability under rank changes, depth changes,
   initialization choices, and controlled perturbations.

This makes the repository useful both as a manuscript reproduction package and
as a starting point for applying NMF algorithms to other nonnegative scientific
datasets.

## Algorithmic Scope

The repository includes implementations and experiment drivers for:

| Method | Purpose |
| --- | --- |
| PCA baseline | Non-NMF dimensionality-reduction reference model. |
| Beta-divergence NMF | NMF with beta/KL-style divergence objectives for non-Gaussian feature scales. |
| Frobenius-norm NMF | Classical NMF objective with multiple optimization routines. |
| Hierarchical rank-2 NMF | Recursive rank-2 splitting for coarse-to-fine factor discovery. |
| Multilayer NMF | Layer-wise NMF factorization for hierarchical representations. |
| Deep NMF | Jointly optimized multilayer NMF for multiscale latent structure. |
| Min-volume deep NMF | Deep NMF with min-volume regularization for improved identifiability and robustness. |
| Semi-supervised NMF | Joint representation learning using feature and label information. |

The current code exposes the following model names through the runner:

```text
pca
beta
fronorm
hier
multilayer
deep
ssnmf
```

Min-volume deep NMF is run as `deep` with `DeepNMFParams(min_vol=True)`; some
experiment scripts also use the label `minvol` as a convenience flag before
internally switching to the deep NMF implementation.

## Repository Structure

The repository is script-first and intentionally keeps the manuscript
experiments visible at the top level.

```text
.
|-- data/
|   `-- data.csv                    # Bundled benchmark data used by the examples
|-- nmfs/                           # Standalone/legacy NMF implementations
|   |-- betaNMF.py
|   |-- deepKLNMF.py
|   |-- FroNMF.py
|   |-- hierNMF.py
|   |-- multilayerKLNMF.py
|   |-- nnsvdlrc.py
|   |-- semiNMF.py
|   |-- sparseNMF.py
|   `-- ssnmf.py
|-- src/
|   |-- core/
|   |   |-- initiliazation.py       # NMF initialization routines
|   |   |-- selection.py            # Model selection and factorization dispatch
|   |   |-- sensitivity.py          # Sensitivity and perturbation utilities
|   |   `-- utils.py                # Dataclasses, data loading, scoring helpers
|   |-- deep/
|   |   |-- deep.py                 # Deep and min-volume deep NMF
|   |   |-- multilayer.py           # Multilayer KL-NMF
|   |   `-- utils.py                # Deep/multilayer parameter classes
|   |-- standard/
|   |   |-- beta.py                 # Beta-divergence NMF
|   |   |-- fronorm.py              # Frobenius-norm NMF
|   |   |-- hier.py                 # Hierarchical rank-2 NMF
|   |   |-- nnls.py                 # Nonnegative least-squares helper
|   |   `-- utils.py                # Semi-supervised NMF wrapper
|   |-- analyze_results.py          # Result summarization utilities
|   |-- exports.py                  # Export helpers for factors and layers
|   |-- interpretation.py           # Attribution, NMI, Jaccard, entropy tools
|   |-- runner.py                   # Main experiment driver
|   `-- supervised.py               # Regression/classification evaluation
|-- run.py                          # Deep NMF exploratory driver
|-- run_d.py                        # Batch supervised-evaluation driver
|-- run_depths.py                   # Deep NMF depth-selection experiments
|-- run_element_level_attribution.py # Factor attribution experiments
|-- run_ie.py                       # Depth/noise interpretation experiments
|-- run_layer_mapping.py            # Deep-layer mapping experiments
|-- run_minvol_identifiability.py   # Min-volume robustness experiments
|-- run_ssnmf.py                    # Semi-supervised NMF experiments
|-- run_supervised_results.py       # Main supervised benchmark driver
|-- data_analysis.py                # Result-table construction helpers
|-- evaluations.py                  # Older evaluation utilities
|-- periodic_map.py                 # Element/periodic-class mapping helpers
`-- quick_analyze.py                # Quick post-processing helpers
```

## Installation

Run the code from the repository root. The current repository does not define a
`setup.py` or `pyproject.toml`, so use a source checkout and install the Python
dependencies into a virtual environment.

```bash
git clone git@github.com:thaksheel/deep_nmf_materials_discovery.git
cd deep_nmf_materials_discovery

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn torch shap openpyxl
```

Optional dependency:

```bash
python -m pip install optuna
```

`optuna` is only needed for the fine-tuning utilities in `src/core/finetune.py`.

Before running scripts that write results, create the output folders:

```bash
mkdir -p exports exports/bin-results exports/layers
```

Most scripts are configured for CPU execution through `torch.device("cpu")`.
This is the recommended starting point for reproducibility. GPU execution may
require editing the relevant script and checking numerical stability for the
chosen system.

## Quick Smoke Check

After installing the dependencies, verify that the source tree imports and that
the bundled benchmark data can be loaded:

```bash
python - <<'PY'
from src import quick_load_data

X, y, df, feature_names = quick_load_data(
    filenmae="./data/data.csv",
    type="regression",
)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("number of features:", len(feature_names))
print("first five features:", feature_names[:5])
PY
```

Expected behavior: the command should print the feature matrix size, target
size, and feature names without raising an import error.

## Using the Core API

The top-level scripts are the easiest way to reproduce manuscript experiments,
but the algorithmic workflow can also be called directly.

```python
import torch

from src import (
    DeepNMFParams,
    Runner,
    RunnerParams,
    SSNMFParam,
    quick_load_data,
)

X, y, df, feature_names = quick_load_data(
    filenmae="./data/data.csv",
    type="regression",
)

deep_params = DeepNMFParams(
    layers_rank=[20, 10, 5],
    layers_depths=3,
    lam=None,
    outerit=50,
    maxiter=50,
    rngseed=42,
    rho=10,
    display=False,
    accADMM=True,
    beta=1,
    min_vol=False,
    normalize=2,
)

runner_params = RunnerParams(
    rank=20,
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
    eval_type="full",
    eps_stab=1e-7,
)

runner = Runner(X, y)
results = runner.run_evaluation(runner_params, ranks=[20, 40, 60])
summary = runner.convert_results_to_df(results)
print(summary)
```

To activate min-volume regularization, use:

```python
deep_params.min_vol = True
deep_params.normalize = 3
runner_params.eps_stab = 1e-6
```

## Main Parameter Choices

The central parameter objects are `DeepNMFParams`, `MultilayerParams`,
`RunnerParams`, and `SSNMFParam`.

Important options:

| Parameter | Meaning |
| --- | --- |
| `model` | One of `pca`, `beta`, `fronorm`, `hier`, `multilayer`, `deep`, or `ssnmf`. |
| `task` | `regression` or `classification`. |
| `ranks` | Factorization ranks evaluated by the runner. |
| `layers_depths` | Number of layers used by multilayer/deep NMF. |
| `layers_rank` | Explicit layer ranks, for example `[20, 10, 5]`. |
| `division_base` | Converts a single rank into decreasing layer ranks when `layers_rank=None`. |
| `convert_type` | `division_base` or `linspace` rank conversion in `Runner.convert_ranks_layers`. |
| `init` | One of `random`, `nndsvd`, `nndsvda`, `nndsvdar`, or `nnsvdlrc`. |
| `fronorm_algo` | One of `MUUP`, `ADMM`, `HALS`, `FPGM`, or `ALSH`. |
| `min_vol` | Enables min-volume regularization for deep NMF. |
| `normalize` | Normalization mode used by multilayer/deep updates; `3` is used with min-volume experiments. |
| `eps_stab` | Numerical stabilizer used by deep/multilayer routines. |
| `get_shap` | Computes SHAP values during supervised evaluation when enabled. |
| `perturb` and `noise_level` | Used for initialization-noise robustness studies. |

The manuscript experiments often sweep over ranks, initialization methods,
depths, and noise levels. Check the configuration block near the top of each
`run_*.py` script before launching a long job.

## Running Manuscript Experiments

Each script corresponds to a major algorithmic experiment or analysis route.
Run commands from the repository root.

| Goal | Script | Typical output |
| --- | --- | --- |
| Supervised comparison across NMF variants and PCA | `python run_supervised_results.py` | Excel tables under `exports/` |
| Deep NMF depth selection with NMI/performance scoring | `python run_depths.py` | `depths_results*.xlsx`, `super_results*.xlsx` |
| Element-level factor attribution | `python run_element_level_attribution.py` | Attribution and coherence/diversity tables |
| Hierarchical layer mapping | `python run_layer_mapping.py` | Layer-attribution workbook and layer-map figure |
| Min-volume identifiability and noise robustness | `python run_minvol_identifiability.py` | Noise-score, entropy, and robustness tables/plots |
| Semi-supervised NMF evaluation | `python run_ssnmf.py` | SSNMF score tables |
| Exploratory deep NMF runs | `python run.py` or `python run_d.py` | Experiment-specific Excel exports |

Some scripts contain manuscript-specific overrides. For example, a script may
define a broad list of models and then narrow it to one active model for a
particular result table. Inspect the final active values of `models`, `tasks`,
`ranks`, `inits`, `depths`, and output paths before running.

## Input Data Format

The algorithms operate on a nonnegative numerical matrix `X`. The bundled loader
`quick_load_data` is tailored to the included benchmark CSV and expects:

```text
Material
a
b
Composition
Stability
```

`quick_load_data` drops those columns from the feature matrix. For regression it
uses `a` as the target. For classification it maps `Stability` from
`stable`/`unstable` to `1`/`0`.

All remaining columns are treated as nonnegative features. In the bundled
benchmark, these include elemental composition features and electronic
descriptor features. To use another dataset, either:

1. Match this column convention, or
2. Write a small loader that returns `X`, `y`, `df`, and `feature_names`, then
   pass `X` and `y` directly into `Runner`.

## Interpretation Inputs

Several interpretation routines map factor loadings back to feature labels. For
element-level materials interpretation, functions such as `map_factors`,
`topk_element_attribution`, `run_layer_mapping.py`, and
`run_minvol_identifiability.py` expect a lookup workbook at:

```text
data/lookup.xlsx
```

That lookup table should contain these columns:

```text
i
elements
class
```

where `i` is the zero-based feature index, `elements` is the feature or element
label, and `class` is the user-defined feature group. The same mechanism can be
adapted to non-elemental datasets by replacing `elements` and `class` with
domain-specific feature names and groups.

## Outputs

Experiment scripts write most results to `exports/`.

Common outputs include:

| Output type | Description |
| --- | --- |
| Supervised score tables | R2, RMSE, MAE, accuracy, F1, runtime, rank, model, initialization. |
| Depth-selection tables | Candidate depths, NMI values, supervised scores, combined depth scores. |
| Attribution tables | Top-k feature/factor mappings with supervised importance. |
| Layer maps | Jaccard-style relationships across adjacent deep NMF layers. |
| Robustness tables | Noise-level, seed, rank, RMSE deviation, entropy deviation, and factor-label stability. |

The exact output filenames are defined inside each script.

## Reproducibility Notes

- Run scripts from the repository root so imports such as `from src import *`
  and paths such as `./data/data.csv` resolve correctly.
- The code uses fixed random seeds in the provided experiment scripts, but NMF
  methods can still show numerical variation across hardware, BLAS libraries,
  PyTorch versions, and CPU/GPU settings.
- The default split convention in the experiment scripts is
  `val_train_test_splits=[0, 0.8, 0.2]`.
- Some manuscript experiments are computationally expensive, especially
  min-volume deep NMF and robustness sweeps across many ranks, seeds, and noise
  levels.
- The repository currently includes research scripts rather than a formal test
  suite. A practical validation path is to run the smoke check, then run a small
  rank subset, and only then launch the full manuscript sweeps.

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**

Run the command from the repository root.

**`FileNotFoundError` for `exports/...`**

Create output folders before running:

```bash
mkdir -p exports exports/bin-results exports/layers
```

**`FileNotFoundError: ./data/lookup.xlsx`**

The factorization and supervised benchmarks do not need `lookup.xlsx`, but
element-level attribution and layer-mapping scripts do. Add a lookup workbook
with columns `i`, `elements`, and `class`.

**A script takes a long time**

Reduce `ranks`, `inits`, `depths`, `seeds`, or `noise_levels` in the script's
configuration block before launching the full sweep.

**CUDA/GPU results are unstable**

Start with the CPU configuration already used in the scripts. If GPU execution
is needed, change the `device` setting deliberately and compare against a small
CPU run.

## Citation

If you use this code, please cite the accompanying manuscript:

```text
Thaksheel Alleck, Chinedu Ekuma, and Akwum Onwunta.
Harnessing Nonnegative Matrix Factorization for Advanced Computational
Materials Modeling.
```

Add the final journal, DOI, or preprint information once it is available.

## Repository

GitHub: https://github.com/thaksheel/deep_nmf_materials_discovery
