# NIMD: Nonnegative Interpretable Matrix Decomposition

NIMD is a Python research package for developing, benchmarking, and
interpreting Nonnegative Matrix Factorization (NMF) methods for scientific
representation learning. It accompanies the manuscript **"Harnessing
Nonnegative Matrix Factorization for Advanced Computational Materials
Modeling"**, but the package name and structure are intentionally algorithmic:
the code is designed around interpretable nonnegative decompositions, not only
around one demonstration dataset.

The import package and executable are both named:

```text
nimd
```

NIMD stands for **Nonnegative Interpretable Matrix Decomposition**.

## Authors

Thaksheel Alleck <tna324@lehigh.edu><br>
Akwum Onwunta <onwuntajunior@gmail.com><br>
Chinedu Ekuma <cekuma1@gmail.com>

## Scientific Motivation

Many scientific datasets are naturally nonnegative: elemental fractions,
spectral intensities, diffraction signals, microscopy counts, kinetic
observables, and other additive descriptors often cannot be meaningfully
negative. Classical dimensionality reduction methods such as PCA are useful and
fast, but their signed components can mix physically unrelated contributions and
make interpretation difficult.

NMF-based methods address a different scientific need. They seek additive,
parts-based factors that can often be mapped back to physical motifs,
composition families, feature groups, or evolving structures. In materials
design and related domains, this matters because a representation is useful not
only when it improves a metric, but also when it helps researchers ask better
questions: which features drive a prediction, which motifs persist across
model depth, which factors are stable under initialization noise, and which
representations remain chemically or physically meaningful.

NIMD therefore treats prediction and interpretation as linked tasks. The code
can compare NMF representations with PCA in regression and classification, but
its central contribution is the ability to study latent factors through
nonnegativity, hierarchy, attribution, overlap, entropy, and identifiability.
That makes the framework relevant to materials informatics, spectroscopy,
composition-property modeling, dynamical datasets with nonnegative states, and
other scientific settings where interpretable low-rank structure is more useful
than a purely opaque embedding.

## What NIMD Provides

NIMD supports an end-to-end workflow:

1. Load a nonnegative feature matrix and supervised target.
2. Initialize factors with random or SVD-based NMF initializers.
3. Fit standard, multilayer, deep, min-volume, or semi-supervised NMF models.
4. Compare learned NMF representations with a PCA baseline.
5. Evaluate latent factors on regression or classification tasks.
6. Interpret factors through top-k feature attribution, SHAP-based downstream
   importance, NMI, Jaccard overlap, entropy, and layer mapping.
7. Test robustness under rank changes, depth changes, initialization choices,
   and controlled perturbations.

## Algorithms

| Method | Role in NIMD |
| --- | --- |
| PCA baseline | Signed low-rank reference model for comparison. |
| Beta-divergence NMF | NMF for non-Gaussian or scale-sensitive nonnegative data. |
| Frobenius-norm NMF | Classical Euclidean NMF with multiple optimization routines. |
| Hierarchical rank-2 NMF | Recursive coarse-to-fine factor discovery. |
| Multilayer NMF | Layer-wise nonnegative factorization for hierarchical structure. |
| Deep NMF | Jointly optimized multilayer decomposition. |
| Min-volume deep NMF | Deep NMF with a geometric regularizer for identifiability and stability. |
| Semi-supervised NMF | Joint use of feature and label information in the factorization. |

The public runner currently accepts these model names:

```text
pca
beta
fronorm
hier
multilayer
deep
ssnmf
```

Min-volume deep NMF is activated by setting `DeepNMFParams(min_vol=True)`.
The command line interface also accepts `minvol` as a convenience model label.

## Repository Layout

```text
.
|-- nimd/                         # Installable Python package
|   |-- core/                     # Dataclasses, initialization, selection, utilities
|   |-- deep/                     # Multilayer, deep, and min-volume deep NMF
|   |-- standard/                 # Beta, Frobenius, hierarchical, NNLS, SSNMF wrappers
|   |-- analysis/                 # Result analysis and post-processing helpers
|   |-- datasets/                 # Packaged benchmark CSV used by the CLI
|   |-- legacy/                   # Older standalone research modules retained for reproducibility
|   |-- cli.py                    # `nimd` command line interface
|   |-- interpretation.py         # Attribution, NMI, Jaccard, entropy, layer mapping
|   |-- runner.py                 # Main experiment driver
|   |-- supervised.py             # Regression/classification evaluation
|   `-- exports.py                # Export and plotting helpers
|-- examples/                     # Manuscript and research driver scripts
|-- tests/                        # Lightweight import/data smoke tests
|-- data/                         # Repository copy of the benchmark dataset
|-- src/                          # Compatibility alias for older `from src import ...`
|-- nmfs/                         # Compatibility alias for older `import nmfs...`
|-- nimd.in                       # Unified input file for command-line runs
|-- setup.cfg                     # Package metadata and dependencies
|-- setup.py                      # Setuptools entry point
|-- pyproject.toml                # Modern build-system declaration
|-- MANIFEST.in                   # Source distribution file rules
`-- README.md
```

## Installation

The primary installation method is `pip install .`.

```bash
git clone git@github.com:thaksheel/deep_nmf_materials_discovery.git
cd deep_nmf_materials_discovery

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

For optional hyperparameter tuning utilities:

```bash
python -m pip install ".[tuning]"
```

For a fuller research environment with tuning and notebook support:

```bash
python -m pip install ".[full]"
```

The manual dependency installation pattern below is no longer the recommended
path:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn torch shap openpyxl
```

It is replaced by the package metadata in `setup.cfg`, so `pip install .`
installs the required runtime dependencies directly.

## Quick Checks

After installation, verify the executable:

```bash
nimd --version
```

Print the packaged benchmark data path:

```bash
nimd data-path
```

Run a one-model smoke test using the packaged benchmark data:

```bash
nimd smoke
```

Run the same package through the unified input file:

```bash
nimd run --input nimd.in
```

To create a fresh editable input file after installing NIMD:

```bash
nimd init-input --output nimd.in
```

Run a compact benchmark across representative models:

```bash
mkdir -p exports
nimd benchmark --models pca beta fronorm hier multilayer deep --rank 4 --output exports/quick_benchmark.json
```

The compact benchmark is intentionally small. It is meant to verify that the
installed package works, not to reproduce the full manuscript sweeps.

## Python API Example

```python
import torch

from nimd import (
    DeepNMFParams,
    Runner,
    RunnerParams,
    SSNMFParam,
    quick_load_data,
)

X, y, df, feature_names = quick_load_data(
    filenmae="data/data.csv",
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

To use min-volume deep NMF:

```python
deep_params.min_vol = True
deep_params.normalize = 3
runner_params.eps_stab = 1e-6
```

## Running the Research Examples

The longer manuscript-style scripts live in `examples/`. Run them from the
repository root after installing the package:

```bash
mkdir -p exports exports/bin-results exports/layers
python examples/run_supervised_results.py
```

Common entry points:

| Goal | Command |
| --- | --- |
| Supervised comparison across NMF variants and PCA | `python examples/run_supervised_results.py` |
| Deep NMF depth selection | `python examples/run_depths.py` |
| Element-level attribution | `python examples/run_element_level_attribution.py` |
| Deep-layer mapping | `python examples/run_layer_mapping.py` |
| Min-volume identifiability and robustness | `python examples/run_minvol_identifiability.py` |
| Semi-supervised NMF experiments | `python examples/run_ssnmf.py` |
| Exploratory deep NMF runs | `python examples/run.py` or `python examples/run_d.py` |

Some examples are intentionally expensive. Before launching a full sweep, check
the active values of `models`, `tasks`, `ranks`, `inits`, `depths`, `seeds`,
`noise_levels`, and output paths near the top of each script.

## Input Data Format

NIMD uses two kinds of input:

1. A control file named `nimd.in`, which tells the executable what to run.
2. A CSV data file, which supplies the feature matrix and supervised target.

The recommended command-line workflow is:

```bash
nimd run --input nimd.in
```

The repository already includes `nimd.in`. Installed users can also generate a
new copy:

```bash
nimd init-input --output nimd.in
```

The input file is plain text and uses INI sections. Lines beginning with `#` or
`;` are comments. Relative paths are resolved from the folder containing the
input file, so `output = exports/results.json` writes beside the input file
unless an absolute path is supplied.

### `nimd.in` Example

```ini
[run]
mode = benchmark
data = packaged
task = regression
models = pca, beta, fronorm, hier, multilayer, deep
ranks = 4
outerit = 2
maxiter = 2
output = exports/nimd_input_benchmark.json

[model]
init = auto
eval_type = auto
fronorm_algo = HALS
modelname = rf
dtype = float64
device = cpu
rng = 42
get_shap = false
display = false
cv = false
return_tensor = false
val_train_test_splits = 0, 0.8, 0.2
eps_stab = auto
convert_type = division_base
end = 10
norm_X = none
norm_init = none
perturb = false
noise_level = none

[deep]
layers_depths = 2
division_base = 2
lam = none
rho = 10
beta = 1
min_vol = false
normalize = 2
innerloop = 1
maxIterADMM = 2
accADMM = true
epsi = 1e-4
HnormType = rows

[ssnmf]
model_num = 4
split = 0.2
tol = 1e-4
numiters = 2
iter_s = 2
lam = 1e-3
seed = 42
```

### `[run]` Tags

| Tag | Meaning |
| --- | --- |
| `mode` | `smoke` runs the first listed model; `benchmark` runs all listed models. |
| `data` | `packaged` uses the benchmark CSV shipped with NIMD. A path such as `data/data.csv` uses a user-supplied CSV. |
| `task` | `regression` uses the numeric target column; `classification` uses the stability labels. |
| `models` | Comma-separated list from `pca`, `beta`, `fronorm`, `hier`, `multilayer`, `deep`, `ssnmf`, and `minvol`. |
| `ranks` | One or more positive integer factorization ranks, for example `4` or `10, 20, 40`. |
| `outerit` | Outer alternating-optimization iterations for deep NMF style models. |
| `maxiter` | Inner or standard optimization iterations. Keep small for checks; increase for research runs. |
| `output` | JSON output path. Use `none` to print JSON to the terminal. |

### `[model]` Tags

| Tag | Meaning |
| --- | --- |
| `init` | Initialization method. `auto` uses `random` for PCA/hierarchical NMF and `nndsvd` otherwise. |
| `eval_type` | `auto`, `feature`, or `full`. `auto` uses feature evaluation for PCA and full reconstruction evaluation otherwise. |
| `fronorm_algo` | Optimization routine for Frobenius NMF: `MUUP`, `ADMM`, `HALS`, `FPGM`, or `ALSH`. |
| `modelname` | Downstream supervised model: commonly `rf`, `linreg`, `logit`, or `mlp`. |
| `dtype` | PyTorch numerical precision, currently `float64` or `float32`. |
| `device` | PyTorch device string. Start with `cpu` for reproducible checks. |
| `rng` | Random seed used by initialization and supervised splits. |
| `get_shap` | `true` computes SHAP values during supervised evaluation. It is slower. |
| `display` | `true` prints per-model progress while running. |
| `cv` | Enables cross-validation in the supervised learner when supported. |
| `return_tensor` | Keeps tensor outputs from deep routines instead of converting to arrays. |
| `val_train_test_splits` | Three comma-separated split values. The default is `0, 0.8, 0.2`. |
| `eps_stab` | Numerical stabilizer. `auto` uses `1e-6` for min-volume runs and `1e-7` otherwise. |
| `convert_type` | Rank-to-layer conversion rule: `division_base` or `linspace`. |
| `end` | Final rank value used when `convert_type = linspace`. |
| `norm_X` | Optional feature normalization, such as `minmax`, `standard`, or `none`. |
| `norm_init` | Optional initialization normalization, such as `scaling`, `feature_norm`, or `none`. |
| `perturb` | `true` applies initialization perturbation for robustness studies. |
| `noise_level` | Perturbation noise level. Use `none` unless `perturb = true`. |

### `[deep]` Tags

| Tag | Meaning |
| --- | --- |
| `layers_depths` | Number of layers for multilayer and deep NMF. |
| `division_base` | Divides the first rank to create lower layer ranks when `convert_type = division_base`. |
| `lam` | Optional comma-separated deep regularization weights, or `none`. |
| `rho` | ADMM penalty parameter for min-volume deep NMF. |
| `beta` | Beta-divergence setting used by relevant NMF updates. |
| `min_vol` | `true` enables min-volume regularization for deep NMF. The `minvol` model label also enables this automatically. |
| `normalize` | Deep NMF normalization mode. Use `3` for min-volume runs. |
| `innerloop` | Inner-loop count used by min-volume ADMM updates. |
| `maxIterADMM` | Maximum ADMM iterations for min-volume updates. |
| `accADMM` | `true` uses accelerated ADMM where available. |
| `epsi` | Deep/multilayer convergence tolerance. |
| `HnormType` | H normalization direction, usually `rows` or `cols`. |

### `[ssnmf]` Tags

| Tag | Meaning |
| --- | --- |
| `model_num` | Semi-supervised NMF variant number. Supported research values are `3`, `4`, `5`, or `6`. |
| `split` | Semi-supervised train/test split fraction. |
| `tol` | Semi-supervised stopping tolerance. |
| `numiters` | Semi-supervised optimization iterations. |
| `iter_s` | Iterations for mapping held-out data into the learned semi-supervised representation. |
| `lam` | Semi-supervised label-weighting parameter. |
| `seed` | Random seed for semi-supervised NMF. |

### CSV Data File

The core algorithms operate on a nonnegative numerical matrix `X`. The bundled
loader `quick_load_data` expects the benchmark CSV convention:

```text
Material
a
b
Composition
Stability
```

Those columns are excluded from the feature matrix. For regression, `a` is used
as the target. For classification, `Stability` is mapped from
`stable`/`unstable` to `1`/`0`. All remaining columns are treated as
nonnegative features.

To use another dataset with `nimd run`, keep this CSV convention and set
`data = path/to/your_data.csv` in `nimd.in`. For a different schema, write a
small loader that returns `X`, `y`, `df`, and `feature_names`, then pass `X` and
`y` directly to `Runner` through the Python API.

## Important Parameters

| Parameter | Meaning |
| --- | --- |
| `model` | One of `pca`, `beta`, `fronorm`, `hier`, `multilayer`, `deep`, or `ssnmf`. |
| `task` | `regression` or `classification`. |
| `ranks` | Factorization ranks evaluated by the runner. |
| `layers_depths` | Number of layers in multilayer/deep NMF. |
| `layers_rank` | Explicit layer ranks, for example `[20, 10, 5]`. |
| `division_base` | Converts one rank into decreasing layer ranks when `layers_rank=None`. |
| `convert_type` | `division_base` or `linspace` in `Runner.convert_ranks_layers`. |
| `init` | `random`, `nndsvd`, `nndsvda`, `nndsvdar`, or `nnsvdlrc`. |
| `fronorm_algo` | `MUUP`, `ADMM`, `HALS`, `FPGM`, or `ALSH`. |
| `min_vol` | Enables min-volume regularization for deep NMF. |
| `normalize` | Deep/multilayer normalization mode; `3` is used for min-volume runs. |
| `eps_stab` | Numerical stabilizer for deep/multilayer updates. |
| `get_shap` | Computes SHAP values during supervised evaluation. |
| `perturb`, `noise_level` | Initialization-noise robustness controls. |

## Interpretation Inputs

Factor attribution and layer mapping can use a lookup workbook:

```text
data/lookup.xlsx
```

The workbook should contain:

```text
i
elements
class
```

where `i` is the zero-based feature index, `elements` is the feature label, and
`class` is a user-defined feature group. For non-materials datasets, the same
scheme can represent any domain-specific feature names and categories.

The factorization and supervised benchmarks do not require `lookup.xlsx`.
Element-level attribution and layer-mapping examples do require it.

## Outputs

The example scripts usually write to `exports/`.

Common outputs include:

| Output | Description |
| --- | --- |
| Supervised score tables | R2, RMSE, MAE, accuracy, F1, runtime, rank, model, initialization. |
| Depth-selection tables | Candidate depths, NMI values, supervised scores, combined depth scores. |
| Attribution tables | Top-k feature/factor mappings with supervised importance. |
| Layer maps | Jaccard-style relationships across adjacent deep NMF layers. |
| Robustness tables | Noise-level, seed, rank, RMSE deviation, entropy deviation, factor-label stability. |

## Development Checks

Compile the package and examples:

```bash
python -m compileall nimd examples tests
```

Run smoke tests:

```bash
python -m pytest
```

Build source and wheel distributions:

```bash
python -m build
```

## Reproducibility Notes

- The package now imports as `nimd`. The old `src` and `nmfs` names are retained
  only as compatibility aliases.
- CPU execution is the default in the examples and CLI. Start there before
  experimenting with GPU execution.
- NMF solutions can vary with initialization, rank, hardware, BLAS libraries,
  and PyTorch versions. Compare compact checks first, then run full sweeps.
- Min-volume deep NMF and robustness experiments are more expensive than the
  compact CLI benchmark.

## Citation

If you use this code, please cite the accompanying manuscript:

```text
Thaksheel Alleck, Chinedu Ekuma, and Akwum Onwunta.
Harnessing Nonnegative Matrix Factorization for Advanced Computational
Materials Modeling.
```

Add final journal, DOI, or preprint information once available.

## Repository

GitHub: https://github.com/thaksheel/deep_nmf_materials_discovery
