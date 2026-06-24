import json
from importlib.resources import files

from nimd import Runner, RunnerParams, quick_load_data
from nimd.cli import main


def test_packaged_benchmark_loads():
    data_path = files("nimd.datasets").joinpath("benchmark_heuslerene.csv")
    X, y, df, feature_names = quick_load_data(str(data_path), type="regression")
    assert X.shape[0] == y.shape[0] == len(df)
    assert X.shape[1] == len(feature_names)
    assert X.shape[0] > 0
    assert X.shape[1] > 0


def test_public_api_exports_runner_types():
    assert Runner is not None
    assert RunnerParams is not None


def test_input_template_can_be_generated(tmp_path):
    output = tmp_path / "nimd.in"
    assert main(["init-input", "--output", str(output)]) == 0
    text = output.read_text()
    assert "[run]" in text
    assert "models =" in text


def test_input_file_runs_smoke_workflow(tmp_path):
    input_file = tmp_path / "nimd.in"
    output_file = tmp_path / "result.json"
    input_file.write_text(
        f"""
[run]
mode = smoke
data = packaged
task = regression
models = pca
ranks = 4
outerit = 2
maxiter = 2
output = {output_file.name}

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
"""
    )
    assert main(["run", "--input", str(input_file)]) == 0
    payload = json.loads(output_file.read_text())
    assert payload["input"] == str(input_file.resolve())
    assert payload["mode"] == "smoke"
    assert payload["records"][0]["model"] == "pca"
    assert payload["records"][0]["rank"] == 4
