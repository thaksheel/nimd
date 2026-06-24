from importlib.resources import files

from nimd import Runner, RunnerParams, quick_load_data


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
