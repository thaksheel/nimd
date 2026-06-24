"""Command line interface for NIMD."""

from __future__ import annotations

import argparse
import configparser
import json
import time
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterable

import torch

from . import __version__
from .core.utils import SSNMFParam, RunnerParams, quick_load_data
from .deep.utils import DeepNMFParams
from .runner import Runner


def _packaged_data():
    return files("nimd.datasets").joinpath("benchmark_heuslerene.csv")


def _packaged_input():
    return files("nimd.templates").joinpath("nimd.in")


def _model_list(raw: Iterable[str]) -> list[str]:
    models: list[str] = []
    for item in raw:
        models.extend(part.strip() for part in item.split(",") if part.strip())
    return models


def _split_values(raw: str | None) -> list[str]:
    """Split comma-separated input values while allowing blank fields."""
    if raw is None:
        return []
    return [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]


def _is_auto(raw: str | None) -> bool:
    """Return True for empty, auto, and default input values."""
    return raw is None or raw.strip().lower() in {"", "auto", "default"}


def _is_none(raw: str | None) -> bool:
    """Return True for empty values that should become Python None."""
    return raw is None or raw.strip().lower() in {"", "none", "null", "nil"}


def _parse_bool(raw: str | None, default: bool) -> bool:
    """Parse common true/false spellings used in the input file."""
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected a boolean value, got {raw!r}")


def _parse_optional_float(raw: str | None) -> float | None:
    """Parse a float or return None for none/null input values."""
    if _is_none(raw):
        return None
    return float(raw)


def _parse_optional_float_list(raw: str | None) -> list[float] | None:
    """Parse comma-separated floats or return None."""
    if _is_none(raw):
        return None
    return [float(value) for value in _split_values(raw)]


def _parse_float_list(raw: str | None, default: list[float]) -> list[float]:
    """Parse comma-separated floats with a default fallback."""
    values = _split_values(raw)
    return [float(value) for value in values] if values else default


def _parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    """Parse comma-separated integers with a default fallback."""
    values = _split_values(raw)
    return [int(value) for value in values] if values else default


def _resolve_path(raw: str | None, input_path: Path) -> Path | None:
    """Resolve paths in the input file relative to the input file location."""
    if _is_none(raw):
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return input_path.parent / path


def _read_input_file(input_path: Path) -> configparser.ConfigParser:
    """Load and minimally validate a NIMD .in file."""
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not config.read(input_path):
        raise FileNotFoundError(f"could not read input file: {input_path}")
    if not config.has_section("run"):
        raise ValueError("input file must define a [run] section")
    return config


def _config_value(
    config: configparser.ConfigParser,
    section: str,
    key: str,
    default: str | None = None,
) -> str | None:
    """Read an input value without requiring every section to exist."""
    if config.has_option(section, key):
        return config.get(section, key)
    return default


def _input_settings(config: configparser.ConfigParser, input_path: Path) -> dict:
    """Convert a NIMD input file into typed workflow settings."""
    mode = _config_value(config, "run", "mode", "benchmark").strip().lower()
    if mode not in {"smoke", "benchmark"}:
        raise ValueError("run.mode must be either 'smoke' or 'benchmark'")

    task = _config_value(config, "run", "task", "regression").strip().lower()
    if task not in {"regression", "classification"}:
        raise ValueError("run.task must be either 'regression' or 'classification'")

    model_values = _split_values(_config_value(config, "run", "models", None))
    if not model_values:
        model_values = [_config_value(config, "run", "model", "pca").strip()]
    valid_models = {
        "pca",
        "beta",
        "fronorm",
        "hier",
        "multilayer",
        "deep",
        "ssnmf",
        "minvol",
    }
    invalid = sorted(set(model_values) - valid_models)
    if invalid:
        raise ValueError(f"unsupported model name(s): {', '.join(invalid)}")
    if mode == "smoke":
        model_values = model_values[:1]

    ranks = _parse_int_list(
        _config_value(config, "run", "ranks", None),
        [int(_config_value(config, "run", "rank", "4"))],
    )
    if any(rank < 1 for rank in ranks):
        raise ValueError("all run.ranks values must be positive integers")

    data_value = _config_value(config, "run", "data", "packaged").strip()
    packaged_data_names = {"packaged", "package", "default", "bundled"}
    data_path = (
        None
        if data_value.lower() in packaged_data_names
        else _resolve_path(data_value, input_path)
    )
    output_path = _resolve_path(_config_value(config, "run", "output", None), input_path)
    val_splits = _parse_float_list(
        _config_value(config, "model", "val_train_test_splits", None),
        [0.0, 0.8, 0.2],
    )
    if len(val_splits) != 3:
        raise ValueError("model.val_train_test_splits must contain exactly three values")

    norm_x_raw = _config_value(config, "model", "norm_X", None)
    norm_init_raw = _config_value(config, "model", "norm_init", None)
    maxiter_raw = _config_value(config, "run", "maxiter", "2")

    return {
        "mode": mode,
        "data": data_path,
        "task": task,
        "models": model_values,
        "ranks": ranks,
        "outerit": int(_config_value(config, "run", "outerit", "2")),
        "maxiter": int(_config_value(config, "run", "maxiter", "2")),
        "output": output_path,
        "init": _config_value(config, "model", "init", "auto").strip().lower(),
        "eval_type": _config_value(config, "model", "eval_type", "auto").strip().lower(),
        "fronorm_algo": _config_value(config, "model", "fronorm_algo", "HALS").strip(),
        "modelname": _config_value(config, "model", "modelname", "rf").strip().lower(),
        "dtype": _config_value(config, "model", "dtype", "float64").strip().lower(),
        "device": _config_value(config, "model", "device", "cpu").strip().lower(),
        "rng": int(_config_value(config, "model", "rng", "42")),
        "get_shap": _parse_bool(_config_value(config, "model", "get_shap", None), False),
        "display": _parse_bool(_config_value(config, "model", "display", None), False),
        "cv": _parse_bool(_config_value(config, "model", "cv", None), False),
        "return_tensor": _parse_bool(_config_value(config, "model", "return_tensor", None), False),
        "val_train_test_splits": val_splits,
        "eps_stab": _config_value(config, "model", "eps_stab", "auto").strip().lower(),
        "convert_type": _config_value(config, "model", "convert_type", "division_base").strip().lower(),
        "end": int(_config_value(config, "model", "end", "10")),
        "norm_X": None if _is_none(norm_x_raw) else norm_x_raw.strip(),
        "norm_init": None if _is_none(norm_init_raw) else norm_init_raw.strip(),
        "perturb": _parse_bool(_config_value(config, "model", "perturb", None), False),
        "noise_level": _parse_optional_float(_config_value(config, "model", "noise_level", None)),
        "layers_depths": int(_config_value(config, "deep", "layers_depths", "2")),
        "division_base": int(_config_value(config, "deep", "division_base", "2")),
        "lam": _parse_optional_float_list(_config_value(config, "deep", "lam", None)),
        "rho": int(_config_value(config, "deep", "rho", "10")),
        "beta": float(_config_value(config, "deep", "beta", "1")),
        "min_vol": _parse_bool(_config_value(config, "deep", "min_vol", None), False),
        "normalize": int(_config_value(config, "deep", "normalize", "2")),
        "innerloop": int(_config_value(config, "deep", "innerloop", "1")),
        "maxIterADMM": int(_config_value(config, "deep", "maxIterADMM", maxiter_raw)),
        "accADMM": _parse_bool(_config_value(config, "deep", "accADMM", None), True),
        "epsi": float(_config_value(config, "deep", "epsi", "1e-4")),
        "HnormType": _config_value(config, "deep", "HnormType", "rows").strip(),
        "ssnmf_model_num": int(_config_value(config, "ssnmf", "model_num", "4")),
        "ssnmf_split": float(_config_value(config, "ssnmf", "split", "0.2")),
        "ssnmf_tol": float(_config_value(config, "ssnmf", "tol", "1e-4")),
        "ssnmf_numiters": int(_config_value(config, "ssnmf", "numiters", maxiter_raw)),
        "ssnmf_iter_s": int(_config_value(config, "ssnmf", "iter_s", maxiter_raw)),
        "ssnmf_lam": float(_config_value(config, "ssnmf", "lam", "1e-3")),
        "ssnmf_seed": int(_config_value(config, "ssnmf", "seed", "42")),
    }


def _torch_dtype(name: str):
    """Map the input dtype tag to a PyTorch dtype."""
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("model.dtype must be either 'float64' or 'float32'")


def _base_params(
    model: str,
    task: str,
    rank: int,
    outerit: int,
    maxiter: int,
    settings: dict | None = None,
):
    settings = settings or {}
    deep_params = DeepNMFParams(
        layers_rank=[rank, max(1, rank // 2)],
        layers_depths=settings.get("layers_depths", 2),
        division_base=settings.get("division_base", 2),
        lam=settings.get("lam", None),
        outerit=outerit,
        maxiter=maxiter,
        rngseed=settings.get("rng", 42),
        rho=settings.get("rho", 10),
        display=settings.get("display", False),
        accADMM=True,
        beta=settings.get("beta", 1),
        min_vol=settings.get("min_vol", False),
        normalize=settings.get("normalize", 2),
        innerloop=settings.get("innerloop", 1),
        maxIterADMM=settings.get("maxIterADMM", maxiter),
        HnormType=settings.get("HnormType", "rows"),
        epsi=settings.get("epsi", 1e-4),
    )
    deep_params.accADMM = settings.get("accADMM", True)
    runner_model = "deep" if model == "minvol" else model
    return RunnerParams(
        rank=rank,
        model=runner_model,
        task=task,
        deep_params=deep_params,
        multi_params=deep_params,
        ssnmf_params=SSNMFParam(
            model_num=settings.get("ssnmf_model_num", 4),
            split=settings.get("ssnmf_split", 0.2),
            tol=settings.get("ssnmf_tol", 1e-4),
            numiters=settings.get("ssnmf_numiters", maxiter),
            iter_s=settings.get("ssnmf_iter_s", maxiter),
            lam=settings.get("ssnmf_lam", 1e-3),
            seed=settings.get("ssnmf_seed", 42),
        ),
        dtype=_torch_dtype(settings.get("dtype", "float64")),
        device=torch.device(settings.get("device", "cpu")),
        val_train_test_splits=settings.get("val_train_test_splits", [0, 0.8, 0.2]),
        display=settings.get("display", False),
        rng=settings.get("rng", 42),
        eval_type="feature" if model == "pca" else "full",
        eps_stab=1e-7,
        get_shap=settings.get("get_shap", False),
        convert_type=settings.get("convert_type", "division_base"),
        fronorm_algo=settings.get("fronorm_algo", "HALS"),
        modelname=settings.get("modelname", "rf"),
        norm_X=settings.get("norm_X", None),
        norm_init=settings.get("norm_init", None),
        cv=settings.get("cv", False),
        return_tensor=settings.get("return_tensor", False),
        perturb=settings.get("perturb", False),
        noise_level=settings.get("noise_level", None),
        end=settings.get("end", 10),
    )


def _prepare_model_params(
    params: RunnerParams,
    model_name: str,
    settings: dict | None = None,
) -> None:
    """Apply model-specific defaults and input-file overrides."""
    settings = settings or {}
    if model_name == "minvol":
        params.model = "deep"
        params.deep_params.min_vol = True
        params.deep_params.normalize = 3
    if _is_auto(settings.get("init", "auto")):
        params.init = "random" if model_name in {"pca", "hier"} else "nndsvd"
    else:
        params.init = settings["init"]
    if not _is_auto(settings.get("eval_type", "auto")):
        params.eval_type = settings["eval_type"]
    if _is_auto(settings.get("eps_stab", "auto")):
        params.eps_stab = 1e-6 if params.deep_params.min_vol else 1e-7
    else:
        params.eps_stab = float(settings["eps_stab"])


def _run_models(
    data_path: Path,
    task: str,
    ranks: list[int],
    models: list[str],
    outerit: int,
    maxiter: int,
    settings: dict | None = None,
):
    X, y, _, feature_names = quick_load_data(str(data_path), type=task)
    runner = Runner(X, y)
    records = []
    for model_name in models:
        params = _base_params(model_name, task, ranks[0], outerit, maxiter, settings)
        _prepare_model_params(params, model_name, settings)
        start = time.time()
        results = runner.run_evaluation(params, ranks=ranks)
        runtime_wall = float(time.time() - start)
        for result in results:
            records.append(
                {
                    "model": model_name,
                    "rank": int(result.rank),
                    "layers": (
                        None
                        if result.ranks is None
                        else [int(value) for value in result.ranks]
                    ),
                    "task": task,
                    "runtime_reported": float(result.runtime),
                    "runtime_wall": runtime_wall,
                    "r2": float(result.score.r2),
                    "rmse": float(result.score.rmse),
                    "mae": float(result.score.mae),
                    "accuracy": float(result.score.accuracy),
                    "f1_macro": float(result.score.f1_macro),
                }
            )
    return {
        "package": "nimd",
        "version": __version__,
        "data": str(data_path),
        "ranks": [int(rank) for rank in ranks],
        "shape": {"samples": int(X.shape[0]), "features": int(X.shape[1])},
        "target_size": int(y.shape[0]),
        "feature_count": len(feature_names),
        "records": records,
    }


def _write_or_print(payload: dict, output: str | None):
    text = json.dumps(payload, indent=2)
    if output:
        outpath = Path(output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(text)
        print(outpath)
    else:
        print(text)


def _run_from_input(input_file: str) -> int:
    """Run a smoke or benchmark workflow from a NIMD .in file."""
    input_path = Path(input_file).expanduser().resolve()
    config = _read_input_file(input_path)
    settings = _input_settings(config, input_path)
    if settings["data"] is None:
        with as_file(_packaged_data()) as data_path:
            payload = _run_models(
                Path(data_path),
                settings["task"],
                settings["ranks"],
                settings["models"],
                settings["outerit"],
                settings["maxiter"],
                settings,
            )
    else:
        payload = _run_models(
            settings["data"],
            settings["task"],
            settings["ranks"],
            settings["models"],
            settings["outerit"],
            settings["maxiter"],
            settings,
        )
    payload["input"] = str(input_path)
    payload["mode"] = settings["mode"]
    output_path = None if settings["output"] is None else str(settings["output"])
    _write_or_print(payload, output_path)
    return 0


def _write_input_template(output: str, force: bool = False) -> int:
    """Write the packaged nimd.in template to a user-selected path."""
    output_path = Path(output).expanduser()
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists; use --force to overwrite it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with as_file(_packaged_input()) as template_path:
        output_path.write_text(Path(template_path).read_text())
    print(output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nimd",
        description="NIMD - Nonnegative Interpretable Matrix Decomposition.",
    )
    parser.add_argument("--version", action="version", version=f"nimd {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("data-path", help="Print the packaged benchmark data path.")

    init_input = subparsers.add_parser(
        "init-input",
        help="Write a fresh nimd.in template.",
    )
    init_input.add_argument(
        "--output",
        default="nimd.in",
        help="Path for the generated input file.",
    )
    init_input.add_argument("--force", action="store_true", help="Overwrite an existing file.")

    run_input = subparsers.add_parser(
        "run",
        help="Run a workflow from a NIMD .in input file.",
    )
    run_input.add_argument(
        "-i",
        "--input",
        default="nimd.in",
        help="Input file path. Defaults to nimd.in.",
    )

    smoke = subparsers.add_parser("smoke", help="Run a one-model installation check.")
    smoke.add_argument("--data", default=None, help="CSV file to load. Defaults to the packaged benchmark.")
    smoke.add_argument("--task", choices=["regression", "classification"], default="regression")
    smoke.add_argument(
        "--model",
        default="pca",
        choices=["pca", "beta", "fronorm", "hier", "multilayer", "deep", "minvol"],
    )
    smoke.add_argument("--rank", type=int, default=4)
    smoke.add_argument("--output", default=None)

    bench = subparsers.add_parser("benchmark", help="Run a compact rank-limited benchmark.")
    bench.add_argument("--data", default=None, help="CSV file to load. Defaults to the packaged benchmark.")
    bench.add_argument("--task", choices=["regression", "classification"], default="regression")
    bench.add_argument("--models", nargs="+", default=["pca", "beta", "fronorm", "hier", "multilayer", "deep"])
    bench.add_argument("--rank", type=int, default=4)
    bench.add_argument("--outerit", type=int, default=2)
    bench.add_argument("--maxiter", type=int, default=2)
    bench.add_argument("--output", default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "data-path":
            with as_file(_packaged_data()) as data_path:
                print(data_path)
            return 0
        if args.command == "init-input":
            return _write_input_template(args.output, args.force)
        if args.command == "run":
            return _run_from_input(args.input)

        models = [args.model] if args.command == "smoke" else _model_list(args.models)
        outerit = args.outerit if args.command == "benchmark" else 2
        maxiter = args.maxiter if args.command == "benchmark" else 2
        if args.data:
            payload = _run_models(
                Path(args.data),
                args.task,
                [args.rank],
                models,
                outerit,
                maxiter,
            )
        else:
            with as_file(_packaged_data()) as data_path:
                payload = _run_models(
                    Path(data_path),
                    args.task,
                    [args.rank],
                    models,
                    outerit,
                    maxiter,
                )
        _write_or_print(payload, args.output)
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"nimd: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
