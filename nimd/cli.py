"""Command line interface for NIMD."""

from __future__ import annotations

import argparse
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


def _model_list(raw: Iterable[str]) -> list[str]:
    models: list[str] = []
    for item in raw:
        models.extend(part.strip() for part in item.split(",") if part.strip())
    return models


def _base_params(model: str, task: str, rank: int, outerit: int, maxiter: int):
    deep_params = DeepNMFParams(
        layers_rank=[rank, max(1, rank // 2)],
        layers_depths=2,
        division_base=2,
        lam=None,
        outerit=outerit,
        maxiter=maxiter,
        rngseed=42,
        rho=10,
        display=False,
        accADMM=True,
        beta=1,
        min_vol=False,
        normalize=2,
        innerloop=1,
        maxIterADMM=maxiter,
    )
    return RunnerParams(
        rank=rank,
        model=model,
        task=task,
        deep_params=deep_params,
        multi_params=deep_params,
        ssnmf_params=SSNMFParam(numiters=maxiter, iter_s=maxiter),
        dtype=torch.float64,
        device=torch.device("cpu"),
        val_train_test_splits=[0, 0.8, 0.2],
        display=False,
        rng=42,
        eval_type="full",
        eps_stab=1e-7,
        get_shap=False,
    )


def _run_models(data_path: Path, task: str, rank: int, models: list[str], outerit: int, maxiter: int):
    X, y, _, feature_names = quick_load_data(str(data_path), type=task)
    runner = Runner(X, y)
    records = []
    for model_name in models:
        params = _base_params(model_name, task, rank, outerit, maxiter)
        if model_name == "minvol":
            params.model = "deep"
            params.deep_params.min_vol = True
            params.deep_params.normalize = 3
            params.eps_stab = 1e-6
        elif model_name == "pca":
            params.eval_type = "feature"
            params.init = "random"
        elif model_name == "hier":
            params.init = "random"
        else:
            params.init = "nndsvd"
        params.fronorm_algo = "HALS"
        start = time.time()
        result = runner.run_evaluation(params, ranks=[rank])[0]
        records.append(
            {
                "model": model_name,
                "rank": int(result.rank),
                "task": task,
                "runtime_reported": float(result.runtime),
                "runtime_wall": float(time.time() - start),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nimd",
        description="NIMD - Nonnegative Interpretable Matrix Decomposition.",
    )
    parser.add_argument("--version", action="version", version=f"nimd {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("data-path", help="Print the packaged benchmark data path.")

    smoke = subparsers.add_parser("smoke", help="Run a one-model installation check.")
    smoke.add_argument("--data", default=None, help="CSV file to load. Defaults to the packaged benchmark.")
    smoke.add_argument("--task", choices=["regression", "classification"], default="regression")
    smoke.add_argument("--model", default="pca", choices=["pca", "beta", "fronorm", "hier", "multilayer", "deep", "minvol"])
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
    if args.command == "data-path":
        with as_file(_packaged_data()) as data_path:
            print(data_path)
        return 0

    models = [args.model] if args.command == "smoke" else _model_list(args.models)
    outerit = args.outerit if args.command == "benchmark" else 2
    maxiter = args.maxiter if args.command == "benchmark" else 2
    if args.data:
        payload = _run_models(Path(args.data), args.task, args.rank, models, outerit, maxiter)
    else:
        with as_file(_packaged_data()) as data_path:
            payload = _run_models(Path(data_path), args.task, args.rank, models, outerit, maxiter)
    _write_or_print(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
