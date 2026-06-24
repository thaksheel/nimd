import numpy as np
from typing import Literal, Optional, List
import pandas as pd
from dataclasses import dataclass
import time

from . import *

# from .interpretation import Interpretation


class Runner(object):

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        self.X = X
        self.y = y
        self.params: RunnerParams = None

    def run_evaluation(
        self,
        params: RunnerParams,
        ranks: List,
    ) -> List[ResultData]:
        self.params = params
        self.nmf_selection = NMFSelection(
            X=self.X,
            init=params.init,
            rank=params.rank,
            norm_init=params.norm_init,
            norm_X=params.norm_X,
            rng=params.rng,
            eval_type=params.eval_type,
            perturb=params.perturb,
            nl=params.noise_level,
        )
        self.supervised_learning = SupervisedLearning(
            task=self.params.task,
            val_train_test_splits=self.params.val_train_test_splits,
            modelname=self.params.modelname,
            cv=self.params.cv,
            rng=self.params.rng,
            get_shap=params.get_shap,
        )
        base_score = self.supervised_learning.evaluate(
            X=self.X, y=self.y, W=None, evaluation_type="full"
        )
        ranks_layers = self.convert_ranks_layers(
            ranks=ranks,
            division_base=self.params.deep_params.division_base,
            layers_depths=self.params.deep_params.layers_depths,
            convert_type=self.params.convert_type,
            end=self.params.end,
        )
        results = []
        for k, rank in enumerate(ranks):
            start_time = time.time()
            Wl, Hl = [], []
            if self.params.model in ["deep", "multilayer"]:
                self.params.deep_params.layers_rank = ranks_layers[k]
                self.params.multi_params.layers_rank = ranks_layers[k]
                W, H, Wl, Hl, output = self.nmf_selection.factorize_deep(
                    rank=rank,
                    model=self.params.model,
                    deep_params=self.params.deep_params,
                    mul_params=self.params.multi_params,
                    device=self.params.device,
                    dtype=self.params.dtype,
                    eps_stab=self.params.eps_stab,
                    return_tensor=self.params.return_tensor,
                )
            elif self.params.model != "ssnmf":
                self.nmf_selection.update_rank(rank)
                W, H, output = self.nmf_selection.factorize_standard(
                    rank=rank,
                    model=self.params.model,
                    fronorm_algo=self.params.fronorm_algo,
                )
                duration = time.time() - start_time
            if self.params.model == "ssnmf":
                X_train, X_test, y_train, y_test = self.nmf_selection.evaluate_ssnmf(
                    self.y,
                    rank,
                    self.params.ssnmf_params,
                    self.params.task,
                )
                score = self.supervised_learning.evaluate(
                    X=None,
                    W=None,
                    y=self.y,
                    evaluation_type=self.params.eval_type,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                )
                output = None
                duration = time.time() - start_time
            else:
                score = self.supervised_learning.evaluate(
                    X=W @ H,
                    W=W,
                    y=self.y,
                    evaluation_type=self.params.eval_type,
                )
                duration = time.time() - start_time
            if self.params.display:
                print(
                    f"method={self.params.model} "
                    f"rank={rank} "
                    f"rank_layer={ranks_layers[k]} "
                    f"r2={score.r2:.4f} "
                    f"rmse={score.rmse:.4f} "
                    f"acc={score.accuracy:.4f} "
                    f"duration={duration:.2f}s "
                    f"init={self.params.init} "
                    f"depths={self.params.deep_params.layers_depths}"
                )
            results.append(
                ResultData(
                    base_score=base_score,
                    nmf_model=self.params.model,
                    ml_model=self.params.modelname,
                    init=self.params.init,
                    min_vol=self.params.deep_params.min_vol,
                    feature_importance=self.supervised_learning.get_shap_values(),
                    W=None if self.params.model == "ssnmf" else W,
                    H=None if self.params.model == "ssnmf" else H,
                    Wl=Wl,
                    Hl=Hl,
                    runtime=duration,
                    score=score,
                    depths=self.params.deep_params.layers_depths,
                    rank=rank,
                    eps=self.params.eps_stab,
                    fronorm_algo=self.params.fronorm_algo,
                    error=output["e"] if self.params.model == "deep" else output,
                    ranks=(
                        ranks_layers[k]
                        if self.params.model in ["deep", "multilayer"]
                        else None
                    ),
                )
            )
        return results

    def convert_ranks_layers(
        self,
        ranks: List[int],
        division_base: int = 2,
        layers_depths: int = 3,
        convert_type: Literal["division_base", "linspace"] = "division_base",
        end: int = 10,
    ) -> List[List[int]]:
        if convert_type == "division_base":
            return [
                [
                    (
                        int(rank // pow(division_base, i))
                        if rank // pow(division_base, i) > 0
                        else 1
                    )
                    for i in range(layers_depths)
                ]
                for rank in ranks
            ]
        elif convert_type == "linspace":
            A = [np.linspace(start, end, layers_depths, dtype=int) for start in ranks]
            return A
        else:
            raise ValueError("pass in ranks or customer_rank_layers in Runner()")

    def exp_minvol_identifiability(
        self,
        params: RunnerParams,
        seeds: List[int],
        noise_levels: List[float],
        ranks: List[int],
        method: Literal["minvol", "unconstraint", "other"],
    ) -> np.ndarray:
        """Return a tensor with shape=(seed, noise_lvl, ResultData)"""
        if method == "minvol":
            params.model = "deep"
            params.deep_params.min_vol = True
            params.deep_params.normalize = 3
            params.eps_stab = 1e-6
            params.deep_params.outerit = 50
            params.deep_params.maxiter = 100
            params.deep_params.maxIterADMM = 60
        elif method == "unconstraint":
            params.model = "deep"
            params.deep_params.min_vol = False
            params.deep_params.normalize = 2
            params.eps_stab = 1e-7
            params.deep_params.outerit = 150
            params.deep_params.maxiter = 150
        R = []
        params.perturb = True
        for seed in seeds:
            r = []
            if params.display:
                print(f"\n-> seed={seed}")
            params.deep_params.rngseed = seed
            params.rng = seed
            for nl in noise_levels:
                if params.display:
                    print(f"--> noise_level={nl}")
                params.noise_level = nl
                evals = self.run_evaluation(params, ranks=ranks)
                results = []
                for e in evals:
                    e.noise = nl
                    e.seed = seed
                    results.append(e)
                r.append(np.array(results))
            R.append(np.array(r))
        return np.array(R)

    def exp_element_level_attribution(
        self,
        params: RunnerParams,
        ranks: List[int],
        nmf_models: List[str],
        interpret,
        rank_topk: int = 3,
        factor_topk: int = 10,
    ) -> pd.DataFrame:
        df_importance_tops = []
        # interpret = Interpretation() # TODO: instantiate interpret properly here
        for model in nmf_models:
            if model == "deep_minvol":
                params.model = "deep"
                params.deep_params.normalize = 3
                params.eps_stab = 1e-6
                params.deep_params.min_vol = True
            else:
                params.eps_stab = 1e-7
                params.deep_params.normalize = 2
                params.deep_params.min_vol = False
                params.model = model
            results = self.run_evaluation(params, ranks)
            df_importance_ = interpret.map_factors(
                results=results,
                lookup_file="./data/lookup.xlsx",
                top_k=factor_topk,
            )
            df_importance_top_ = (
                df_importance_.sort_values(
                    ["rank", "importance"], ascending=[True, False]
                )
                .groupby("rank")
                .head(rank_topk)
                .reset_index(drop=True)
            )
            df_importance_top_["nmf_model"] = [model] * len(df_importance_top_)
            df_importance_tops.append(df_importance_top_)
            if self.params.display:
                print(f"\n --> completed nmf_model={model}")
        df_importance_top = pd.concat(df_importance_tops)
        return df_importance_top

    def exp_supervised_results(
        self,
        ranks: List[int],
        models: List[str],
        inits: List[str],
        algos: List[str],
        params: RunnerParams,
    ):
        results = []
        for model in models:
            if model == "minvol":
                params.model = "deep"
                params.deep_params.normalize = 3
                params.eps_stab = 1e-6
                params.deep_params.min_vol = True
            else:
                params.model = model
                params.deep_params.normalize = 2
                params.eps_stab = 1e-7
                params.deep_params.min_vol = False
            params.eval_type = "feature" if model == "pca" else "full"
            for init in inits:
                params.init = "random" if model in ["pca", "hier"] else init
                for algo in algos:
                    params.fronorm_algo = algo if model == "fronorm" else None
                    rds = self.run_evaluation(params=params, ranks=ranks)
                    results.append(rds)
                    if model != "fronorm":
                        break
                if model in ["pca", "hier"]:
                    break
        results = [r for re in results for r in re]
        return results

    def convert_results_to_df(self, results_data: List[ResultData]):
        fields = results_data[0].__dict__.keys()
        exclude = [
            f for f in fields if isinstance(results_data[0].__dict__[f], np.ndarray)
        ]
        data = [r.__dict__ for r in results_data]
        fields_score = list(results_data[0].score.__dict__.keys())
        exclude_score = [
            f
            for f in fields_score
            if isinstance(results_data[0].score.__dict__[f], np.ndarray)
        ]
        df = pd.DataFrame(data)
        d = np.array(
            [
                np.array([r.score.__dict__[f] for r in results_data])
                for f in fields_score
                if f not in exclude_score
            ]
        )
        fields_score = [f for f in fields_score if f not in exclude_score]
        df[fields_score] = d.T
        df.drop(columns=exclude, inplace=True)
        df.drop(
            columns=[
                "base_score",
                "score",
                "ranks",
                "feature_importance",
                "error",
                "Wl",
                "Hl",
            ],
            inplace=True,
        )
        return df
