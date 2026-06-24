import pandas as pd
import numpy as np
import optuna
from nimd.legacy.evaluations import SSNMFParam, NMFEvaluations, DeepNMFParams
from typing import Literal

from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")
# TODO: create an optuna framework and test on lam values fine tune based on sum of R^2 for all ranks z


def quick_load_data(filenmae: str, type: Literal["regression", "classification"]):
    df = pd.read_csv(filenmae)
    if type == "classification":
        df["stab"] = df["Stability"].map({"unstable": 0, "stable": 1}).astype(int)
        X = df.select_dtypes(include=[np.number]).drop(columns=["stab"]).copy()
        y = df["stab"].copy()
    elif type == "regression":
        X = df.drop(["Material", "a", "b", "Composition", "Stability"], axis=1).copy()
        y = df["a"].copy()
    return df, X, y


def make_objective(nmf_obj: NMFEvaluations, deep_params: DeepNMFParams, ranks):
    def objective(trial: optuna.Trial):
        # TODO: implement a suggest_int for lam
        lam1 = trial.suggest_int("lam1", 1, 20)
        lam2 = trial.suggest_int("lam2", 1, 20)
        lam3 = trial.suggest_int("lam3", 1, 20)
        nmf_obj.deepparams.lam = (lam1, lam2, lam3)
        # nmf_obj.deepparams.lam = trial.suggest_categorical( "lam", [(4, 2, 1), (12, 3, 1), None, (1, 3, 12), (1, 1, 1), (2, 1, 1), (3, 2, 1), (6, 2, 1), (9, 3, 1)] )
        nmf_obj.deepparams.outerit = trial.suggest_int("outerit", 25, 500)
        nmf_obj.deepparams.maxiter = trial.suggest_int("maxiter", 25, 500)
        init= trial.suggest_categorical(
            "init", ['random', 'nndsvd', 'nndsvda', 'nndsvdar', 'nnsvdlrc']
        )
        scores = nmf_obj.evaluates(
            nmf="deep",
            init=init,
            evaluation_type="feature",
            normalize_X="minmax",
            normalize_init=None,
            algo="MUUP",
            ranks=ranks,
        )
        return sum([s[0] for s in scores])

    return objective

if __name__ == "__main__":
    lam_space = [i for i in range(1, 7)]
    search_space = {
        # "lam": [(4, 2, 1), (12, 3, 1), None, (1, 1, 1)],
        "lam1": lam_space,
        "lam2": lam_space,
        "lam3": lam_space,
        "outerit": [300],
        "maxiter": [300],
        "init": ['nnsvdlrc']
    }
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(nmf_evaluations, deepparams, ranks))
    study_data = [dt.__dict__["_params"] for dt in study.trials]
    df_study = pd.DataFrame(study_data)
    df_study['scores'] = [dt.__dict__['_values'][0] for dt in study.trials]
    df_study.to_excel(f"./exports/sa_deep_{score_type}_3.xlsx", index=False)

    print("Best parameters:", study.best_params)
    print("Best AIC:", study.best_value)

    print("END")
    # Checkpoint: run the sa as if and test the long range of iterations for sa report


