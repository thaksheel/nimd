import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import warnings
from typing import Literal, Optional, List, Dict, Any
import copy 
import time
import itertools
import random 

from .. import Runner, RunnerParams
from ..deep import DeepNMFParams

warnings.filterwarnings("ignore")


# Checkpoint: I have something that samples a space but needs to be tested on deep nmf to see if it works on a range of ranks and finds the best parameters. I have it set up in __main__ just need to debug and see. Once done I can implement the other methods in class()

class Sensitivity:
    def __init__(
        self,
        runner: Runner,
        runner_params: RunnerParams,
        max_trials: int = 200, 
    ):
        self.runner = runner
        self.runner_params = runner_params
        self.max_trials = max_trials 

    def _sample_param_space(self, study_params: Dict[str, List]) -> List[Dict[str, List]]: 
        keys = list(study_params.keys())
        values = list(study_params.values())
        full_space = list(itertools.product(*values))
        if len(full_space) > self.max_trials:
            samples = random.sample(full_space, self.max_trials)
        else:
            samples = full_space 

        param_dicts = []
        for sample in samples: 
            d = {k: v for k, v in zip(keys, sample)}
            param_dicts.append(d)
        return param_dicts

    def study_deep(
        self,
        study_params: Dict[str, List],
        ranks: List[int], 
    ):
        trials = self._sample_param_space(study_params=study_params)
        best_score = -float("inf")
        best_params = None 
        history = []
        for trial in trials:
            p: RunnerParams = copy.deepcopy(self.runner_params)
            for k, v in trial.items(): 
                setattr(p, k, v)
            score = self.evaluates(ranks=ranks)
            history.append(
                {
                    "params": trial, 
                    "score": score, 
                }
            )
            # TODO: need to account for scores such as R2 and RMSE with different directions 
            if score > best_score: 
                best_score = score 
                best_params = trial 
        return best_params, best_score, history


    def study_deep_minvol(
        self,
    ):
        pass

    def study_standard(
        self,
    ):
        pass

    def evaluates(self, ranks: List):
        data = self.runner.run_evaluation(
            params=self.params,
            ranks=ranks,
        )
        return sum([s for s in data.scores])


if __name__ == "__main__":
    from .. import Runner, RunnerParams
    import torch 

    runner = Runner(X=None, y=None)
    runner_params = RunnerParams (
        dtype=torch.float64, 
        device=torch.device("cuda"), 
        rank=10, 
        model="deep", 
        task="regression", 
        score_type="r2", 
        val_train_test_splits=[0, 0.8, 0.2], 
    )
    sensitivty = Sensitivity(
        runner=runner, 
        runner_params=runner_params, 
        max_trials=10, 
    )
    ranks = [i for i in range(20, 70, 10)]
    study_params = {
        "maxiter": [20, 40, 80, 100, 140, 160, 200, 240], 
        "outeriter": [20, 40, 80, 100, 140, 160, 200, 240], 
        "beta": [1, 2], 
        "epsi": [1e-10, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1], 
        "lam": [None, (4,2,1), (1,2,4), (10, 2, 1), (1, 2, 10)],  
    }
    best_params, best_score, history = sensitivty.study_deep(ranks=ranks, study_params=study_params)

    print(best_score)
    print(best_params)

    print("END")
