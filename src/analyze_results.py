import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List
from scipy.stats import ttest_ind
from typing import List, Literal, Optional, Dict, Tuple


class AnalyzeResults:
    def __init__(self):
        pass

    def scores_table(
        self,
        df: pd.DataFrame,
        inits: List[str], 
        metrics: List[str],
        tasks: List[str],
        models: List[str],
        type: Literal["mean", "std"],
    ):
        dfs = []
        for init in inits:
            for task in tasks:
                means = []
                for model in models:
                    if type == "mean":
                        val = [
                            df[
                                (df.nmf_model == model)
                                & (df.task == task)
                                & (df.init == init)
                            ][metric].mean()
                            for metric in metrics
                        ]
                    elif type == "std":
                        val = [
                            df[
                                (df.nmf_model == model)
                                & (df.task == task)
                                & (df.init == init)
                            ][metric].std()
                            for metric in metrics
                        ]
                    means.append(val)
                df_s = pd.DataFrame(columns=["model"], data=models)
                df_s[metrics] = means
                df_s["task"] = [task] * len(df_s)
                df_s["init"] = [init] * len(df_s)
                dfs.append(df_s)
            df_scores = pd.concat(dfs)
        return df_scores

    def export_results(self, df: pd.DataFrame, outpath: str):
        tasks = df.task.unique().tolist()
        nmf_models = df.nmf_model.unique().tolist()
        metrics = ["r2", "rmse", "mae", "accuracy", "f1_macro", "runtime"]
        inits = df.init.unique().tolist()
        df_scores_means = self.scores_table(df, inits, metrics, tasks, nmf_models, type="mean")
        df_scores_std = self.scores_table(df, inits, metrics, tasks, nmf_models, type="std")
        with pd.ExcelWriter(outpath) as writer:
            df_scores_means.to_excel(writer, sheet_name="means", index=False)
            df_scores_std.to_excel(writer, sheet_name="stddev", index=False)
        return df_scores_means, df_scores_std

    def ttest(
        self, df: pd.DataFrame, pval_threshold: float = 0.05, metric: str = "accuracy"
    ):
        models = df.method.unique()
        results = np.array(
            [df[df.method == method][metric].to_numpy() for method in models]
        )
        results_counts = self.pairwise_ttest(
            models, results, pval_threshold=pval_threshold
        )
        df_pvals = pd.DataFrame(columns=models, index=models, data=results_counts)
        return df_pvals

    def pairwise_ttest(
        self, models: List, results: np.ndarray, pval_threshold: float = 0.05
    ):
        pvals = []
        for i in range(len(models)):
            pvals.append(
                np.array(
                    [
                        ttest_ind(results[i], result, equal_var=False)[1]
                        for result in results
                    ]
                )
            )
        pvals = np.array(pvals)
        eval_pvals = pvals <= pval_threshold
        return eval_pvals
