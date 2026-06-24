import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Literal
import re


def _table(M: str, full: pd.DataFrame, ot: Literal["mean", "std"]) -> pd.DataFrame:
    if ot == "mean":
        summary = full.groupby([M, "init"])["score"].mean().reset_index()
    if ot == "std":
        summary = (
            full.groupby([M, "init"])["score"].std().reset_index()
        )
    table = summary.pivot(index=M, columns="init", values="score")
    return table


def scores_summary_table(files: List[str]) -> List[pd.DataFrame]:
    models = [
        re.search(r"results_([a-zA-Z0-9_]+)\d\.xlsx$", s).group(1)
        for s in files
    ]
    datasets = dict(zip(models, files))
    dfs = []
    for model_name, path in datasets.items():
        df = pd.read_excel(path)
        df["model"] = model_name
        dfs.append(df)
    full: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    tables = [
        _table(m, full, ot)
        for m in ["model", "fronorm_algo"]
        for ot in ["mean", "std"]
    ]
    return tables


if __name__ == "__main__":
    files = [
        "./exports/bin-results/results_fronorm0.xlsx",
        "./exports/bin-results/results_pca0.xlsx",
        "./exports/bin-results/results_beta0.xlsx",
        "./exports/bin-results/results_hier0.xlsx",
        "./exports/bin-results/results_multilayer0.xlsx",
        "./exports/bin-results/results_deep0.xlsx",
        # "./exports/bin-results/results_deep_minvol0.xlsx",
    ]
    tables = scores_summary_table(files)
    [
        table.to_excel(f"./exports/table_{k}.xlsx", index=True)
        for k, table in enumerate(tables)
    ]
    print("END")
