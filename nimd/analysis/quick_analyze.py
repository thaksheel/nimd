import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nimd.analyze_results import AnalyzeResults


ar = AnalyzeResults()

# NOTE: SSNMF scores agg processing
if False:
    filename = "./exports/results_ssnmf2.xlsx"
    df = pd.read_excel(filename)

    nums = df.model_num.unique().tolist()
    inits = df.init.unique().tolist()
    lams = df.lam.unique().tolist()
    ranks = df["rank"].unique().tolist()

    df_ssnmf = (
        df.query("20 <= rank <= 70")
        .groupby(["lam", "model_num", "init"])
        .mean(numeric_only=True)
        .reset_index()
    )
    best_by_init = df.loc[df.groupby("init")["score"].idxmax()].reset_index(drop=True)
    best_by_init.drop(columns=["Unnamed: 0"], inplace=True)
    best_by_init.to_excel("./exports/ssnmf_scores.xlsx")


if True:  # for new nmf exports by methods results
    filename = "./exports/results_super1.xlsx"
    df = pd.read_excel(filename)
    df_scores_means, df_scores_std = ar.export_results(
        df, outpath="./exports/super_scores.xlsx"
    )
    metrics = ["r2", "rmse", "mae", "accuracy", "f1_macro", "runtime"]
    df_melt = df_scores_means.melt(
        id_vars=["model", "init"], value_vars=metrics, var_name="metric", value_name="value"
    )
    df_pivot = df_melt.pivot_table(
        index=["model", "metric"], columns="init", values="value"
    ).reset_index()
    df_pivot = df_pivot.sort_values(["model", "metric"])

print("END")
