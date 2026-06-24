from typing import Literal
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from typing import Literal, Optional, List


def expand_layers(Wl, Hl):
    Ws = []
    for W in Wl[::-1]:
        for H in Hl[::-1]:
            if H.shape[0] != W.shape[1]:
                continue
            W = W @ H
        Ws.append(W)
    return Ws

def exports_factors_deep(
    export_type: Literal["all", "Ws_expand", "X_recon", "factors"],
    outpath: str,
    nmf: str,
    task: str,
    ranks: List,
    Ws: Optional[np.ndarray] = None,
    Hs: Optional[np.ndarray] = None,
    Wls: Optional[np.ndarray] = None,
    Hls: Optional[np.ndarray] = None,
):
    if export_type == "Ws_expand" and Wls is not None and Hls is not None:
        for k, rank in enumerate(ranks):
            filename = outpath + f"deepnmf_Wl_rank{rank}.xlsx"
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                layers = expand_layers(Wls[k], Hls[k])
                for l, W in enumerate(layers):
                    # TODO: ensure that each l has shape=(864,74)
                    df = pd.DataFrame(W)
                    df.to_excel(writer, sheet_name=f"l{l+1}", index=False)
    else:
        return export_factors_standard(
            export_type=export_type,
            outpath=outpath,
            Hs=Hs,
            Ws=Ws,
            task=task,
            ranks=ranks,
            nmf=nmf,
        )


def export_factors_standard(
    export_type: Literal["all", "X_recon", "factors"],
    outpath: str,
    Ws: Optional[np.ndarray],
    Hs: Optional[np.ndarray],
    task: str,
    nmf: str,
    ranks: List,
):
    if export_type == "X_recon":
        filename = outpath + f"Xs_{nmf}_{task}.xlsx"
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            arrays = [i @ Hs[j] for j, i in enumerate(Ws)]
            for i, array in enumerate(arrays):
                df = pd.DataFrame(array)
                df.to_excel(writer, sheet_name=f"r{ranks[i]}", index=False)
    elif export_type == "factors":
        for type in ["Ws", "Hs"]:
            filename = outpath + f"{type}_{nmf}_{task}.xlsx"
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                if type == "Ws":
                    arrays = Ws
                elif type == "Hs":
                    arrays = Hs
                for i, array in enumerate(arrays):
                    df = pd.DataFrame(array)
                    df.to_excel(writer, sheet_name=f"r{ranks[i]}", index=False)
    elif export_type == "all":
        for type in ["Ws", "Hs", "Xs"]:
            filename = outpath + f"{type}_{nmf}_{task}.xlsx"
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                if type == "Ws":
                    arrays = Ws
                elif type == "Hs":
                    arrays = Hs
                elif type == "Xs":
                    arrays = [i @ Hs[j] for j, i in enumerate(Ws)]
                for i, array in enumerate(arrays):
                    df = pd.DataFrame(array)
                    df.to_excel(writer, sheet_name=f"r{ranks[i]}", index=False)
    return True


def affichage(
    ranks,
    scores,
    base_score,
    outfile: str,
    title: str,
    evaluation_measure: Literal["xy", "classic", "ratio"],
    save: bool = False,
    truncate: bool = False,
    cv: bool = False,
    y_axis: str = "R^2"
):
    if evaluation_measure != "xy":
        if truncate:
            accuracy_mean = [np.mean(e) if np.mean(e) > 0 else 0 for e in scores]
        else:
            accuracy_mean = [np.mean(e) for e in scores]
    if evaluation_measure == "classic":
        eval0 = base_score
    elif evaluation_measure == "ratio":
        eval0 = 1
    elif evaluation_measure == "xy":
        eval0 = [e for e in scores[1]]
    if cv == False:
        if evaluation_measure != "xy":
            plt.figure()
            plt.plot(ranks, accuracy_mean, "o-", markersize=3)
            plt.axhline(
                y=eval0,
                color="red",
                linestyle="--",
                label="baseline",
            )
            plt.title(title)
            plt.ylabel(y_axis)
            plt.xlabel(f"Rank({ranks[0]}-{ranks[-1]})")
            plt.legend()
            if save:
                plt.savefig(outfile)
            plt.show()
        else:
            plt.figure()
            sc = plt.scatter(
                scores[1],
                scores[0],
                c=[i for i in range(len(scores[0]))],
                cmap="viridis",
                s=5,
            )
            # plt.plot(evals[1], evals[0], "o", markersize=3)
            plt.plot(
                scores[1],
                eval0,
                "-",
                markersize=1,
                alpha=0.75,
                color="red",
                label="baseline",
            )
            plt.title(title)
            plt.ylabel(f"NMF {y_axis}")
            plt.xlabel(f"PCA {y_axis}")
            plt.legend()
            plt.colorbar(sc, label="ranks")
            if save:
                plt.savefig(outfile)
            plt.show()
    elif cv == True:
        accuracy_std = [
            stats.norm.interval(0.95, loc=np.mean(e), scale=stats.sem(e))
            for e in scores
        ]
        plt.figure()
        plt.plot(accuracy_mean, "o-", markersize=3)
        plt.axhline(
            y=eval0,
            color="red",
            linestyle="--",
            label="baseline",
        )
        errors = [(h - l) / 2 for l, h in accuracy_std]
        plt.errorbar(
            accuracy_mean,
            yerr=errors,
            fmt="o",
            color="steelblue",
            alpha=0.3,
            capsize=3,
            label=f"95% StdDev",
        )
        plt.title(title)
        plt.ylabel(y_axis)
        plt.xlabel(f"Rank({ranks[0]}-{ranks[-1]})")
        plt.legend()
        if save:
            plt.savefig(outfile)
        plt.show()
