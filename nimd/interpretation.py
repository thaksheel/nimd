import numpy as np
from typing import List, Literal
import pandas as pd
import torch
from dataclasses import dataclass
import seaborn as sns
from matplotlib import pylab as plt
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import entropy

from .core.utils import HierarchyAbundance
from .runner import Runner
from .core.utils import RunnerParams, ResultData


@dataclass
class DepthsData:
    nmf_model: str
    init: str
    convert_type: str
    rank: int
    depth: int
    score: int
    runtime: float
    nmi_last: float
    depth_score: float
    end_rank: int
    division_base: str


class Interpretation:
    def __init__(self):
        pass

    def _error_norm(self, errors: torch.tensor) -> np.ndarray:
        err = errors.cpu().numpy()
        col_min = err.min(axis=0)
        col_max = err.max(axis=0)
        norm = (err - col_min) / (col_max - col_min + 1e-12)
        return norm

    def nmi_between_layers(self, H1: np.ndarray, H2: np.ndarray):
        z1 = np.argmax(H1, axis=0)
        H2_effective = H2 @ H1
        z2 = np.argmax(H2_effective, axis=0)
        nmi = normalized_mutual_info_score(z1, z2)
        return nmi

    def _nmi_scores(self, Hl: List[np.ndarray]):
        H_base = Hl[0]
        nmi = []
        for k, H in enumerate(Hl[1:]):
            H_effective = H if k < 1 else H @ H_effective
            nmi.append(self.nmi_between_layers(H_base, H_effective))
        return np.array(nmi)

    def depths_ratio(
        self,
        Hl: list[np.ndarray],
        errors: torch.tensor,
        score: int,
        out_type: Literal["errors", "score"],
    ):
        nmi_base = self._nmi_scores(Hl)[-1]
        sum_norm_error = self._error_norm(errors).sum()
        if out_type == "errors":
            ratio = nmi_base / (sum_norm_error + 1e-12)
        elif out_type == "score":
            ratio = nmi_base / (score + 1e-12)
        return ratio

    def depths_ratios(
        self, results: List[ResultData], out_type: Literal["errors", "score"]
    ):
        return np.array(
            [
                self.depths_ratio(
                    Hl=result.Hl,
                    errors=result.error,
                    score=result.score,
                    out_type=out_type,
                )
                for result in results
            ]
        )

    def depths_score(
        self,
        Hl: list[np.ndarray],
        score: int,
        weight: List[float] = [0.7, 0.3],
        eps: float = 1e-9,
        penalty: float = 0.05,
    ):
        nmi_base = self._nmi_scores(Hl)[-1]
        score = max(score, eps)
        nmi_base = max(nmi_base, eps)
        base = np.prod(np.array([pow(score, weight[0]), pow(nmi_base, weight[1])]))
        if nmi_base <= 0 or score <= 0:
            return base - penalty
        return base

    def depths_scores(
        self,
        results: List[ResultData],
        weight: List[float] = [0.7, 0.3],
        eps: float = 1e-9,
        penalty: float = 0.05,
    ):
        return np.array(
            [
                self.depths_score(
                    result.Hl,
                    result.score.r2,
                    weight,
                    eps,
                    penalty,
                )
                for result in results
            ]
        )

    def plot_errors(self, errors: torch.tensor):
        norm = self._error_norm(errors)
        for i in range(norm.shape[1]):
            plt.plot(norm[:, i], marker="o", label=f"layer{i}")
        plt.title("norm deep nmf error")
        plt.xlabel("iters")
        plt.ylabel("norm errors")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return norm

    def _layers_mapping(
        self,
        data: np.ndarray,
        lookup: pd.DataFrame,
        layer: int,
        rank: int,
        type: Literal["threshold", "top_k"],
        threshold: float = 0.01,
        top_k: int = 3,
        element_cut_off: int = 64,
    ):
        # TODO: threshold has to be fine tuned to return the total number of elements each time
        composition: List[HierarchyAbundance] = []
        df = pd.DataFrame(data[:, :element_cut_off])
        for i, row in df.iterrows():
            if type == "threshold":
                mask = row > threshold
                abundance_series = row[mask == True]
            elif type == "top_k":
                abundance_series = row.nlargest(top_k)
            idx = abundance_series.index
            for j, abundance in enumerate(abundance_series):
                ha = HierarchyAbundance(
                    component_num=i,
                    abundance=abundance,
                    previous_component_num=idx[j] if layer != 0 else 0,
                    layer=layer,
                    rank=rank,
                    threshold=threshold,
                )
                # if layer == 0:
                ha.element_name = lookup[lookup["i"] == idx[j]]["elements"].iloc[0]
                ha.element_num = idx[j]
                ha.metal_class = lookup[lookup["i"] == idx[j]]["class"].iloc[0]
                composition.append(ha)
        return composition

    def _convert_to_proportions(self, data: np.ndarray):
        data_ = []
        for row in data:
            arr = [(el - row.min()) / (row.max() - row.min()) for el in row]
            data_.append([i / sum(arr) for i in arr])
        return np.array(data_)

    def topk_element_attribution(
        self,
        Hs: List[np.ndarray],
        threshold_by_layers: List[float],
        type_t: Literal["top_k", "threshold"],
        lookup: pd.DataFrame,
        top_k: int = 3,
        outfilename: str = None,
    ) -> pd.DataFrame:
        """Generate and export periodic‑table maps from a sequence of NMF factor matrices.

        Parameters
        ----------
        Hs : list of ndarray List of factor matrices from a deep or multilayer NMF model. Each entry corresponds to a layer and should have shape ``(n_elements, r_l)`` where ``r_l`` is the rank for that layer. Example for ranks [2, 3, 4] with 74 elements: ``Hs = [H1 (74×2), H2 (74×3), H3 (74×4)]``.
        ranks : list of int
            Rank values associated with each matrix in ``Hs``. Used to label Excel sheets in the exported file.
        threshold_by_layers : list of float
            Layer‑specific thresholds applied when constructing periodic maps. Must be the same length as ``Hs``.
        layers : list of int
            Layer identifiers used for annotation in the exported Excel file. Must satisfy ``len(layers) == len(Hs)``.
        lookup : pandas.DataFrame
            Lookup table mapping element indices to metadata. Must contain columns ``["i", "elements", "class"]`` where: - ``i`` is the element index matching rows of each ``H`` matrix, - ``elements`` is the element symbol, - ``class`` is an optional grouping or category.
        outpath : str Directory or filename prefix for export. Examples: ``"./exports/"`` or ``"./exports/run1_"``.
        nmf : str Identifier for the NMF model used (e.g., ``"deep"``, ``"multilayer"``, ``"deep-minvol"``). Included in the output filename.
        task : str, optional
            Task type (e.g., ``"regression"``, ``"classification"``) used to annotate the output filename.

        Notes
        -----
        - All factor matrices are converted to proportions before generating periodic maps.
        - The function exports an Excel workbook where each sheet corresponds to a rank in ``ranks``.
        - Designed for deep NMF representations where each ``H`` encodes element abundances across factors.

        Returns
        -------
        None

        The function writes an Excel file to ``outpath``.
        """
        arrays = Hs
        ranks = [h.shape[0] for h in Hs]
        layers = [i for i in range(len(Hs))]
        dfs = []
        for i, H in enumerate(arrays):
            H_weight = self._convert_to_proportions(H)
            compositions = self._layers_mapping(
                data=H_weight,
                lookup=lookup,
                layer=i,
                rank=ranks[i],
                threshold=threshold_by_layers[i],
                type=type_t,
                top_k=top_k,
            )
            dfs.append(pd.DataFrame([c.__dict__ for c in compositions]))
        if outfilename:
            with pd.ExcelWriter(outfilename, engine="openpyxl") as writer:
                for i, df in enumerate(dfs):
                    df.to_excel(writer, sheet_name=f"r{ranks[i]}", index=False)
        df = pd.concat(dfs)
        return df

    def map_factors(
        self,
        results: List[ResultData],
        lookup_file: str,
        top_k: int = 1,
    ) -> pd.DataFrame:
        """Creates df with feature importance for all ranks using self.topk_element_attribution with feature importance."""
        dfs = []
        ranks = np.array([r.rank for r in results])
        Hss = [[r.H] for r in results]
        importances = [r.feature_importance for r in results]
        noise = [r.noise for r in results]
        seed = [r.seed for r in results]
        score = [r.score.r2 for r in results]
        for r, _ in enumerate(ranks):
            df_a = self.topk_element_attribution(
                Hs=Hss[r],
                type_t="top_k",
                top_k=top_k,
                threshold_by_layers=[0.05],
                lookup=pd.read_excel(lookup_file),
            )
            df_a["importance"] = df_a["component_num"].map(lambda i: importances[r][i])
            # df_a["noise"] = df_a["component_num"].map(lambda i: noise[r][i])
            # df_a["seed"] = df_a["component_num"].map(lambda i: seed[r][i])
            # df_a["score"] = df_a["component_num"].map(lambda i: score[r][i])
            df_a["noise"] = [noise[r]] * len(df_a)
            df_a["seed"] = [seed[r]] * len(df_a)
            df_a["score"] = [score[r]] * len(df_a)
            # df_a["seed"] = seed[r]
            # df_a["score"] = score[r]
            dfs.append(df_a)
        return pd.concat(dfs)

    def heatmap(
        self,
        H,
        nmf: str,
        rank: int,
        colnames: List,
        export_outpath: str = None,
    ):
        """
        Selects the latest coefficient matrix (from Hs) with rows indicating ranks and columns the original data features. For PCA, use the model.components_ as data matrix
        """
        if nmf != "pca":
            H = self._convert_to_proportions(H)
        df = pd.DataFrame(
            H,
            columns=colnames,
            index=[f"c{i+1}" for i in range(rank)],
        )
        plt.figure()
        sns.heatmap(df, cmap="coolwarm", annot=False, fmt=".2f")
        plt.title(f"Heatmap {nmf.upper()} with rank={rank}")
        plt.xlabel("original feature names")
        plt.ylabel("Components")
        plt.tight_layout()
        if export_outpath:
            filename = f"heatmap_{nmf}_r{rank}.png"
            plt.savefig(export_outpath + filename)
            plt.close()
        else:
            plt.show()
        return True

    def _factor_sets(self, H: np.ndarray, top_k: int = 1, min_weight: float = 0.0):
        n_f, n_e = H.shape
        top_idx = np.argsort(H, axis=0)[-top_k:, :]
        factors_sets = [set() for _ in range(n_f)]
        for e in range(n_e):
            for r in range(top_k):
                f = top_idx[r, e]
                if H[f, e] >= min_weight:
                    factors_sets[f].add(e)
        return factors_sets

    def _top_p_labels(self, W: np.ndarray, p: float):
        num_f, num_e = W.shape
        B = np.zeros_like(W, dtype=int)
        top_k = max(1, int(p * num_e))
        for k in range(num_f):
            idx = np.argsort(W[k])[::-1][:top_k]
            B[k, idx] = 1
        return B

    def jaccard_matrix(
        self,
        H_l1: np.ndarray,
        H_l2: np.ndarray,
        top_k: int = 1,
        min_weight: float = 0.0,
    ):
        factor_sets_l1, factor_sets_l2 = (
            self._factor_sets(H_l1, top_k=top_k, min_weight=min_weight),
            self._factor_sets(H_l2, top_k=top_k, min_weight=min_weight),
        )
        n_prev = len(factor_sets_l1)
        n_curr = len(factor_sets_l2)
        J = np.zeros((n_prev, n_curr), dtype=float)
        for i in range(n_prev):
            A = factor_sets_l1[i]
            if not A:
                continue
            for j in range(n_curr):
                B = factor_sets_l2[j]
                if not B:
                    continue
                inter = len(A & B)
                union = len(A | B)
                J[i, j] = inter / union if union > 0 else 0.0
        return J

    def nmi_alignment(self, H1: np.ndarray, H2: np.ndarray, p: float = 0.02):
        H2_effective = H2 @ H1
        B1 = self._top_p_labels(H1, p)
        B2 = self._top_p_labels(H2_effective, p)
        num_f1, _ = B1.shape
        num_f2, _ = B2.shape
        nmi_matrix = np.zeros((num_f1, num_f2))
        for k in range(num_f1):
            for l in range(num_f2):
                nmi_matrix[k, l] = normalized_mutual_info_score(B1[k], B2[l])
        return nmi_matrix

    def plot_hierarchy_map(
        self,
        M: np.ndarray,
        annot=True,
        type: Literal["nmi", "jaccard"] = "jaccard",
        ax: plt.Axes = None,
        l1_name: str = "first_layer",
        l2_name: str = "second_layer",
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        if type == "jaccard":
            label = "jaccard_overlap"
            title = "Jaccard Overlap between Layers"
            ylabel = l1_name
            xlabel = l2_name
        elif type == "nmi":
            title = "NMI Alignment between Layers"
            ylabel = l1_name
            xlabel = l2_name
            label = "nmi_alignment"
        sns.heatmap(
            M,
            annot=annot,
            cmap="coolwarm",
            linewidths=0.5,
            linecolor="gray",
            # cbar_kws={"label": label},
            fmt=".1f",
            ax=ax,
        )
        ax.set_xlabel(xlabel=xlabel)
        ax.set_ylabel(ylabel=ylabel)
        ax.set_title(title)
        if ax is None:
            plt.show()

    def collect_data(
        self,
        results: List[ResultData],
        score_nmi: np.ndarray,
        convert_type: str = "division_base",
        end_rank: int = 10,
        division_base: float = 1.2,
    ) -> List[DepthsData]:
        depths = []
        for r, depth_score in zip(results, score_nmi):
            nmi_last = self._nmi_scores(r.Hl).tolist()[-1]
            depths.append(
                DepthsData(
                    nmf_model=r.nmf_model,
                    init=r.init,
                    convert_type=convert_type,
                    rank=r.rank,
                    depth=r.depths,
                    score=r.score.r2,
                    runtime=r.runtime,
                    nmi_last=nmi_last,
                    depth_score=float(depth_score),
                    end_rank=end_rank,
                    division_base=division_base,
                )
            )
        return depths

    def plot_performance_by_depths(
        self, results: List[ResultData], runner_params, params
    ):
        s = np.array([r.depths for r in results])
        y = np.array([r.score for r in results])
        x = np.array([r.rank for r in results])

        plt.figure()
        plt.scatter(x, y, s=s * 25, edgecolors="black", alpha=0.7)
        plt.xlabel("ranks")
        plt.ylabel("R^2")
        plt.title(
            f"ranks=[20, 40, 50, 70] for depths=[2...10] init={runner_params.init} \n division_base={params.division_base}"
        )
        plt.show()

    def search_depth_space(
        self,
        rank: int,
        depth: int,
        runner: Runner,
        runner_params: RunnerParams,
    ) -> int:
        # TODO: do not use runner here. Create a separate class for this or add to runner as exp.
        runner_params.deep_params.set_layer_depth(depth)
        result = runner.run_evaluation(params=runner_params, ranks=[rank])
        runner_params.deep_params.lam = None
        nmis = self._nmi_scores(result[0].Hl)
        max_depths = np.where(nmis < 0.5)[0]
        if max_depths.size == 0:
            depth = len(nmis)
        else:
            int(max_depths[0] + 1)
        return depth

    def depths_tunner(
        self,
        ranks: List[int],
        runner: Runner,
        runner_params: RunnerParams,
        depth_limit: int = 30,
        weight: List[float] = [0.7, 0.3],
    ) -> List:
        results = []
        for rank in ranks:
            max_depth = self.search_depth_space(
                rank, depth_limit, runner, runner_params
            )
            depths = [i for i in range(2, max_depth + 1)]
            for depth in depths:
                runner_params.deep_params.set_layer_depth(depth)
                result = runner.run_evaluation(params=runner_params, ranks=[rank])
                results.append(result)
                runner_params.deep_params.lam = None
        results = np.concatenate(results).tolist()
        sc_nmi = self.depths_scores(
            results=results,
            weight=weight,
        )
        depths_data = self.collect_data(results, sc_nmi)
        df = pd.DataFrame([db.__dict__ for db in depths_data])
        df = df.loc[df.groupby("rank")["depth_score"].idxmax()]
        depths_data = [d for k, d in enumerate(depths_data) if k in df.index.tolist()]
        return [depths_data, results]

    def performance_deviation(
        self,
        results_data: np.ndarray[ResultData],
        score_type: Literal["accuracy", "r2", "mae", "rmse"] = "rmse",
        outtype: Literal["deviation", "raw"] = "deviation",
    ) -> np.ndarray:
        """minvol_r: tensor with shape=(seed, noise, ranks)"""
        if len(results_data.shape) == 3:
            n_seed, n_noise, n_rank = results_data.shape
            x = results_data.transpose(0, 2, 1)
            results_data = x.reshape(n_seed * n_rank, n_noise)

        # expected shape=(#, noise_levels)
        if score_type == "r2":
            scores_noise = np.vectorize(lambda x: x.score.r2)(results_data)
        elif score_type == "accuracy":
            scores_noise = np.vectorize(lambda x: x.score.accuracy)(results_data)
        elif score_type == "mae":
            scores_noise = np.vectorize(lambda x: x.score.mae)(results_data)
        elif score_type == "rmse":
            scores_noise = np.vectorize(lambda x: x.score.rmse)(results_data)
        if outtype == "deviation":
            scores_noise = np.abs(scores_noise - scores_noise[:, 0].reshape(-1, 1))
        return scores_noise

    def plot_noise_score_influence(
        self,
        minvol_r: List[ResultData],
        un_r: List[ResultData],
        noise_levels: List[float],
    ):
        n = minvol_r.shape[0]
        if len(minvol_r.shape) == 3:
            n_seed, _, n_rank = minvol_r.shape
            n = n_seed * n_rank
        # shape=(#, noise_levels)
        plt.figure()
        for k, results in enumerate([minvol_r, un_r]):
            scores = self.performance_deviation(results)
            mean_curve = np.abs(scores.mean(0))
            std_curve = np.abs(scores.std(0))
            l = ["minvol", "unconstraint"]
            plt.plot(mean_curve, marker="o", linestyle="--", label=f"{l[k]}")
            # plt.fill_between(
            #     np.arange(len(mean_curve)),
            #     mean_curve - std_curve,
            #     mean_curve + std_curve,
            #     alpha=0.2,
            #     color="steelblue",
            # )
        plt.legend()
        plt.xlabel("Noise Levels")
        # plt.title("shows influence of noise levels on performance of nmf models")
        plt.ylabel(f"MAD-RMSE N={n}")
        plt.xticks(
            ticks=np.arange(len(noise_levels)),
            labels=noise_levels,
            rotation=45,  # optional
        )
        plt.savefig("./exports/mad_rmse.png")
        plt.show()

    def plot_noise_score_influence_from_df(
        self,
        df: pd.DataFrame,
        col_metric: Literal["r2", "rmse", "mae", "f1_macro", "accuracy"] = "r2",
    ):
        """
        df must contain columns:
        ['r2', 'noise', 'rank', 'seed', 'model']
        where model ∈ {'minvol', 'unconstraint'}
        """
        noise_levels = sorted(df["noise"].unique())
        models = ["minvol", "unconstraint"]
        plt.figure(figsize=(8, 5))
        for model in models:
            sub = df[df["model"] == model]
            pivot = sub.pivot_table(
                index=["seed", "rank"],
                columns="noise",
                values=col_metric,
            ).sort_index(axis=1)
            baseline = pivot[0.0].values.reshape(-1, 1)
            deviation = np.abs(pivot.values - baseline)
            mean_curve = deviation.mean(axis=0)
            std_curve = deviation.std(axis=0)
            plt.plot(mean_curve, marker="o", linestyle="--", label=model)
            # plt.fill_between(
            #     np.arange(len(mean_curve)),
            #     mean_curve - std_curve,
            #     mean_curve + std_curve,
            #     alpha=0.2,
            #     color="steelblue",
            # )
        plt.xlabel("Noise Levels")
        plt.ylabel(f"MAD-R² N={50}")
        plt.xticks(
            ticks=np.arange(len(noise_levels)),
            labels=noise_levels,
            rotation=45,  # optional
        )
        plt.legend()
        # plt.title("Influence of Noise on R² Performance")
        plt.tight_layout()
        plt.show()

    def build_element_rank_matrix(self, df: pd.DataFrame):
        ranks = sorted(df["rank"].unique())
        elements = sorted(df["element_name"].unique())
        matrix = pd.DataFrame(0, index=ranks, columns=elements)
        for r in ranks:
            subset: pd.DataFrame = df[df["rank"] == r]
            counts = subset["element_name"].value_counts()
            matrix.loc[r, counts.index] = counts.values

        return matrix

    def compute_entropy(
        self,
        df: pd.DataFrame,
        ranks: list[int],
        noise_lvls: list[float],
        seeds: list[int],
        outtype: Literal["deviation", "raw"] = "deviation",
    ):
        """Return array shape=(rank, noise_levels)"""  # shape=(seed, noise, rank)
        EE = []
        for seed in seeds:
            E = []
            for rank in ranks:
                e = []
                for n in noise_lvls:
                    _df = df[
                        (df.noise == n) & (df["rank"] == rank) & (df["seed"] == seed)
                    ]
                    attribution = self.build_element_rank_matrix(_df).to_numpy()
                    e.append(entropy(attribution[0], base=2))
                E.append(np.array(e))
            EE.append(np.array(E))
        EE = np.array(EE)
        En = EE.reshape(EE.shape[0] * EE.shape[1], EE.shape[2])
        if outtype == "deviation":
            En = np.abs(En - En[:, 0].reshape(-1, 1))
        return En

    def plot_entropy_noise(
        self,
        entropy_minvol: np.ndarray,
        entropy_un: np.ndarray,
        noise_lvl: List[float],
    ):
        """Takes in array shape=(rank, noise_levels). Ensure to use `self.compute_entropy` on List[ResultsData]"""
        n = entropy_minvol.shape[0]
        plt.figure()
        for k, entropy_ in enumerate([entropy_minvol, entropy_un]):
            l = ["minvol", "unconstraint"]
            plt.plot(entropy_.mean(0), marker="o", linestyle="--", label=l[k])
            mean_curve = np.abs(entropy_.mean(0))
            std_curve = np.abs(entropy_.std(0))
            # plt.fill_between(
            #     np.arange(len(mean_curve)),
            #     mean_curve - std_curve,
            #     mean_curve + std_curve,
            #     alpha=0.2,
            #     color="steelblue",
            # )
        plt.ylabel(f"MAD-Entropy N={n}")
        plt.xlabel("Noise Levels")
        plt.xticks(
            ticks=np.arange(len(noise_lvl)), labels=noise_lvl, rotation=45  # optional
        )
        plt.legend()
        # plt.title(
        #     "influence of noise levels on factor labelling through entropy values"
        # )
        plt.savefig("./exports/mad_entropy.png")
        plt.show()

    def top_attributions(self, results: List[ResultData]):
        df_importance = self.map_factors(
            results=results.reshape(-1),
            lookup_file="./data/lookup.xlsx",
        )
        out = df_importance.sort_values(
            ["noise", "rank", "abundance"], ascending=[True, True, False]
        )
        top_rows = out.groupby(["noise", "rank"]).head(1)
        return top_rows

    def extract_field(self, resultdata_tensor: np.ndarray, field: str):
        num_seeds = len(resultdata_tensor)
        num_noise = len(resultdata_tensor[0])
        num_ranks = len(resultdata_tensor[0][0])
        out = np.empty((num_seeds, num_noise, num_ranks), dtype=object)
        for s in range(num_seeds):
            for n in range(num_noise):
                for r in range(num_ranks):
                    obj = resultdata_tensor[s][n][r]
                    value = getattr(obj, field)
                    out[s, n, r] = value
        try:
            return out.astype(float)
        except Exception:
            return out

    def results_tensor_to_df(self, results_data: List[ResultData]):
        rows = []
        n_seed, n_noise, n_rank = results_data.shape
        for s in range(n_seed):
            for n in range(n_noise):
                for r in range(n_rank):
                    obj: ResultData = results_data[s, n, r]
                    rows.append(
                        {
                            "rmse": obj.score.rmse,
                            "r2": obj.score.r2,
                            "f1_macro": obj.score.f1_macro,
                            "accuracy": obj.score.accuracy,
                            "mae": obj.score.mae,
                            "seed": obj.seed,
                            "noise": obj.noise,
                            "rank": obj.rank,
                        }
                    )
        return pd.DataFrame(rows)

    def element_coherence(self, df: pd.DataFrame, feature: str, g_classes: List[float]):
        cls_counts = df[feature].value_counts(normalize=True)
        H = entropy(cls_counts)
        G = entropy(g_classes)
        return 1 - (H / G)

    def element_diversity(self, df: pd.DataFrame, feature: str):
        unique_cls = df[feature].unique().tolist()
        return len(unique_cls) / df[feature].shape[0]

    def evaluate_attribution(
        self,
        df: pd.DataFrame,
        feature: Literal["element_num", "element_name", "metal_class"],
        g_classes: List[float],
    ) -> pd.DataFrame:
        """Evaluates the coherence and diversity of elemental attributions by nmf_models and returns a df across ranks and topk for each factor sorted by feature importance."""
        results = []
        for model, df_model in df.groupby("nmf_model"):
            model_coherence = self.element_coherence(df_model, feature, g_classes)
            model_diversity = self.element_diversity(df_model, feature)
            row = {
                "nmf_model": model,
                "coherence": model_coherence,
                "diversity": model_diversity,
            }
            class_counts = df_model[feature].value_counts()
            for cls, count in class_counts.items():
                row[f"cl_{cls}"] = count
            results.append(row)
        df_results = pd.DataFrame(results)
        return df_results
