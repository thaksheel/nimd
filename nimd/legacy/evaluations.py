from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (
    cross_val_score,
    RepeatedStratifiedKFold,
    RepeatedKFold,
    train_test_split,
)
import multiprocessing
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nmfs import NNLS
import nmfs.hierNMF as hierNMF
from nmfs.betaNMF import betaNMF
from scipy.optimize._nnls import nnls
from nmfs.nnsvdlrc import NNSVDLRC
from typing import Literal, Optional, Dict, List
from dataclasses import dataclass, asdict
from sklearn.decomposition._nmf import norm
from sklearn.utils.validation import check_array, check_non_negative, check_random_state
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import PCA
import math
import time
import seaborn as sns

from .nmfs.ssnmf import SSNMF
import nmfs.FroNMF as FroNMF
from nmfs import multilayerKLNMF
from nmfs import deepKLNMF
from nmfs.deepKLNMF import DeepNMFParams

p = time.time()


@dataclass
class HierarchyAbundance:
    component_num: int
    abundance: float
    rank: int
    layer: int
    threshold: float
    previous_component_num: int = 0
    element_num: int = None
    element_name: str = None
    metal_class: str = None


@dataclass
class SSNMFParam:
    model_num: Literal[3, 4, 5, 6] = 3
    split: float = 0.2
    tol: float = 1e-4
    numiters: int = 100
    saveerrs: bool = False
    iter_s: int = 20
    lam: float = 1e-4
    seed: int = 47


class NMFEvaluations:
    def __init__(
        self,
        df,
        X,
        y,
        task: Literal["regression", "classification"],
        parallel: bool,
        deepparams: DeepNMFParams,
        ssnmfparams: SSNMFParam,
        layers: int,
        cross_validation: bool,
        evaluation_measure: Literal["classic", "ratio", "xy"] = "classic",
        seed: int = 123,
    ):
        self.df = df
        self.task = task
        self.parallel = parallel
        self.cross_validation = cross_validation
        self.seed = seed
        self.deepparams = deepparams
        self.ssnmfparams = ssnmfparams
        self.layers = layers
        self.evaluation_measure = evaluation_measure

        # null fields
        self.Ws: List[np.ndarray] = []
        self.Hs: List[np.ndarray] = []
        self.Wl: List[np.ndarray] = []
        self.Hl: List[np.ndarray] = []
        self.evals = []
        self.rank = None
        self.ranks = []
        self.init = None
        self.evaluation_type = None
        self.nmf = None
        self.pca_components: List[np.ndarray] = []

        # operations
        # TODO: there is somewthing wrong here when random_state = 21
        self.eval0 = self.supervised_learning(
            *train_test_split(X, y, test_size=0.2, random_state=seed)
        )
        if type(X) == pd.core.frame.DataFrame:
            X = X.to_numpy()
        if type(y) == pd.core.series.Series:
            y = y.to_numpy()
        self.X = X
        self.y = y

    def _evaluate_wrapper(self, args):
        _, nmf, init, rank, eval_type, norm_X, norm_init, algo, exports = args
        return self.evaluate(
            nmf, init, rank, eval_type, norm_X, norm_init, algo, exports
        )

    def layers_mapping(
        self,
        data: np.ndarray,
        lookup: pd.DataFrame,
        layer: int,
        rank: int,
        threshold: float = 0.01,
        element_cut_off: int = 64,
    ):
        # TODO: threshold has to be fine tuned to return the total number of elements each time
        composition: List[HierarchyAbundance] = []
        df = pd.DataFrame(data[:, :element_cut_off])
        for i, row in df.iterrows():
            mask = row > threshold
            abundance_series = row[mask == True]
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
                if layer == 0:
                    ha.element_name = lookup[lookup["i"] == idx[j]]["elements"].iloc[0]
                    ha.element_num = idx[j]
                    ha.metal_class = lookup[lookup["i"] == idx[j]]["class"].iloc[0]
                composition.append(ha)
        return composition

    def convert_to_proportions(self, data: np.ndarray):
        data_ = []
        for row in data:
            arr = [(el - row.min()) / (row.max() - row.min()) for el in row]
            data_.append([i / sum(arr) for i in arr])
        return np.array(data_)

    def periodic_maps(
        self,
        threshold_by_layers: List[float],
        layers: List[int],
        lookup: pd.DataFrame,
        outpath: str,
    ):
        """
        Here always converting to proportions before export of periodic map
        """
        # if self.nmf == "deep" or self.nmf == "multilayer" and self.ranks is None:
        #     arrays = self.Hl
        # else:
        arrays = self.pca_components if self.nmf == "pca" else self.Hs
        filename = outpath + f"periodic_map_{self.nmf}_{self.task}.xlsx"
        dfs = []
        for i, H in enumerate(arrays):
            H_weight = self.convert_to_proportions(H)
            compositions = [
                self.layers_mapping(
                    H_weight,
                    lookup,
                    layer,
                    self.ranks[i],
                    threshold_by_layers[i],
                )
                for i, layer in enumerate(layers)
            ]
            dfs.append(
                pd.DataFrame(
                    [
                        asdict(ar)
                        for ar in [
                            com for composition in compositions for com in composition
                        ]
                    ]
                )
            )
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for i, df in enumerate(dfs):
                df.to_excel(writer, sheet_name=f"r{self.ranks[i]}", index=False)
        return self

    def heatmap(self, colnames: List, data_matrix: np.ndarray = None):
        """
        Selects the latest coefficient matrix (from Hs) with rows indicating ranks and columns the original data features. For PCA, use the model.components_ as data matrix
        """
        if data_matrix == None and self.nmf != "pca":
            data_matrix = self.Hs[-1]
        elif data_matrix == None and self.nmf == "pca":
            data_matrix = self.convert_to_proportions(self.pca_components[-1])
        df = pd.DataFrame(
            data_matrix,
            columns=colnames,
            index=[f"c{i+1}" for i in range(self.rank)],
        )
        plt.figure()
        sns.heatmap(df, cmap="coolwarm", annot=False, fmt=".2f")
        plt.title(f"Heatmap {self.nmf.upper()} with rank={self.rank}")
        plt.xlabel("original feature names")
        plt.ylabel("Components")
        plt.tight_layout()
        plt.show()
        return self

    def save_heatmaps(self, colnames: List, outpath: str):
        """
        Uses a coefficient matrix (H) with rows indicating ranks and columns the original data features. For PCA, use the model.components_ as data matrix
        """
        if self.nmf != "pca":
            data_matrices = self.Hs
        elif self.nmf == "pca":
            data_matrices = [
                self.convert_to_proportions(component)
                for component in self.pca_components
            ]
        for i, data_matrix in enumerate(data_matrices):
            df = pd.DataFrame(
                data_matrix,
                columns=colnames,
                index=[f"r{i+1}" for i in range(data_matrix.shape[0])],
            )
            filename = f"heatmap_{self.nmf}_r{self.ranks[i]}.png"
            plt.figure()
            sns.heatmap(df, cmap="coolwarm", annot=False, fmt=".2f")
            plt.title(f"Heatmap {self.nmf} with ranks={self.ranks[i]}")
            plt.xlabel("original feature names")
            plt.ylabel("ranks")
            plt.tight_layout()
            plt.savefig(outpath + filename)
            plt.close()
        return self

    def evaluate(
        self,
        nmf: Literal["ssnmf", "deep", "multilayer", "fronorm", "beta", "hier", "pca"],
        init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"],
        rank: int,
        evalutation_type: Literal["feature", "full"],
        normalize_X: Optional[Literal["minmax"]],
        normalize_init: Optional[Literal["scaling", "feature_norm"]],
        algo: Optional[Literal["MUUP", "ADMM", "HALS", "FPGM", "ALSH"]],
    ):
        if self.evaluation_measure == "classic":
            out = self._evaluate(
                nmf, init, rank, evalutation_type, normalize_X, normalize_init, algo
            )
        else:
            pca_eval = self._evaluate(
                "pca", init, rank, evalutation_type, normalize_X, normalize_init, algo
            )
            nmf_eval = self._evaluate(
                nmf, init, rank, evalutation_type, normalize_X, normalize_init, algo
            )
            if self.evaluation_measure == "ratio":
                out = nmf_eval[0] / pca_eval[0]
            elif self.evaluation_measure == "xy":
                out = [nmf_eval[0], pca_eval[0]]
        return out

    def evaluates(
        self,
        nmf: Literal["ssnmf", "deep", "multilayer", "fronorm", "beta", "hier", "pca"],
        init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"],
        evalutation_type: Literal["feature", "full"],
        normalize_X: Optional[Literal["minmax"]],
        normalize_init: Optional[Literal["scaling", "feature_norm"]],
        algo: Optional[Literal["MUUP", "ADMM", "HALS", "FPGM", "ALSH"]],
        ranks: Optional[List],
    ):
        if self.evaluation_measure == "classic":
            outs = self._evaluates(
                nmf, init, evalutation_type, normalize_X, normalize_init, algo, ranks
            )
        else:
            pca_eval = self._evaluates(
                "pca", init, evalutation_type, normalize_X, normalize_init, algo, ranks
            )
            nmf_eval = self._evaluates(
                nmf, init, evalutation_type, normalize_X, normalize_init, algo, ranks
            )
            pca_eval_mean = [np.mean(e) for e in pca_eval]
            nmf_eval_mean = [np.mean(e) for e in nmf_eval]
            if self.evaluation_measure == "ratio":
                outs = [e / pca_eval_mean[i] for i, e in enumerate(nmf_eval_mean)]
            elif self.evaluation_measure == "xy":
                outs = [nmf_eval_mean, pca_eval_mean]
        return outs

    def _evaluates(
        self,
        nmf: Literal["ssnmf", "deep", "multilayer", "fronorm", "beta", "hier", "pca"],
        init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"],
        evalutation_type: Literal["feature", "full"],
        normalize_X: Optional[Literal["minmax"]],
        normalize_init: Optional[Literal["scaling", "feature_norm"]],
        algo: Optional[Literal["MUUP", "ADMM", "HALS", "FPGM", "ALSH"]],
        ranks: Optional[List],
    ):
        if self.parallel:
            results = []
            with multiprocessing.Pool(processes=12) as pool:
                args_list = [
                    (
                        self,
                        nmf,
                        init,
                        rank,
                        evalutation_type,
                        normalize_X,
                        normalize_init,
                        algo,
                    )
                    for rank in ranks
                ]
                results = pool.map(self._evaluate_wrapper, args_list)
            return results
        else:
            return [
                self._evaluate(
                    nmf,
                    init,
                    rank,
                    evalutation_type,
                    normalize_X,
                    normalize_init,
                    algo,
                )
                for rank in ranks
            ]

    def _evaluate(
        self,
        nmf: Literal["ssnmf", "deep", "multilayer", "fronorm", "beta", "hier", "pca"],
        init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"],
        rank: int,
        evalutation_type: Literal["feature", "full"],
        normalize_X: Optional[Literal["minmax"]],
        normalize_init: Optional[Literal["scaling", "feature_norm"]],
        algo: Optional[Literal["MUUP", "ADMM", "HALS", "FPGM", "ALSH"]],
    ):
        self.init = init
        self.nmf = nmf
        self.rank = rank
        self.evaluation_type = evalutation_type
        # TODO: init for betanmf in multilayer has to be implemented for each layer so there should be an init method in multilayer
        if nmf not in ["deep", "multilayer", "ssnmf"]:
            W0, H0 = self.initialize_nmf(
                self.X, rank, init=init, random_state=self.seed
            )
            self.deepparams.W0, self.deepparams.H0 = W0.copy(), H0.copy()
        X = self.X.copy()
        if normalize_X:
            X = self.normalize(normalize=normalize_X)
        # TODO: this is not implemented properly since W0 and H0 will change
        if normalize_init:
            W0, H0 = self.normalize(normalize=normalize_init)
        if algo and nmf == "fronorm":
            W, H, *_ = FroNMF.fro_nmf(X, rank, algo, W0, H0)
        elif nmf == "beta":
            W, H, e, t = betaNMF(X, rank, options={"W": W0, "H": H0})
        elif nmf == "hier":
            # TODO: there is no space to add an init here
            # TODO: error when r > 70
            if rank > 68:
                rank = 68
            IDX, W, _ = hierNMF.hierclust2nmf(X, rank)
            H, *_ = NNLS.NNLS(W, X)
        elif nmf == "ssnmf":
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=self.ssnmfparams.split, random_state=self.seed
            )
            W0, H0 = self.initialize_nmf(
                X_train, rank, init=init, random_state=self.seed
            )
            if self.task == "regression":
                y = np.array([y_train.copy().T])
            elif self.task == "classification":
                y = pd.get_dummies(y_train).T.to_numpy().astype(int)
            A0, S0, B0 = (
                H0.T,
                W0.T,
                self.correct_matrix(np.random.rand(y.shape[0], rank)),
            )
            model = SSNMF(
                X_train.T,
                rank,
                Y=y,
                lam=self.ssnmfparams.lam,
                tol=self.ssnmfparams.tol,
                A=A0,
                B=B0,
                S=S0,
                modelNum=self.ssnmfparams.model_num,
            )
            model.mult(
                numiters=self.ssnmfparams.numiters,
                saveerrs=self.ssnmfparams.saveerrs,
                eps=1e-14,
            )
            S_train, A_train = model.S, model.A
            S_test = self.get_s(
                model,
                X_test.T,
                A_train,
                self.ssnmfparams.model_num,
                iter_s=self.ssnmfparams.iter_s,
            )
            return self.evaluation(
                X=None,
                W=None,
                evaluation_type=evalutation_type,
                X_train=S_train.T,
                X_test=S_test.T,
                y_train=y_train,
                y_test=y_test,
            )
        elif nmf == "multilayer":
            rank_layer = [math.ceil(rank / pow(2, i)) for i in range(self.layers)]
            # NOTE: do not need to transpose X such that Wl are the features
            # TODO: create a multilayer param specific to multilayer
            Wl, Hl, _ = multilayerKLNMF.multilayer_klnmf(
                X, rank_layer, init, asdict(self.deepparams)
            )
            W = Wl[-1]
            for i, H in enumerate(Hl[::-1]):
                if i == len(Hl) - 1:
                    break
                W = W @ H
            H = Hl[0].T
        elif nmf == "deep":
            rank_layer = [math.ceil(rank / pow(2, i)) for i in range(self.layers)]
            Wl, Hl, *_ = deepKLNMF.deep_kl_nmf(X, rank_layer, init, self.deepparams)
            W = Wl[-1]
            for i, H in enumerate(Hl[::-1]):
                if i == len(Hl) - 1:
                    break
                W = W @ H
            H = Hl[0]
        elif nmf == "pca":
            pca = PCA(n_components=rank)
            W = pca.fit_transform(X)
            H = np.ones(H0.shape)
            self.pca_components.append(pca.components_)
        self.Ws.append(W)
        self.Hs.append(H)
        self.ranks.append(rank)
        return self.evaluation(W @ H, W, evalutation_type)

    def exports(
        self, exports: Literal["full", "layers", "recon", "factors"], outpath: str
    ):
        if exports == "layers":
            # TODO: this only works for one rank iteration instead of ranks
            for type in ["Wl", "Hl"]:
                filename = outpath + f"{type}_layer{self.layers}_rank{self.rank}.xlsx"
                arrays = self.Wl if type == "Wl" else self.Hl
                with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                    for i, array in enumerate(arrays):
                        df = pd.DataFrame(array)
                        df.to_excel(writer, sheet_name=f"l{i+1}", index=False)
        elif exports == "recon":
            filename = outpath + f"Xs_{self.nmf}_{self.task}.xlsx"
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                arrays = [i @ self.Hs[j] for j, i in enumerate(self.Ws)]
                for i, array in enumerate(arrays):
                    df = pd.DataFrame(array)
                    df.to_excel(writer, sheet_name=f"r{i+2}", index=False)
        elif exports == "factors":
            for type in ["Ws", "Hs"]:
                filename = outpath + f"{type}_{self.nmf}_{self.task}.xlsx"
                with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                    if type == "Ws":
                        arrays = self.Ws
                    elif type == "Hs":
                        arrays = self.Hs
                    for i, array in enumerate(arrays):
                        df = pd.DataFrame(array)
                        df.to_excel(writer, sheet_name=f"r{i+2}", index=False)
        elif exports == "full":
            for type in ["Ws", "Hs", "Xs"]:
                filename = outpath + f"{type}_{self.nmf}_{self.task}.xlsx"
                with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                    if type == "Ws":
                        arrays = self.Ws
                    elif type == "Hs":
                        arrays = self.Hs
                    elif type == "Xs":
                        arrays = [i @ self.Hs[j] for j, i in enumerate(self.Ws)]
                    for i, array in enumerate(arrays):
                        df = pd.DataFrame(array)
                        df.to_excel(writer, sheet_name=f"r{i+2}", index=False)
        return self

    def evaluation(self, X, W, evaluation_type, **kwargs):
        """
        Select 'full' vs 'feature' matrix evaluation and pass to supervised_learning()
        """
        if evaluation_type == "full" and X is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, self.y, test_size=0.2, random_state=self.seed
            )
        elif evaluation_type == "feature" and X is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                W, self.y, test_size=0.2, random_state=self.seed
            )
        elif X is None and W is None:
            X_train = kwargs.get("X_train")
            X_test = kwargs.get("X_test")
            y_train = kwargs.get("y_train")
            y_test = kwargs.get("y_test")
        self.evals = self.supervised_learning(X_train, X_test, y_train, y_test)
        return self.evals

    def normalize(self, normalize: Literal["minmax", "scaling", "feature_norm"]):
        if normalize == "minmax":
            col_min = self.X.min(axis=0)
            col_max = self.X.max(axis=0)
            Xnorm = (self.X - col_min) / (col_max - col_min)
            return Xnorm
        elif normalize == "feature_norm":
            # Normalize W and H so that columns/rows have the same norm, that is,  ||W(:,k)|| = ||H(k,:)|| for all k.
            normW = np.sqrt(np.sum(W**2, axis=0)) + 1e-16
            normH = np.sqrt(np.sum(H**2, axis=1)) + 1e-16
            d = np.sqrt(normW) / np.sqrt(normH)
            H *= d[:, None]
            for k in range(W.shape[1]):
                W[:, k] = W[:, k] / np.sqrt(normW[k]) * np.sqrt(normH[k])
            return W, H
        elif normalize == "scaling":
            # Scale initialization so that argmin_a ||a * WH - X||_F = 1
            XHt = self.X @ H.T
            HHt = H @ H.T
            scaling = np.sum(XHt * W) / np.sum(HHt * (W.T @ W))
            W *= scaling
            return W, H

    def check_init(self, A, shape, whom):
        A = check_array(A)
        if shape[0] != "auto" and A.shape[0] != shape[0]:
            raise ValueError(
                f"Array with wrong first dimension passed to {whom}. Expected {shape[0]}, "
                f"but got {A.shape[0]}."
            )
        if shape[1] != "auto" and A.shape[1] != shape[1]:
            raise ValueError(
                f"Array with wrong second dimension passed to {whom}. Expected {shape[1]}, "
                f"but got {A.shape[1]}."
            )
        check_non_negative(A, whom)
        if np.max(A) == 0:
            raise ValueError(f"Array passed to {whom} is full of zeros.")

    def correct_matrix(self, W: np.ndarray) -> np.ndarray:
        """Corrects for zeros and negative values to machine epsillon using np.finfo(float).eps"""
        machine_epsilon = np.finfo(float).eps
        W[W == 0] = machine_epsilon
        W = np.maximum(W, machine_epsilon)
        return W

    def find_sparsity(self, X: np.ndarray):
        return np.count_nonzero(X == 0) / X.size

    def initialize_nmf(
        self,
        X,
        n_components,
        init: str = Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"],
        eps=1e-6,
        random_state=None,
    ):
        """Algorithms for NMF initialization.

        Computes an initial guess for the non-negative
        rank k matrix approximation for X: X = WH.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix to be decomposed.

        n_components : int
            The number of components desired in the approximation.

        init :  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
            Method used to initialize the procedure.
            Valid options:

            - None: 'nndsvda' if n_components <= min(n_samples, n_features),
                otherwise 'random'.

            - 'random': non-negative random matrices, scaled with:
                sqrt(X.mean() / n_components)

            - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
                initialization (better for sparseness)

            - 'nndsvda': NNDSVD with zeros filled with the average of X
                (better when sparsity is not desired)

            - 'nndsvdar': NNDSVD with zeros filled with small random values
                (generally faster, less accurate alternative to NNDSVDa
                for when sparsity is not desired)

            - 'custom': use custom matrices W and H

            .. versionchanged:: 1.1
                When `init=None` and n_components is less than n_samples and n_features
                defaults to `nndsvda` instead of `nndsvd`.

        eps : float, default=1e-6
            Truncate all values less then this in output to zero.

        random_state : int, RandomState instance or None, default=None
            Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
            reproducible results across multiple function calls.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        W : array-like of shape (n_samples, n_components)
            Initial guesses for solving X ~= WH.

        H : array-like of shape (n_components, n_features)
            Initial guesses for solving X ~= WH.

        References
        ----------
        C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
        nonnegative matrix factorization - Pattern Recognition, 2008
        http://tinyurl.com/nndsvd
        """
        check_non_negative(X, "NMF initialization")
        n_samples, n_features = X.shape

        if (
            init is not None
            and init != "random"
            and n_components > min(n_samples, n_features)
        ):
            raise ValueError(
                "init = '{}' can only be used when "
                "n_components <= min(n_samples, n_features)".format(init)
            )

        if init is None:
            if n_components <= min(n_samples, n_features):
                init = "nndsvda"
            else:
                init = "random"

        # NNSVDLRC initialization
        if init == "nnsvdlrc":
            W, H, *_ = NNSVDLRC(X, n_components)
            return W, H

        # Random initialization
        if init == "random":
            avg = np.sqrt(X.mean() / n_components)
            rng = check_random_state(random_state)
            H = avg * rng.standard_normal(size=(n_components, n_features)).astype(
                X.dtype, copy=False
            )
            W = avg * rng.standard_normal(size=(n_samples, n_components)).astype(
                X.dtype, copy=False
            )
            np.abs(H, out=H)
            np.abs(W, out=W)
            return W, H

        # NNDSVD initialization
        U, S, V = randomized_svd(X, n_components, random_state=random_state)
        W = np.zeros_like(U)
        H = np.zeros_like(V)

        # The leading singular triplet is non-negative
        # so it can be used as is for initialization.
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

        for j in range(1, n_components):
            x, y = U[:, j], V[j, :]

            # extract positive and negative parts of column vectors
            x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
            x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

            # and their norms
            x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
            x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

            m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

            # choose update
            if m_p > m_n:
                u = x_p / x_p_nrm
                v = y_p / y_p_nrm
                sigma = m_p
            else:
                u = x_n / x_n_nrm
                v = y_n / y_n_nrm
                sigma = m_n

            lbd = np.sqrt(S[j] * sigma)
            W[:, j] = lbd * u
            H[j, :] = lbd * v

        W[W < eps] = 0
        H[H < eps] = 0

        if init == "nndsvd":
            pass
        elif init == "nndsvda":
            avg = X.mean()
            W[W == 0] = avg
            H[H == 0] = avg
        elif init == "nndsvdar":
            rng = check_random_state(random_state)
            avg = X.mean()
            W[W == 0] = abs(avg * rng.standard_normal(size=len(W[W == 0])) / 100)
            H[H == 0] = abs(avg * rng.standard_normal(size=len(H[H == 0])) / 100)
        else:
            raise ValueError(
                "Invalid init parameter: got %r instead of one of %r"
                % (init, (None, "random", "nndsvd", "nndsvda", "nndsvdar"))
            )

        return W, H

    def supervised_learning(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
    ):
        if self.task == "regression":
            model = LinearRegression(n_jobs=-1)
            scoring = "r2"
            cv_strategy = RepeatedKFold(
                n_splits=5, n_repeats=10, random_state=self.seed
            )
        elif self.task == "classification":
            model = LogisticRegression(n_jobs=-1)
            scoring = "accuracy"
            cv_strategy = RepeatedStratifiedKFold(
                n_splits=5, n_repeats=10, random_state=self.seed
            )
        if self.cross_validation == False:
            model = model.fit(X_train, y_train)
            test_scores = [model.score(X_test, y_test)]
        elif self.cross_validation == True:
            X = np.concatenate((X_train, X_test), axis=0)
            model = LinearRegression(n_jobs=-1)
            test_scores = cross_val_score(
                model, X, self.y, cv=cv_strategy, scoring=scoring
            )
        return test_scores

    def get_s(
        self,
        model: SSNMF,
        X_test: np.ndarray,
        A_train: np.ndarray,
        model_num: int,
        iter_s: int = 20,
    ):
        """
        Given a trained (S)SNMF model i.e. learned data dictionary, A, the function applies the (S)SNMF model
        to the test data to produce the representation of the test data.

        Returns:
            S_test (ndarray): representation matrix of the test data, shape(#topics, #test features)
        """
        r = A_train.shape[1]
        n1 = X_test.shape[0]
        n2 = X_test.shape[1]
        # Frobenius measure on features data (use nonnegative least squares)
        if model_num == 3 or model_num == 4:
            S_test = np.zeros([r, n2])
            err = []
            for i in range(n2):
                S_test[:, i], e = nnls(A_train, X_test[:, i])
                # err.append(e/np.linalg.norm(X_test[:, i]))
        # I-divergence measure on features data (use mult. upd. I-div)
        np.random.seed(41)
        W_test = np.ones((n1, n2))
        if model_num == 5 or model_num == 6:
            S_test = np.random.rand(r, X_test.shape[1])
            for i in range(iter_s):
                S_test = np.transpose(
                    model.dictupdateIdiv(
                        np.transpose(X_test),
                        np.transpose(S_test),
                        np.transpose(A_train),
                        np.transpose(W_test),
                        eps=1e-10,
                    )
                )
        return S_test

    def affichage(
        self,
        ranks,
        evals,
        outfile: str,
        title: str,
        save: bool = False,
        truncate: bool = False,
    ):
        if self.evaluation_measure != "xy":
            if truncate:
                accuracy_mean = [np.mean(e) if np.mean(e) > 0 else 0 for e in evals]
            else:
                accuracy_mean = [np.mean(e) for e in evals]
        if self.evaluation_measure == "classic":
            eval0 = self.eval0
        elif self.evaluation_measure == "ratio":
            eval0 = 1
        elif self.evaluation_measure == "xy":
            eval0 = [e for e in evals[1]]
        if self.cross_validation == False:
            if self.evaluation_measure != "xy":
                plt.figure()
                plt.plot(ranks, accuracy_mean, "o-", markersize=3)
                plt.axhline(
                    y=eval0,
                    color="red",
                    linestyle="--",
                    label="baseline",
                )
                plt.title(title)
                if self.task == "regression":
                    plt.ylabel("R^2")
                elif self.task == "classification":
                    plt.ylabel("accuracy")
                plt.xlabel("Rank(r)")
                plt.legend()
                if save:
                    plt.savefig(outfile)
                plt.show()
            else:
                plt.figure()
                sc = plt.scatter(
                    evals[1],
                    evals[0],
                    c=[i for i in range(len(evals[0]))],
                    cmap="viridis",
                    s=5,
                )
                # plt.plot(evals[1], evals[0], "o", markersize=3)
                plt.plot(
                    evals[1],
                    eval0,
                    "-",
                    markersize=1,
                    alpha=0.75,
                    color="red",
                    label="baseline",
                )
                plt.title(title)
                if self.task == "regression":
                    plt.ylabel("NMF R^2")
                    plt.xlabel("PCA R^2")
                elif self.task == "classification":
                    plt.ylabel("NMF accuracy")
                    plt.xlabel("PCA accuracy")
                plt.legend()
                plt.colorbar(sc, label="ranks")
                if save:
                    plt.savefig(outfile)
                plt.show()
        elif self.cross_validation == True:
            accuracy_std = [
                stats.norm.interval(0.95, loc=np.mean(e), scale=stats.sem(e))
                for e in evals
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
            if self.task == "regression":
                plt.ylabel("R^2")
            elif self.task == "classification":
                plt.ylabel("accuracy")
            plt.xlabel("Rank(r)")
            plt.legend()
            if save:
                plt.savefig(outfile)
            plt.show()


if __name__ == "__main__":
    df = pd.read_csv("./data/data.csv")
    # NOTE: regression
    df["stab"] = df["Stability"].map({"unstable": 0, "stable": 1}).astype(int)
    X_stab = df.select_dtypes(include=[np.number]).drop(columns=["stab"]).copy()
    y_stab = df["stab"].copy()
    rank = 73
    layers = 3
    deepparams = DeepNMFParams(
        L=layers,
        outerit=200,
        maxiter=200,
        rngseed=47,
        min_vol=False,
        rho=10,
        epsi=1e-10,
        display=False,
        accADMM=True,
        lam=[4, 2, 1],
    )
    ssnmfparams = SSNMFParam(
        model_num=6,
        lam=1e-4,
    )
    if False:
        nmf_evaluations = NMFEvaluations(
            df=df,
            X=X_stab,
            y=y_stab,
            task="classification",
            parallel=False,
            cross_validation=False,
            seed=47,
            deepparams=deepparams,
            ssnmfparams=ssnmfparams,
            layers=layers,
            evaluation_measure="xy",
        )
        ranks = [i for i in range(2, rank + 1)]
        evals2 = nmf_evaluations.evaluates(
            nmf="deep",
            init="nndsvd",
            evalutation_type="feature",
            normalize_X="minmax",
            normalize_init=None,
            algo="ALSH",
            ranks=ranks,
        )
        nmf_evaluations.affichage(
            ranks,
            evals2,
            None,
            "Classification using feature matrix \ninit=nndsvd | method=deep",
            truncate=True,
        )

    # NOTE: regression
    if True:
        X_lat = df.drop(
            ["Material", "a", "b", "Composition", "Stability", "stab"], axis=1
        ).copy()
        y_lat = df["a"].copy()
        lookup = pd.read_excel("./data/lookup.xlsx")
        rank = 71
        nmf_evaluations = NMFEvaluations(
            df=df,
            X=X_lat,
            y=y_lat,
            task="regression",
            parallel=False,
            cross_validation=False,
            seed=47,
            deepparams=deepparams,
            ssnmfparams=ssnmfparams,
            layers=layers,
            evaluation_measure="xy",
        )
        ranks = [i for i in range(2, rank + 1)]
        evals2 = nmf_evaluations.evaluates(
            nmf="deep",
            init="nndsvd",
            evalutation_type="feature",
            normalize_X="minmax",
            normalize_init=None,
            algo="MUUP",
            ranks=ranks,
        )
        nmf_evaluations.affichage(
            ranks,
            evals2,
            None,
            "Regression using feature matrix \nmethod=deep | init=nndsvd",
            truncate=True,
        )
    print(f"Runtime: {round((time.time() - p)/60, 2)}mins")
