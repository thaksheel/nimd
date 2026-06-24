from typing import Literal, Optional
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..standard import beta_nmf, fro_nmf, hier_nmf, nnls, ss_nmf
from .initiliazation import initialize_nmf
from .utils import SSNMFParam

from ..deep.utils import DeepNMFParams, MultilayerParams
from ..deep.deep import DeepNMF
from ..deep.multilayer import MultilayerKLNMF


class NMFSelection:

    def __init__(
        self,
        X: np.ndarray,
        init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"],
        rank,
        eval_type: Literal["feature", "full"],
        norm_X: Optional[Literal["minmax"]],
        norm_init: Optional[Literal["scaling", "feature_norm"]],
        rng: int = 42,
        perturb: bool = False,
        nl: float = None,
    ):
        self.X = X
        self.init = init
        self.rank = rank
        self.eval_type = eval_type
        self.norm_X = norm_X
        self.norm_init = norm_init
        self.rng = rng
        self.perturb = perturb
        self.nl = nl

        # null fields
        self.W0 = None
        self.H0 = None

    def _preprocess(self, X, rank):
        # TODO: add standard scaler and more details on preprocessing
        if self.norm_X:
            X = self.normalize(normalize=self.norm_X)
        self.W0, self.H0 = initialize_nmf(
            X,
            n_components=rank,
            init=self.init,
            random_state=self.rng,
            perturb=self.perturb,
            noise_level=self.nl,
        )
        if self.norm_init == "feature_norm" or self.norm_init == "scaling":
            self.W0, self.H0 = self.normalize(normalize=self.norm_init)
        return X, self.W0, self.H0

    def update_rank(self, rank):
        self.rank = rank
        self.W0, self.H0 = initialize_nmf(
            self.X, n_components=rank, init=self.init, random_state=self.rng
        )
        if self.norm_X == "minmax":
            self.X = self.normalize(normalize=self.norm_X)
        elif self.norm_init == "feature_norm" or self.norm_init == "scaling":
            self.W0, self.H0 = self.normalize(normalize=self.norm_init)

        return self

    def factorize_standard(
        self,
        rank: int,
        model: Literal["fronorm", "beta", "hier", "pca"],
        fronorm_algo: Literal["MUUP", "ADMM", "HALS", "FPGM", "ALSH"],
    ) -> tuple[np.ndarray, np.ndarray]:
        if model == "beta":
            return self.beta_nmf(rank)
        elif model == "fronorm":
            return self.fronorm_nmf(algo=fronorm_algo, rank=rank)
        elif model == "hier":
            return self.hier_nmf(rank)
        elif model == "pca":
            return self.pca(rank)
        else:
            raise ValueError("self.model out of expected list for standard nmf")

    def factorize_deep(
        self,
        rank: int,
        model: Literal["deep", "multilayer"],
        deep_params: DeepNMFParams,
        mul_params: MultilayerParams,
        device,
        dtype,
        eps_stab=1e-4,
        return_tensor: bool = False,
    ) -> tuple:
        X_, W, H = self._preprocess(self.X, rank)
        deep_params.H0 = H
        deep_params.W0 = W
        if model == "deep":
            Wl, Hl, output = self.deep_nmf(
                X=X_,
                params=deep_params,
                mlparams=mul_params,
                device=device,
                dtype=dtype,
                eps_stab=eps_stab,
            )
        elif model == "multilayer":
            Wl, Hl, output = self.multilayer_nmf(
                X=X_,
                params=mul_params,
                device=device,
                dtype=dtype,
                eps_stab=eps_stab,
            )
        else:
            raise ValueError("self.model out of expected list for deep nmf")
        W, H = self._compress_layers(Wl, Hl)
        if return_tensor:
            return W, H, Wl, Hl, output
        else:
            Wl = [w.cpu().numpy() for w in Wl]
            Hl = [w.cpu().numpy() for w in Hl]
            return W.cpu().numpy(), H.cpu().numpy(), Wl, Hl, output

    def fronorm_nmf(
        self,
        rank: int,
        algo: Literal["MUUP", "ADMM", "HALS", "FPGM", "ALSH"],
    ):
        X_, W0, H0 = self._preprocess(self.X, rank)
        options = {
            "init": {"W": W0, "H": H0},
        }
        W, H, e, t, etrue, i, options = fro_nmf(
            X=X_,
            r=self.rank,
            algo=algo,
            options=options,
            W0=self.W0,
            H0=self.H0,
        )
        return W, H, e

    def beta_nmf(
        self,
        rank: int,
    ):
        X_, W0, H0 = self._preprocess(self.X, rank)
        options = {
            "W": W0,
            "H": H0,
        }
        W, H, e, t = beta_nmf(X_, self.rank, options=options)
        return W, H, e

    def hier_nmf(
        self,
        rank: int,
    ):
        X_, W0, H0 = self._preprocess(self.X, rank)
        # FIXME: issues with ranks
        if rank > 68:
            rank = 68
        idx, W, _ = hier_nmf(self.X, rank)
        H, *_ = nnls(W=W, X=X_)
        return W, H, None

    def evaluate_ssnmf(
        self,
        y: np.ndarray,
        rank: int,
        params: SSNMFParam,
        task: Literal["regression", "classification"],
    ):
        X_, W0, H0 = self._preprocess(self.X, rank)
        X_train, X_test, y_train, y_test = ss_nmf(
            X=X_,
            y=y,
            W0=W0,
            H0=H0,
            init=self.init,
            rank=rank,
            params=params,
            task=task,
            rng=self.rng,
        )
        return X_train, X_test, y_train, y_test

    def pca(self, rank: int):
        X_, W0, H0 = self._preprocess(self.X, rank)
        pca = PCA(n_components=rank)
        W = pca.fit_transform(X_)
        H = np.ones(self.H0.shape)
        return W, H, None

    def _compress_layers(self, Wl, Hl):
        # TODO: need to review the paper and improve this
        W = Wl[-1]
        for i, H in enumerate(Hl[::-1]):
            if i == len(Hl) - 1:
                break
            W = W @ H
        H = Hl[0]
        return Wl[0], Hl[0]

    def deep_nmf(
        self,
        X: np.ndarray,
        params: DeepNMFParams,
        mlparams: MultilayerParams,
        device: Literal["cuda", "cpu"],
        dtype=torch.float64,
        eps_stab=1e-4,
    ):
        mlnmf = MultilayerKLNMF(
            X=X, params=mlparams, device=device, dtype=dtype, eps_stab=eps_stab
        )
        deep = DeepNMF(
            X=X,
            params=params,
            device=device,
            dtype=dtype,
            multilayer_nmf=mlnmf,
            eps_stab=eps_stab,
        )
        Wl, Hl, output = deep.run(self.init, self.perturb, self.nl)
        return Wl, Hl, output

    def multilayer_nmf(
        self,
        X: np.ndarray,
        params: MultilayerParams,
        device: Literal["cuda", "cpu"],
        dtype=torch.float64,
        eps_stab=1e-4,
    ):
        multilayer = MultilayerKLNMF(
            X=X, params=params, device=device, dtype=dtype, eps_stab=eps_stab
        )
        Wl, Hl, output = multilayer.run(init=self.init)
        return Wl, Hl, output

    def normalize(
        self, normalize: Literal["minmax", "standard", "scaling", "feature_norm"]
    ):
        if normalize == "minmax":
            mm = MinMaxScaler()
            X_mm = mm.fit_transform(self.X)
            # col_min = self.X.min(axis=0)
            # col_max = self.X.max(axis=0)
            # Xnorm = (self.X - col_min) / (col_max - col_min)
            return X_mm
        elif normalize == "standard":
            mm = StandardScaler()
            X_mm = mm.fit_transform(self.X)
            return X_mm
        elif normalize == "feature_norm":
            W = self.W0.copy()
            H = self.H0.copy()
            normW = np.sqrt(np.sum(W**2, axis=0)) + 1e-16
            normH = np.sqrt(np.sum(H**2, axis=1)) + 1e-16
            d = np.sqrt(normW) / np.sqrt(normH)
            H *= d[:, None]
            for k in range(W.shape[1]):
                W[:, k] = W[:, k] / np.sqrt(normW[k]) * np.sqrt(normH[k])
            return W, H
        elif normalize == "scaling":
            # Scale initialization so that argmin_a ||a * WH - X||_F = 1
            W = self.W0.copy()
            H = self.H0.copy()
            XHt = self.X @ H.T
            HHt = H @ H.T
            scaling = np.sum(XHt * W) / np.sum(HHt * (W.T @ W))
            W *= scaling
            return W, H
