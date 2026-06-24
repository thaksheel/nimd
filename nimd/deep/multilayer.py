import numpy as np
import torch
from typing import Literal, List
import warnings

from .utils import ensure_tensor, MultilayerParams
from ..core.initiliazation import initialize_nmf


class MultilayerKLNMF:
    def __init__(
        self,
        X,
        params,
        device: Literal["cuda", "cpu"],
        dtype: torch.dtype,
        eps_stab: float = 1e-9,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype = dtype
        self.eps_stab = eps_stab
        self.params: MultilayerParams = params
        self.X = X

    def simplex_col_proj(self, Y):
        """
        Projects each column vector in the matrix Y onto the probability simplex.
        Torch implementation of the Wang & Carreira-Perpinan algorithm.
        """
        Y_t = ensure_tensor(Y, device=self.device, dtype=self.dtype)
        original_shape = Y_t.shape
        if Y_t.dim() == 1:
            Y_t = Y_t.view(-1, 1)

        # Y_t: (D, N), we want to process columns
        D, N = Y_t.shape
        Y_trans = Y_t.t()  # (N, D)

        X = torch.sort(Y_trans, dim=1, descending=True).values  # (N, D)
        cumsum = torch.cumsum(X, dim=1)
        k = torch.arange(1, D + 1, device=Y_t.device, dtype=Y_t.dtype).view(1, -1)
        X_tmp = (cumsum - 1) / k

        # rho: largest j s.t. X_ij > X_tmp_ij
        comp = X > X_tmp
        rho = comp.sum(dim=1) - 1  # (N,)

        lambda_vals = X_tmp[torch.arange(N, device=Y_t.device), rho]  # (N,)
        lambda_vals = lambda_vals.view(-1, 1)  # (N,1)

        X_proj_trans = torch.clamp(Y_trans - lambda_vals, min=0.0)
        X_proj = X_proj_trans.t()

        if original_shape == torch.Size([original_shape[0]]):
            return X_proj.view(-1)
        return X_proj

    def simplex_proj(self, Y):
        """
        Projects columns of Y onto the L1-ball {x | x >= 0, sum(x) <= 1}.
        Torch implementation.
        """
        Y_t = ensure_tensor(Y, device=self.device, dtype=self.dtype)
        Y_pos = torch.clamp(Y_t, min=0.0)
        col_sums = Y_pos.sum(dim=0)
        idx_to_project = col_sums > 1.0
        if idx_to_project.any():
            Y_pos[:, idx_to_project] = self.simplex_col_proj(Y_pos[:, idx_to_project])
        return Y_pos

    def orth_nnls(self, M, U, Mn=None):
        """
        Torch version of orth_nnls.
        """
        M_t = ensure_tensor(M, device=self.device, dtype=self.dtype)
        U_t = ensure_tensor(U, device=self.device, dtype=self.dtype)

        if Mn is None:
            norm2m = torch.linalg.norm(M_t, dim=0)
            Mn_t = M_t / (norm2m + 1e-16)
        else:
            Mn_t = ensure_tensor(Mn, device=self.device, dtype=self.dtype)

        m, n = Mn_t.shape
        m_u, r = U_t.shape

        norm2u = torch.linalg.norm(U_t, dim=0)
        Un_t = U_t / (norm2u + 1e-16)

        A = Mn_t.t() @ Un_t  # (n, r)
        best_u_indices = torch.argmax(A, dim=1)  # (n,)

        V = torch.zeros(r, n, device=M_t.device, dtype=M_t.dtype)
        U_best = U_t[:, best_u_indices]  # (m, n)
        norm2u_best_sq = norm2u[best_u_indices] ** 2  # (n,)

        # weights = (M .* U_best) columnwise / norm2u_best_sq
        weights = torch.sum(M_t * U_best, dim=0) / (norm2u_best_sq + 1e-16)  # (n,)

        V[best_u_indices, torch.arange(n, device=M_t.device)] = weights
        return V

    def nnls_init(self, X, W, WtW, WtX):
        """
        Torch version of nnls_init.
        """
        X_t = ensure_tensor(X, device=self.device, dtype=self.dtype)
        W_t = ensure_tensor(W, device=self.device, dtype=self.dtype)
        WtW_t = ensure_tensor(WtW, device=self.device, dtype=self.dtype)
        WtX_t = ensure_tensor(WtX, device=self.device, dtype=self.dtype)

        cond_W = torch.linalg.cond(W_t)
        if cond_W > 1e6:
            H = self.orth_nnls(X_t, W_t)
        else:
            sol = torch.linalg.lstsq(W_t, X_t)
            H = torch.clamp(sol.solution, min=0.0)
            numerator = torch.sum(H * WtX_t)
            denominator = torch.sum(WtW_t * (H @ H.t()))
            if denominator > self.eps_stab:
                alpha = numerator / denominator
                H = H * alpha
        row_sums = H.sum(dim=1)
        zero_rows = torch.where(row_sums == 0)[0]
        if len(zero_rows) > 0:
            max_h_val = H.max()
            if max_h_val <= 0:
                max_h_val = torch.tensor(1.0, device=H.device, dtype=H.dtype)
            n_cols = H.shape[1]
            rand_vals = (
                0.001
                * max_h_val
                * torch.rand(
                    (len(zero_rows), n_cols),
                    device=H.device,
                    dtype=H.dtype,
                )
            )
            H[zero_rows, :] = rand_vals

        return H

    def updatemu_hcols(self, C, D, H, mu, epsi: float, eps_stab):
        """
        Torch version of updatemu_hcols.
        """
        C = ensure_tensor(C, device=self.device, dtype=self.dtype)
        D = ensure_tensor(D, device=self.device, dtype=self.dtype)
        H = ensure_tensor(H, device=self.device, dtype=self.dtype)
        mu = ensure_tensor(mu, device=self.device, dtype=self.dtype)

        K, T = C.shape
        JK1 = torch.ones((K, 1), device=self.device, dtype=self.dtype)
        delta = 1.0
        do_loop = True
        max_iter_mu = 10**4
        k = 0
        while do_loop and k < max_iter_mu:
            mu_prev = mu.clone()
            Mat = H * (C / (D - JK1 @ mu.T + eps_stab))
            xi = (torch.sum(Mat, dim=0) - delta).unsqueeze(1)
            Matp = H * C / (D - JK1 @ mu.T + eps_stab) ** 2
            xip = torch.sum(Matp, dim=0).unsqueeze(1)
            mu = mu - xi / xip
            if torch.max(torch.abs(mu - mu_prev)) <= epsi:
                do_loop = False
            k += 1
        if k == max_iter_mu:
            warnings.warn(
                "updatemu_hcols: Newton-Raphson reached max iterations. Change ephis for better results."
            )
        return mu

    def nnls_fpgm(self, X, W, inneriter: int):
        """
        Torch version of nnls_fpgm.
        """
        X_t = ensure_tensor(X, device=self.device, dtype=self.dtype)
        W_t = ensure_tensor(W, device=self.device, dtype=self.dtype)
        WtW = W_t.t() @ W_t
        WtX = W_t.t() @ X_t

        if self.params.init_H is None:
            H = self.nnls_init(X_t, W_t, WtW, WtX)
        else:
            H = ensure_tensor(self.params.init_H, device=self.device, dtype=self.dtype)

        L = torch.linalg.norm(WtW, 2)  # Lipschitz constant

        alpha = [self.params.alpha0]
        beta_list = []

        if self.params.proj == 1:
            H = self.simplex_proj(H)
        elif self.params.proj == 0:
            H = torch.clamp(H, min=0.0)
        elif self.params.proj == 2:
            H = self.simplex_col_proj(H.t()).t()
        elif self.params.proj == 3:
            H = self.simplex_col_proj(H)

        Y = H.clone()
        i = 0
        eps0 = 0.0
        eps = 1.0

        while i < inneriter and (eps >= self.params.delta * eps0 if eps0 > 0 else True):
            Hp = H.clone()

            alpha_i = alpha[i]
            alpha_i_plus_1 = (torch.sqrt(alpha_i**4 + 4 * alpha_i**2) - alpha_i**2) / 2
            alpha.append(alpha_i_plus_1.item())
            beta_val = alpha_i * (1 - alpha_i) / (alpha_i**2 + alpha_i_plus_1)
            beta_list.append(beta_val)

            # Gradient step
            H = Y - (WtW @ Y - WtX) / L

            if self.params.proj == 1:
                H = self.simplex_proj(H)
            elif self.params.proj == 0:
                H = torch.clamp(H, min=0.0)
            elif self.params.proj == 2:
                H = self.simplex_col_proj(H.t()).t()
            elif self.params.proj == 3:
                H = self.simplex_col_proj(H)

            # Extrapolation
            Y = H + beta_val * (H - Hp)
            if i == 0:
                eps0 = torch.linalg.norm(H - Hp, "fro").item()
                if eps0 == 0:
                    break
            eps = torch.linalg.norm(H - Hp, "fro").item()
            i += 1

        return H, WtW, WtX

    def normalize_wh(
        self,
        W,
        H,
        sumtoone: Literal[1, 2, 3, 4],
        X=None,
    ):
        """
        Torch version of normalize_wh (dense only).
        """
        W_t = ensure_tensor(W, device=self.device, dtype=self.dtype)
        H_t = ensure_tensor(H, device=self.device, dtype=self.dtype)
        X_t = (
            ensure_tensor(X, device=self.device, dtype=self.dtype)
            if X is not None
            else None
        )

        if sumtoone == 1:  # Columns of H sum to at most 1
            Hn = self.simplex_proj(H_t)
            if torch.linalg.norm(Hn - H_t) > 1e-3 * torch.linalg.norm(Hn):
                H_t = Hn
                self.params.init_H = W_t.t()
                if X_t is not None:
                    W_new, _, _ = self.nnls_fpgm(X_t.t(), H_t.t(), inneriter=100)
                else:
                    raise ValueError(
                        f"X not given for normalize_wh sumtoone_type={sumtoone}"
                    )
                W_t = W_new.t()
            H_t = Hn

        elif sumtoone == 2:  # Rows of H sum to 1
            scalH = H_t.sum(dim=1)
            scalH[scalH == 0] = 1.0
            inv_scalH = 1.0 / scalH
            H_t = torch.diag(inv_scalH) @ H_t
            W_t = W_t @ torch.diag(scalH)

        elif sumtoone == 3:  # Columns of W sum to 1
            scalW = W_t.sum(dim=0)
            scalW[scalW == 0] = 1.0
            H_t = torch.diag(scalW) @ H_t
            W_t = W_t @ torch.diag(1.0 / scalW)

        elif sumtoone == 4:  # Columns of H sum to 1
            Hn = self.simplex_col_proj(H_t)
            if torch.linalg.norm(Hn - H_t) > 1e-3 * torch.linalg.norm(Hn):
                H_t = Hn
                self.params.init_H = W_t.t()
                if X_t is not None:
                    W_new, _, _ = self.nnls_fpgm(X_t.t(), H_t.t(), inneriter=100)
                else:
                    raise ValueError(
                        f"X not given for normalize_wh sumtoone_type={sumtoone}"
                    )
                W_t = W_new.t()
            H_t = Hn

        return W_t, H_t

    def betadiv(self, X, Y, beta: float):
        """
        Torch version of betadiv (dense only).
        """
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        Y = ensure_tensor(Y, device=self.device, dtype=self.dtype)
        if beta == 0:  # Itakura-Saito
            ratio = X / (Y + self.eps_stab)
            Z = ratio - torch.log(ratio + self.eps_stab) - 1.0
        elif beta == 1:  # KL
            Z = X * torch.log(X / (Y + self.eps_stab) + self.eps_stab) - X + Y
        else:
            Y_beta = Y**beta
            Y_beta_1 = Y ** (beta - 1)
            Z = (X**beta + (beta - 1) * Y_beta - beta * X * Y_beta_1) / (
                beta * (beta - 1) + self.eps_stab
            )
        return torch.sum(Z)

    def nd_mubeta(self, X, W, H, beta: float):
        """
        Torch version of nd_mubeta (dense only).
        Returns N, D, error.
        """
        X_t = ensure_tensor(X, device=self.device, dtype=self.dtype)
        W_t = ensure_tensor(W, device=self.device, dtype=self.dtype)
        H_t = ensure_tensor(H, device=self.device, dtype=self.dtype)

        WH = W_t @ H_t
        e = None

        if beta == 1:
            XdWH = X_t / (WH + self.eps_stab)
            N = W_t.t() @ XdWH
            D = torch.sum(W_t, dim=0).view(-1, 1)
            e = self.betadiv(X_t, WH, beta)

        elif beta == 2:
            N = W_t.t() @ X_t
            D = (W_t.t() @ W_t) @ H_t
            e = 0.5 * torch.linalg.norm(X_t - WH, "fro") ** 2

        else:
            WH_beta_2 = WH ** (beta - 2)
            WH_beta_1 = WH ** (beta - 1)
            N = W_t.t() @ (WH_beta_2 * X_t)
            D = W_t.t() @ WH_beta_1
            e = self.betadiv(X_t, WH, beta)

        return N, D, e

    def mu_beta(self, X, W, H, beta: float):
        """
        Torch version of mu_beta.
        """
        X_t = ensure_tensor(X, device=self.device, dtype=self.dtype)
        W_t = ensure_tensor(W, device=self.device, dtype=self.dtype)
        H_t = ensure_tensor(H, device=self.device, dtype=self.dtype)

        N, D, e = self.nd_mubeta(X_t, W_t, H_t, beta)
        if 1 <= beta <= 2:
            H_t = H_t * (N / (D + self.eps_stab))
        else:
            if beta < 1:
                gamma = 1.0 / (2.0 - beta)
            else:
                gamma = 1.0 / (beta - 1.0)
            H_t = H_t * ((N / (D + self.eps_stab)) ** gamma)

        H_t = torch.clamp(H_t, min=self.eps_stab)
        return H_t, e

    def beta_nmf(
        self,
        X,
        r: int,
        init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"],
        perturb: bool = False,
        nl: float = None,
    ):
        """
        Torch version of beta_nmf. Dense, beta-divergence NMF.
        """
        X_t = ensure_tensor(X, device=self.device, dtype=self.dtype)
        W, H = initialize_nmf(
            X_t.cpu().numpy(),
            r,
            init,
            eps=self.eps_stab,
            random_state=self.params.rngseed,
            perturb=perturb,
            noise_level=nl,
        )
        W = ensure_tensor(W, device=self.device, dtype=self.dtype)
        H = ensure_tensor(H, device=self.device, dtype=self.dtype)
        if self.params.rngseed is not None:
            np.random.seed(self.params.rngseed)
            torch.manual_seed(self.params.rngseed)
        e_list: List[float] = []
        i = 0
        while i < self.params.maxiter:
            H, _ = self.mu_beta(X_t, W, H, self.params.beta)
            W_T, ei = self.mu_beta(X_t.t(), H.t(), W.t(), self.params.beta)
            W = W_T.t()
            e_list.append(ei.item() if torch.is_tensor(ei) else float(ei))
            if i >= 11 and abs(e_list[-1] - e_list[-11]) < self.params.accuracy * abs(
                e_list[-10]
            ):
                if self.params.display:
                    print("\nConvergence reached.")
                break
            # Scaling: max entry in each column of W is 1
            for k in range(r):
                mxk = torch.max(W[:, k])
                if mxk > self.eps_stab:
                    W[:, k] = W[:, k] / mxk
                    H[k, :] = H[k, :] * mxk
            if self.params.display:
                if (i + 1) % 10 == 0:
                    print(f"{i+1:3d}...", end="", flush=True)
                if (i + 1) % 100 == 0:
                    print()
            i += 1
        if self.params.display:
            print("\n")

        return W, H, e_list

    def run(
        self,
        init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"],
        perturb: bool = False,
        nl: float = None,
    ):
        X_t = ensure_tensor(self.X, device=self.device, dtype=self.dtype)
        layers = len(self.params.layers_rank)
        if self.params.rngseed is not None:
            np.random.seed(self.params.rngseed)
            torch.manual_seed(self.params.rngseed)

        W_list = [None] * layers
        H_list = [None] * layers
        e = torch.zeros(layers, device=self.device, dtype=self.dtype)

        for i in range(layers):
            current_X = X_t if i == 0 else W_list[i - 1]
            if current_X is None:
                raise ValueError("current_X became None in multilayer_klnmf")

            r = self.params.layers_rank[i]
            if self.params.HnormType == "rows":
                W_i, H_i, ei = self.beta_nmf(current_X, r, init, perturb, nl)
                W_i, H_i = self.normalize_wh(
                    W_i,
                    H_i,
                    self.params.normalize,
                    current_X,
                )
                if len(ei) > 0:
                    e[i] = torch.tensor(ei[-1], device=self.device, dtype=self.dtype)
                else:
                    e[i] = self.betadiv(current_X, W_i @ H_i, self.params.beta)
                W_list[i], H_list[i] = W_i, H_i

            elif self.params.HnormType == "cols":
                m, n = current_X.shape
                W_i, H_i = initialize_nmf(
                    current_X,
                    r,
                    init,
                    eps=self.eps_stab,
                    random_state=self.params.rngseed,
                    device=self.device,
                    perturb=perturb,
                    nl=nl,
                )
                for k in range(self.params.maxiter):
                    prod = W_i @ H_i + self.eps_stab
                    Wt = W_i.t()
                    C = Wt @ (prod ** (self.params.beta - 2) * current_X)
                    D = Wt @ (prod ** (self.params.beta - 1))

                    I = torch.argmin(D, dim=0)
                    idx = (I, torch.arange(n, device=self.device))
                    mu_0_H = (D[idx] - C[idx] * H_i[idx]).view(-1, 1)
                    mu_H = self.updatemu_hcols(
                        C=C,
                        D=D,
                        H=H_i,
                        mu=mu_0_H,
                        epsi=self.params.epsi,
                        eps_stab=self.eps_stab,
                    )

                    H_i = H_i * (C / (D - mu_H.t() + self.eps_stab))
                    H_i = torch.clamp(H_i, min=self.eps_stab)
                    W_T_i, _ = self.mu_beta(
                        current_X.t(), H_i.t(), W_i.t(), self.params.beta
                    )
                    W_i = W_T_i.t()
                    if self.params.display:
                        if (k + 1) % 10 == 0:
                            print(f"{k+1:3d}...", end="", flush=True)
                        if (k + 1) % 100 == 0:
                            print()
                if self.params.display and self.params.maxiter > 0:
                    print()
                W_list[i], H_list[i] = W_i, H_i
                e[i] = self.betadiv(current_X, W_i @ H_i, self.params.beta)
            else:
                raise ValueError(f"Unknown HnormType: {self.params.HnormType}")
            if self.params.display:
                print(f"Layer {i + 1} done.")

        return W_list, H_list, e
