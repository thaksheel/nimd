import torch
import numpy as np
from typing import List, Optional, Literal
from dataclasses import dataclass, field


@dataclass
class MultilayerParams:
    """
    Parameters for Multilayer NMF model.

    Attributes:
        layers_rank (np.ndarray): Number of layers in the multilayer NMF model in the format [12, 6, 3].
        maxiter (int): Iterations for initialization with multilayer NMF.
        display (bool): If True, prints progress during optimization.
        epsi (float): Numerical tolerance for convergence (suggested range: 1e-3 to 1e-10).
        beta (float): Controls divergence type for the objective. Allowed values: [0, 0.5, 1, 1.5, 2]. Some MU updates only allow [0, 1, 1.5].
        rngseed (float): Random seed for reproducibility.
        HnormType (Literal["rows", "cols"]): Normalization type for H. "rows" finds min value in each row, "cols" in each column, for MU initialization.
        normalize (Literal[1, 2, 3, 4]): Scaling method for enforcing stochasticity constraints (default: 2):
            - 1: Columns of H sum to at most 1.
            - 2: Rows of H sum to exactly 1.
            - 3: Columns of W sum to exactly 1 (required for min_vol).
            - 4: Columns of H sum to exactly 1.         
        proj: used in simplex projection can be 0, 1, 2, 3 
        init_H: use nnls to init H else use given H 
        inner_iter: iterations number for nnls fgpm step 
        alpha0: first entry in the alpha list used in nnls fgpm 
        delta: stopping criteria in the nnls fgpm step 
        accuracy: stopping criteria for convergence in beta nmf step guided by max_iter 
    """
    layers_rank: np.ndarray
    division_base: int = 2
    layers_depths: int = 3 
    maxiter: int = 100
    display: bool = False
    HnormType: Literal["rows", "cols"] = "rows"
    normalize: Literal[1, 2, 3, 4] = 2
    epsi: float = 1e-4
    beta: float = 1
    rngseed: float = 42
    display: bool = False
    accuracy: float = 1e-4
    proj: int = 0
    delta: float = 1e-6
    inner_iter: int = 500
    alpha0: float = 0.05
    init_H = None


@dataclass
class DeepNMFParams(MultilayerParams):
    """
    Parameters for Deep KL-NMF model.

    Attributes:
        layers_rank (List): Number of layers in the deep NMF model in the format [12, 6, 3].
        maxiter (int): Iterations for initialization with multilayer NMF.
        outerit (int): Iterations for deep NMF alternating optimization.
        display (bool): If True, prints progress during optimization.
        min_vol (bool): If True, activates min-volume regularization.
        epsi (float): Numerical tolerance for convergence (suggested range: 1e-3 to 1e-10).
        beta (float): Controls divergence type for the objective. Allowed values: [0, 0.5, 1, 1.5, 2]. Some MU updates only allow [0, 1, 1.5].
        rngseed (float): Random seed for reproducibility.
        HnormType (Literal["rows", "cols"]): Normalization type for H. "rows" finds min value in each row, "cols" in each column, for MU initialization.
        normalize (Literal[1, 2, 3, 4]): Scaling method for enforcing stochasticity constraints (default: 2):
            - 1: Columns of H sum to at most 1.
            - 2: Rows of H sum to exactly 1.
            - 3: Columns of W sum to exactly 1 (required for min_vol).
            - 4: Columns of H sum to exactly 1.        
        accADMM (bool): If True, uses accelerated ADMM procedure for min-vol regularization.
        maxIterADMM (int): Maximum iterations for ADMM (default: 200).
        innerloop (int): Inner loop count for step 1 of ADMM (default: 1).
        rho (int): ADMM parameter for min-vol regularization, affects Z computation (suggested range: 10-100).
        thres (float): Stopping criterion for ADMM (default: 1e-4), based on ||Wi - Zi||.
        delta (List): Used in min-vol ADMM, typically ones(1, L).
        alpha (List): Used in min-vol regularization, assigned per layer (typically zeros(1, L)).
        alpha_tilde (List): Used to compute alpha, typically 0.05 * np.ones(1, L). Example values vary by dataset.
        W0 (Optional[np.ndarray]): Initial W matrix for beta nmf init
        H0 (Optional[np.ndarray]): Initial H matrix for beta nmf init 
        Wl (Optional[List]): Initial W for each layer from multilayer nmf 
        Hl (Optional[List]): Initial H for each layer from multilayer nmf 
        lam (Optional[List]): Weight parameter for each layer (lambda). lam much match the same of layers
    """
    outerit: int = 100
    lam: Optional[List] = None
    min_vol: bool = False
    accADMM: bool = False
    maxIterADMM: int = 200
    innerloop: int = 1
    rho: int = 20
    thres: float = 1e-4
    W0: Optional[np.ndarray] = None
    H0: Optional[np.ndarray] = None
    Wl: Optional[List] = None
    Hl: Optional[List] = None
    # post-init
    delta: List = field(init=False)
    alpha: List = field(init=False)
    alpha_tilde: List = field(init=False)

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        self.alpha = np.zeros(self.layers_depths)
        self.delta = np.ones(self.layers_depths)
        self.alpha_tilde = np.ones(self.layers_depths) * 0.05 

    def set_layer_depth(self, depth): 
        self.layers_depths = depth
        self.alpha = np.zeros(depth)
        self.delta = np.ones(depth)
        self.alpha_tilde = np.ones(depth) * 0.05 
        return True 


def ensure_tensor(arr, device=None, dtype=None):
    """
    Returns a tensor if np.array.

    :param arr: either tensor or np.ndarray.
    :param device: cuda by default else cpu.
    :param dtype: float64 by default due to numerical instability. 
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float64
    if isinstance(arr, np.ndarray) or isinstance(arr, list):
        return torch.from_numpy(np.array(arr)).to(device=device, dtype=dtype)
    elif torch.is_tensor(arr):
        return arr.to(device=device, dtype=dtype)
    else:
        raise TypeError("Unsupported type for to_device")



__all__ = [
    "ensure_tensor",
    "MultilayerParams", 
    "DeepNMFParams", 
]
