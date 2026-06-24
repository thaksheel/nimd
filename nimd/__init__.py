from .deep.multilayer import MultilayerKLNMF
from .deep.deep import DeepNMF
from .deep.utils import DeepNMFParams, MultilayerParams
from .standard import beta_nmf, hier_nmf, fro_nmf
from .core.utils import HierarchyAbundance, SSNMFParam
from .core.selection import NMFSelection
from .supervised import SupervisedLearning
from .core.utils import quick_load_data, RunnerParams, Score, ResultData, AnalyzeResults
from .runner import Runner
from .interpretation import Interpretation
from .supervised import SupervisedLearning


__version__ = "0.1.0"

# FIXME: this way of import loads all none needed module at once! Overhead problems
__all__ = [
    "MultilayerKLNMF",
    "DeepNMF",
    "DeepNMFParams",
    "MultilayerParams",
    "Score",
    "ResultData",
    "AnalyzeResults",
    "beta_nmf",
    "hier_nmf",
    "fro_nmf",
    "SSNMFParam",
    "HierarchyAbundance",
    "NMFSelection",
    "quick_load_data",
    "Runner",
    "Interpretation",
    "SupervisedLearning",
    "RunnerParams",
    "__version__",
]
