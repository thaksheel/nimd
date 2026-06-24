"""Compatibility alias for the renamed :mod:`nimd` package.

New code should import from ``nimd``. This module keeps older notebooks and
scripts working while the project moves away from the generic ``src`` name.
"""

from nimd import *  # noqa: F401,F403

import nimd as _nimd

__path__ = _nimd.__path__
__all__ = getattr(_nimd, "__all__", [])
