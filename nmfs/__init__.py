"""Compatibility alias for legacy NMF modules.

The maintained package location is ``nimd.legacy.nmfs``. This namespace keeps
older imports such as ``import nmfs.NNLS`` available.
"""

from nimd.legacy import nmfs as _legacy_nmfs

__path__ = _legacy_nmfs.__path__
