"""Public interface for electrostatic potentials and GC2D HDF5 loading."""

from .gc2d_h5 import GC2DH5Potential, load_gc2d_h5_potential
from .potential import Potential

__all__ = ["GC2DH5Potential", "Potential", "load_gc2d_h5_potential"]
