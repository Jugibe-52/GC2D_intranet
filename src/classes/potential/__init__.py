"""Interpolated potentials and Hamiltonian systems built from them."""

from contracts import TrajectoryParams

from .grid import Grid
from .potential import Potential, PotentialFields, PotentialMode, Spline2D
from .potential_hamsys import PotentialHamsys
from .potential_hamsys_fc import PotentialHamsysFC
from .potential_hamsys_gc import PotentialHamsysGC
from .potential_hamsys_research import PotentialHamsysResearch, potential_hamsys_research
from .potential_research import PotentialResearch, potential_researche


def create_potential_hamsys(potential: Potential, params: TrajectoryParams) -> PotentialHamsys:
	"""Build the potential Hamiltonian system selected by ``params['type']``."""
	trajectory_type = params.get("type")
	if trajectory_type == "gc":
		return PotentialHamsysGC(potential, params)
	if trajectory_type == "fo":
		return PotentialHamsysFC(potential, params)
	raise ValueError(f"Unsupported trajectory type: {trajectory_type!r}.")


__all__ = [
	"Potential",
	"PotentialFields",
	"PotentialMode",
	"Spline2D",
	"Grid",
	"PotentialHamsys",
	"PotentialHamsysFC",
	"PotentialHamsysGC",
	"PotentialHamsysResearch",
	"PotentialResearch",
	"create_potential_hamsys",
	"potential_hamsys_research",
	"potential_researche",
]
