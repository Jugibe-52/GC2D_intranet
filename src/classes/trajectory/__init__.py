"""Trajectory models for interpolated two-dimensional potentials."""

from contracts import TrajectoryParams

from ..potential import Potential
from .trajectory import Trajectory
from .trajectory_fc import TrajectoryFC
from .trajectory_gc import TrajectoryGC


def create_trajectory(potential: Potential, params: TrajectoryParams) -> Trajectory:
	"""Build the trajectory model selected by ``params['type']``."""
	trajectory_type = params.get("type")
	if trajectory_type == "gc":
		return TrajectoryGC(potential, params)
	if trajectory_type == "fo":
		return TrajectoryFC(potential, params)
	raise ValueError(f"Unsupported trajectory type: {trajectory_type!r}.")


__all__ = ["Trajectory", "TrajectoryFC", "TrajectoryGC", "create_trajectory"]
