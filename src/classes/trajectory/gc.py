"""Guiding-centre trajectory."""

from .trajectory import Trajectory


class TrajectoryGC(Trajectory):
	"""Guiding-centre state stored as the blocks ``[x, y]``."""

	state_dimension = 2


__all__ = ["TrajectoryGC"]
