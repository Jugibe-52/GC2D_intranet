"""Guiding-centre trajectory."""

from __future__ import annotations

import numpy as np

from .trajectory import Trajectory


class TrajectoryGC(Trajectory):
	"""Guiding-centre state stored as the blocks ``[x, y]``."""

	state_dimension = 2

	def __init__(
		self,
		state: np.ndarray | None = None,
		*,
		rho: float = 0.0,
	) -> None:
		super().__init__(state, rho=rho)


__all__ = ["TrajectoryGC"]
