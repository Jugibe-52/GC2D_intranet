"""Guiding-centre trajectory."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .trajectory import Trajectory


class GCState(NamedTuple):
	"""Named components of a guiding-centre state."""

	x: np.ndarray
	y: np.ndarray


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

	def split(self, state: np.ndarray) -> GCState:
		"""Return the named ``x`` and ``y`` blocks."""
		x, y = super().split(state)
		return GCState(x, y)


__all__ = ["GCState", "TrajectoryGC"]
