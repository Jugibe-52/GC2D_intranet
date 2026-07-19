"""Guiding-centre trajectories and their two-component state layout."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .trajectory import Trajectory


class GCState(NamedTuple):
	"""Named ``x`` and ``y`` blocks with matching particle/time dimensions."""

	x: np.ndarray
	y: np.ndarray


class TrajectoryGC(Trajectory):
	"""Guiding-centre state stored in component-major order ``[x, y]``."""

	state_dimension = 2

	def __init__(
		self,
		state: np.ndarray | None = None,
		*,
		rho: float = 0.0,
	) -> None:
		"""Create a guiding-centre trajectory and validate its optional state."""
		super().__init__(state, rho=rho)

	def split(self, state: np.ndarray) -> GCState:
		"""Return named position blocks without discarding trailing axes."""
		x, y = super().split(state)
		return GCState(x, y)


__all__ = ["GCState", "TrajectoryGC"]
