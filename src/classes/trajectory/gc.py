"""Guiding-centre trajectories and their two-component state layout."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .trajectory import Trajectory


class GCState(NamedTuple):
	"""Guiding-centre coordinates with matching particle/sample dimensions.

	Each field has shape ``(N, *sample_axes)``: axis zero identifies the
	particle (or contour vertex) and trailing axes identify solution samples.
	"""

	x: np.ndarray
	y: np.ndarray


class TrajectoryGC(Trajectory):
	"""Guiding-centre positions stored in component-major order ``[x, y]``.

	For ``N`` particles, the flat initial state is
	``[x_1, ..., x_N, y_1, ..., y_N]``.  No velocity block is needed because the
	GC equations determine coordinate rates directly from the effective field.
	"""

	state_dimension = 2

	def __init__(
		self,
		state: np.ndarray | None = None,
		*,
		rho: float = 0.0,
	) -> None:
		"""Create a GC trajectory with normalized Larmor radius ``rho``."""
		super().__init__(state, rho=rho)

	def split(self, state: np.ndarray) -> GCState:
		"""Return named ``x`` and ``y`` blocks, preserving sample axes."""
		x, y = super().split(state)
		return GCState(x, y)


__all__ = ["GCState", "TrajectoryGC"]
