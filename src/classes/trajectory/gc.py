"""Guiding-centre trajectory entity."""

from __future__ import annotations

import numpy as np

from contracts import Array

from .trajectory import Trajectory


class TrajectoryGC(Trajectory):
	"""Guiding-centre state layout ``[x, y]``."""

	kind = "gc"
	state_dimension = 2
	degrees_of_freedom = 1

	def get_velocities(self, state: Array) -> None:
		self.split_state(state)
		return None

	def _complete_initial_state(
		self,
		x: Array,
		y: Array,
		rng: np.random.Generator | np.random.RandomState,
	) -> Array:
		del rng
		return np.concatenate((x, y))


__all__ = ["TrajectoryGC"]
