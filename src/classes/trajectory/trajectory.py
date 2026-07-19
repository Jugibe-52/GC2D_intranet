"""Base trajectory state and block-layout rules."""

from __future__ import annotations

from typing import ClassVar

import numpy as np


class Trajectory:
	"""Particle parameters and an explicitly assigned initial state."""

	state_dimension: ClassVar[int]

	def __init__(self, *, rho: float = 0.0) -> None:
		rho = float(rho)
		if not np.isfinite(rho) or rho < 0:
			raise ValueError("`rho` must be finite and non-negative.")
		self.rho = rho
		self._state: np.ndarray | None = None

	@property
	def state(self) -> np.ndarray | None:
		"""Return a defensive copy of the assigned initial state."""
		return None if self._state is None else self._state.copy()

	def set_initial_state(self, state: np.ndarray) -> None:
		"""Validate and store a one-dimensional block-layout state."""
		value = np.asarray(state, dtype=float)
		if value.ndim != 1 or value.size == 0:
			raise ValueError("The initial state must be a non-empty one-dimensional array.")
		if not np.all(np.isfinite(value)):
			raise ValueError("The initial state must contain only finite values.")
		self.split(value)
		self._state = value.copy()

	def split(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
		"""Split ``[component_1, ..., component_n]`` into equal blocks."""
		value = np.asarray(state)
		if value.ndim == 0 or value.shape[0] % self.state_dimension:
			raise ValueError(
				f"The first state dimension must be divisible by {self.state_dimension} "
				f"for {self.__class__.__name__}."
			)
		return tuple(np.split(value, self.state_dimension, axis=0))

	def positions(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Return the ``x`` and ``y`` blocks of a state or solution."""
		x, y, *_ = self.split(state)
		return x, y

__all__ = ["Trajectory"]
