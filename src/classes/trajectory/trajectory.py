"""Base trajectory state and block-layout rules."""

from __future__ import annotations

from typing import ClassVar

import numpy as np


class Trajectory:
	"""Particle parameters and an optional initial state."""

	state_dimension: ClassVar[int]

	def __init__(
		self,
		state: np.ndarray | None = None,
		*,
		rho: float = 0.0,
	) -> None:
		rho = float(rho)
		if not np.isfinite(rho) or rho < 0:
			raise ValueError("`rho` must be finite and non-negative.")
		self.rho = rho
		self._state: np.ndarray | None = None
		if state is not None:
			self.set_initial_state(state)

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
		if value.ndim == 0 or value.shape[0] == 0 or value.shape[0] % self.state_dimension:
			raise ValueError(
				f"The first state dimension must be divisible by {self.state_dimension} "
				f"for {self.__class__.__name__}."
			)
		return tuple(np.split(value, self.state_dimension, axis=0))

	def pack(self, *components: np.ndarray) -> np.ndarray:
		"""Pack equally shaped physical components into block layout."""
		if len(components) != self.state_dimension:
			raise ValueError(
				f"{self.__class__.__name__} requires {self.state_dimension} components."
			)
		values = tuple(np.asarray(component) for component in components)
		if not values or values[0].ndim == 0 or values[0].shape[0] == 0:
			raise ValueError("State components must be non-empty arrays.")
		if any(value.shape != values[0].shape for value in values[1:]):
			raise ValueError("All state components must have the same shape.")
		return np.concatenate(values, axis=0)

	def particle_count(self, state: np.ndarray) -> int:
		"""Return the number of particles represented by a state."""
		value = np.asarray(state)
		self.split(value)
		return int(value.shape[0] // self.state_dimension)

	def positions(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Return the ``x`` and ``y`` blocks of a state or solution."""
		x, y, *_ = self.split(state)
		return x, y

__all__ = ["Trajectory"]
