"""Shared initial-state storage and physical-component block layouts."""

from __future__ import annotations

from typing import cast

from abc import ABC, abstractmethod

import numpy as np


from contracts.state_layout import PackedStateLayout


class StateConfiguration(ABC):
	"""Own an optional initial state and delegate its interpretation to a layout.

	The abstract :attr:`layout` property prevents this storage base from being
	instantiated without a concrete state representation. Physical model
	parameters remain owned by the dynamical system bound in an initial-value
	problem.
	"""

	def __init__(
		self,
		state: np.ndarray | None = None,
	) -> None:
		"""Create a configuration with an optional flat physical state."""
		self._state: np.ndarray | None = None
		if state is not None:
			self.set_initial_state(state)

	@property
	@abstractmethod
	def layout(self) -> PackedStateLayout:
		"""Return the concrete component-major state layout."""

	@property
	def state(self) -> np.ndarray | None:
		"""Return a copy so callers cannot mutate the stored initial condition."""
		return None if self._state is None else self._state.copy()

	@property
	def initial_state(self) -> np.ndarray | None:
		"""Return an independent copy of the stored initial physical state."""
		return self.state

	def set_initial_state(self, state: np.ndarray) -> None:
		"""Validate and store a one-dimensional component-major state."""
		value = np.asarray(state, dtype=float)
		if value.ndim != 1 or value.size == 0:
			raise ValueError("The initial state must be a non-empty one-dimensional array.")
		if not np.all(np.isfinite(value)):
			raise ValueError("The initial state must contain only finite values.")
		self.layout.validate_packed_state_layout(value)
		self._state = value.copy()

	# Compatibility accessors keep existing notebooks working while the
	# simulation core consumes ``configuration.layout`` directly.
	@property
	def state_dimension(self) -> int:
		"""Return the delegated component count."""
		return self.layout.state_dimension

	def validate_packed_state_layout(self, state: np.ndarray) -> np.ndarray:
		"""Delegate packed-layout validation to :attr:`layout`."""
		return cast(np.ndarray, self.layout.validate_packed_state_layout(state))

	def split(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
		"""Delegate component splitting to :attr:`layout`."""
		return self.layout.split(state)

	def as_blocks(self, state: np.ndarray) -> np.ndarray:
		"""Delegate explicit block exposure to :attr:`layout`."""
		return cast(np.ndarray, self.layout.as_blocks(state))

	def from_blocks(self, blocks: np.ndarray) -> np.ndarray:
		"""Delegate block flattening to :attr:`layout`."""
		return cast(np.ndarray, self.layout.from_blocks(blocks))

	def particle_count(self, state: np.ndarray) -> int:
		"""Delegate particle counting to :attr:`layout`."""
		return self.layout.particle_count(state)



__all__ = [
	"PackedStateLayout",
	"StateConfiguration",
]
