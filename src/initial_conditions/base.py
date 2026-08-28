"""Shared initial-state storage and physical-component block layouts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class PackedStateLayout:
	"""Reusable operations for one component-major state layout.

	``state_dimension`` is the number of scalar components stored per particle;
	concrete layouts define the physical meaning and order of those blocks.
	For ``N`` particles, a state has shape
	``(state_dimension * N, *sample_axes)`` and each component returned by
	:meth:`split` has shape ``(N, *sample_axes)``.  The optional ``sample_axes``
	usually represent saved integration times and are never part of the particle
	count.
	"""

	__slots__ = ()

	# Number of component blocks on the leading state axis.  Subclasses provide
	# the value because GC and FC states carry different physical variables.
	state_dimension: ClassVar[int]

	def split(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
		"""Split the leading axis into equally sized physical components.

		A single state has shape ``(state_dimension * N,)``.  A solution with
		shape ``(state_dimension * N, *sample_axes)`` is split along axis zero,
		leaving every component with shape ``(N, *sample_axes)``.
		"""
		blocks = self.as_blocks(state)
		return tuple(blocks[index] for index in range(self.state_dimension))

	def as_blocks(self, state: np.ndarray) -> np.ndarray:
		"""Expose a packed state as ``(components, particles, *samples)``.

		The returned array is a reshape view whenever NumPy can preserve the input
		memory layout.  Making both physical axes explicit avoids repeating manual
		block offsets inside numerical algorithms.
		"""
		value = self.validate_packed_state_layout(state)
		particle_count = value.shape[0] // self.state_dimension
		return value.reshape(
			(self.state_dimension, particle_count, *value.shape[1:])
		)

	def validate_packed_state_layout(self, state: np.ndarray) -> np.ndarray:
		"""Validate and return a component-major state-array layout.

		The leading axis must contain a non-zero whole number of physical
		component blocks.  Block consumers call this indirectly through
		:meth:`as_blocks`; integrators can call it when only layout validation is
		required.
		"""
		value = np.asarray(state)
		if (
			value.ndim == 0
			or value.shape[0] == 0
			or value.shape[0] % self.state_dimension
		):
			raise ValueError(
				f"The first state dimension must be divisible by {self.state_dimension} "
				f"for {self.__class__.__name__}."
			)
		return value

	def from_blocks(self, blocks: np.ndarray) -> np.ndarray:
		"""Flatten ``(components, particles, *samples)`` into state layout.

		This is the inverse view operation of :meth:`as_blocks`.  It is useful for
		internal algorithms that already produce all component blocks in one array
		and therefore need not concatenate them individually.
		"""
		value = np.asarray(blocks)
		if (
			value.ndim < 2
			or value.shape[0] != self.state_dimension
			or value.shape[1] == 0
		):
			raise ValueError(
				f"Blocks for {self.__class__.__name__} must have shape "
				f"({self.state_dimension}, N, *sample_axes) with N greater than zero."
			)
		return value.reshape(
			(self.state_dimension * value.shape[1], *value.shape[2:])
		)

	@classmethod
	def pack_components(cls, *components: np.ndarray) -> np.ndarray:
		"""Pack named-constructor inputs without requiring a trajectory instance.

		Every component has shape ``(N, *sample_axes)``. The concrete trajectory
		class supplies their required count and physical order through
		``state_dimension`` and the order in which callers pass the arrays.
		"""
		if len(components) != cls.state_dimension:
			raise ValueError(
				f"{cls.__name__} requires {cls.state_dimension} components."
			)
		values = tuple(np.asarray(component) for component in components)
		if not values or values[0].ndim == 0 or values[0].shape[0] == 0:
			raise ValueError("State components must be non-empty arrays.")
		if any(value.shape != values[0].shape for value in values[1:]):
			raise ValueError("All state components must have the same shape.")
		# Stack once to make the component axis explicit, then flatten that axis and
		# the particle axis through a view. This keeps packing as one allocation.
		blocks = np.stack(values, axis=0)
		return np.asarray(
			blocks.reshape(
				(cls.state_dimension * blocks.shape[1], *blocks.shape[2:])
			)
		)

	def particle_count(self, state: np.ndarray) -> int:
		"""Return ``N`` from the leading axis, ignoring all sample axes."""
		return int(self.as_blocks(state).shape[1])

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
		return self.layout.validate_packed_state_layout(state)

	def split(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
		"""Delegate component splitting to :attr:`layout`."""
		return self.layout.split(state)

	def as_blocks(self, state: np.ndarray) -> np.ndarray:
		"""Delegate explicit block exposure to :attr:`layout`."""
		return self.layout.as_blocks(state)

	def from_blocks(self, blocks: np.ndarray) -> np.ndarray:
		"""Delegate block flattening to :attr:`layout`."""
		return self.layout.from_blocks(blocks)

	def particle_count(self, state: np.ndarray) -> int:
		"""Delegate particle counting to :attr:`layout`."""
		return self.layout.particle_count(state)

# Compatibility spelling retained for callers importing the old base class.
Trajectory = StateConfiguration


__all__ = ["PackedStateLayout", "StateConfiguration", "Trajectory"]
