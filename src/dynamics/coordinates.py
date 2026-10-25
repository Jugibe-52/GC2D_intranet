"""Immutable physical coordinate descriptions and NumPy component operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from numbers import Integral
from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from types import ModuleType
	from contracts.arrays import BackendArray
	from contracts.dynamics.coordinates import StateLayout


class PhysicalCoordinates(ABC):
	"""Stateless component-major array operations with an explicit convention.

	Concrete classes fix the component count and physical accessors. A vector
	contains d blocks of N particles; trailing sample axes are preserved.
	Instances contain no model, potential, initial state or configurable roles.
	"""

	__slots__ = ()
	array_namespace: ClassVar[ModuleType] = np

	@property
	@abstractmethod
	def state_dimension(self) -> int:
		"""Fixed number of physical components per particle."""

	def __eq__(self, other: object) -> bool:
		return type(self) is type(other)

	def __hash__(self) -> int:
		return hash(type(self))

	def snapshot(self) -> PhysicalCoordinates:
		"""Own a copy of an explicit custom NumPy convention."""
		return deepcopy(self)

	def validate_packed_state_layout(self, state: Any) -> Any:
		"""Require complete, nonempty component-major particle blocks."""
		value: BackendArray = self.array_namespace.asarray(state)
		if value.ndim == 0 or not value.shape[0] or value.shape[0] % self.state_dimension:
			raise ValueError(f'The first state dimension must be divisible by {self.state_dimension}.')
		return value

	def as_blocks(self, state: Any) -> Any:
		"""View an array as (components, particles, *sample_axes)."""
		value: BackendArray = self.validate_packed_state_layout(state)
		return value.reshape(self.state_dimension, value.shape[0] // self.state_dimension, *value.shape[1:])

	def split(self, state: Any) -> tuple[Any, ...]:
		"""Return ordered component views using this coordinate class's array library."""
		return tuple(self.as_blocks(state))

	def from_blocks(self, blocks: Any) -> Any:
		"""Flatten complete component and particle axes without changing order."""
		value: BackendArray = self.array_namespace.asarray(blocks)
		if value.ndim < 2 or value.shape[0] != self.state_dimension or not value.shape[1]:
			raise ValueError(f'Blocks must have shape ({self.state_dimension}, N, *sample_axes), N > 0.')
		return value.reshape(self.state_dimension * value.shape[1], *value.shape[2:])

	def pack_components(self, *components: Any) -> Any:
		"""Own a packed array from equal-shaped component blocks."""
		if len(components) != self.state_dimension:
			raise ValueError(f'This coordinate description requires {self.state_dimension} components.')
		xp: ModuleType = self.array_namespace
		values: tuple[Any, ...] = tuple(xp.asarray(component) for component in components)
		if values[0].ndim == 0 or not values[0].shape[0]:
			raise ValueError('Physical components must be non-empty arrays.')
		if any(value.shape != values[0].shape for value in values[1:]):
			raise ValueError('All state components must have the same shape.')
		return self.from_blocks(xp.stack(values, axis=0))

	def particle_count(self, state: Any) -> int:
		return int(self.as_blocks(state).shape[1])

	def positions(self, state: Any) -> tuple[Any, Any]:
		"""Reject position access unless a concrete convention supplies it."""
		raise ValueError('This coordinate class does not define planar positions.')

	def velocities(self, state: Any) -> tuple[Any, Any]:
		"""Reject velocity access unless a concrete convention supplies it."""
		raise ValueError('This coordinate class does not define velocity components.')


class GCCoordinates(PhysicalCoordinates):
	"""Fixed guiding-center blocks: all x values followed by all y values."""

	__slots__ = ()

	@property
	def state_dimension(self) -> int:
		return 2

	def positions(self, state: Any) -> tuple[Any, Any]:
		x: Any
		y: Any
		x, y = self.split(state)
		return x, y

	def snapshot(self) -> GCCoordinates:
		"""Retain the GC convention in NumPy, including from its JAX subclass."""
		return GCCoordinates()


class FCCoordinates(PhysicalCoordinates):
	"""Fixed full-cyclotron blocks in (x, y, vx, vy) order."""

	__slots__ = ()

	@property
	def state_dimension(self) -> int:
		return 4

	def positions(self, state: Any) -> tuple[Any, Any]:
		blocks: tuple[Any, ...] = self.split(state)
		return blocks[0], blocks[1]

	def velocities(self, state: Any) -> tuple[Any, Any]:
		blocks: tuple[Any, ...] = self.split(state)
		return blocks[2], blocks[3]

	def snapshot(self) -> FCCoordinates:
		"""Retain the FC convention in NumPy, including from its JAX subclass."""
		return FCCoordinates()


def describe_coordinates(layout: StateLayout) -> PhysicalCoordinates:
	"""Own an explicit NumPy convention, never choosing a model by dimension."""
	from contracts.dynamics.coordinates import StateLayout

	if not isinstance(layout, StateLayout):
		raise TypeError('`layout` must implement StateLayout, including snapshot().')
	description: StateLayout = layout.snapshot()
	if not isinstance(description, PhysicalCoordinates) or description.array_namespace is not np:
		raise TypeError('A coordinate snapshot must be a NumPy PhysicalCoordinates implementation.')
	dimension: int = description.state_dimension
	if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension <= 0:
		raise ValueError('The state dimension must be a positive integer.')
	if dimension != layout.state_dimension:
		raise ValueError('A coordinate snapshot must preserve the state dimension.')
	return description


__all__ = ['PhysicalCoordinates', 'GCCoordinates', 'FCCoordinates', 'describe_coordinates']
