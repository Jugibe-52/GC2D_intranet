"""Immutable physical coordinate descriptions and backend-preserving packing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from contracts.arrays import array_namespace

if TYPE_CHECKING:
	from types import ModuleType
	from contracts.arrays import BackendArray
	from contracts.dynamics.coordinates import StateLayout


class CoordinateValues(tuple[Any, ...]):
	"""Component views with names supplied by a coordinate description."""

	_names: tuple[str, ...]

	def __new__(cls, names: tuple[str, ...], values: tuple[Any, ...]) -> CoordinateValues:
		result: CoordinateValues = super().__new__(cls, values)
		result._names = names
		return result

	def __getattr__(self, name: str) -> Any:
		if name not in self._names:
			raise AttributeError(name)
		return self[self._names.index(name)]

	def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
		return type(self), (self._names, tuple(self))


@dataclass(frozen=True, slots=True)
class PhysicalCoordinates:
	"""Ordered component names and their planar position/velocity roles.

	A physical vector contains d blocks of N particles. Trailing sample axes
	are preserved by every operation. The description contains no model,
	potential, initial state, numerical method, or units conversion.
	"""

	component_names: tuple[str, ...]
	position_names: tuple[str, ...] = ()
	velocity_names: tuple[str, ...] = ()

	def __post_init__(self) -> None:
		for name in ('component_names', 'position_names', 'velocity_names'):
			object.__setattr__(self, name, tuple(getattr(self, name)))
		names: tuple[str, ...] = self.component_names
		if not names or any(not isinstance(name, str) or not name.isidentifier() for name in names) or len(set(names)) != len(names):
			raise ValueError('Physical components need unique non-empty identifier names.')
		for roles in (self.position_names, self.velocity_names):
			if len(roles) not in (0, 2) or len(set(roles)) != len(roles) or any(name not in names for name in roles):
				raise ValueError('Position and velocity roles must name two distinct declared components or be empty.')
		if set(self.position_names) & set(self.velocity_names):
			raise ValueError('Position and velocity roles must not overlap.')

	@property
	def state_dimension(self) -> int:
		return len(self.component_names)

	def validate_packed_state_layout(self, state: Any) -> Any:
		"""Require complete, nonempty component-major particle blocks."""
		value: BackendArray = array_namespace(state).asarray(state)
		if value.ndim == 0 or not value.shape[0] or value.shape[0] % self.state_dimension:
			raise ValueError(f'The first state dimension must be divisible by {self.state_dimension}.')
		return value

	def as_blocks(self, state: Any) -> Any:
		"""View an array as (components, particles, *sample_axes)."""
		value: BackendArray = self.validate_packed_state_layout(state)
		return value.reshape(self.state_dimension, value.shape[0] // self.state_dimension, *value.shape[1:])

	def split(self, state: Any) -> CoordinateValues:
		"""Return component views with named access, preserving NumPy/JAX arrays."""
		return CoordinateValues(self.component_names, tuple(self.as_blocks(state)))

	def from_blocks(self, blocks: Any) -> Any:
		"""Flatten complete component and particle axes without changing order."""
		value: BackendArray = array_namespace(blocks).asarray(blocks)
		if value.ndim < 2 or value.shape[0] != self.state_dimension or not value.shape[1]:
			raise ValueError(f'Blocks must have shape ({self.state_dimension}, N, *sample_axes), N > 0.')
		return value.reshape(self.state_dimension * value.shape[1], *value.shape[2:])

	def pack_components(self, *components: Any) -> Any:
		"""Own a packed array from equal-shaped component blocks."""
		if len(components) != self.state_dimension:
			raise ValueError(f'This coordinate description requires {self.state_dimension} components.')
		xp: ModuleType = array_namespace(*components)
		values: tuple[Any, ...] = tuple(xp.asarray(component) for component in components)
		if values[0].ndim == 0 or not values[0].shape[0]:
			raise ValueError('Physical components must be non-empty arrays.')
		if any(value.shape != values[0].shape for value in values[1:]):
			raise ValueError('All state components must have the same shape.')
		return self.from_blocks(xp.stack(values, axis=0))

	def particle_count(self, state: Any) -> int:
		return int(self.as_blocks(state).shape[1])

	def _roles(self, state: Any, names: tuple[str, ...]) -> tuple[Any, Any]:
		if len(names) != 2:
			raise ValueError('This coordinate description has no corresponding planar components.')
		blocks: CoordinateValues = self.split(state)
		return blocks[self.component_names.index(names[0])], blocks[self.component_names.index(names[1])]

	def positions(self, state: Any) -> tuple[Any, Any]:
		return self._roles(state, self.position_names)

	def velocities(self, state: Any) -> tuple[Any, Any]:
		return self._roles(state, self.velocity_names)


def describe_coordinates(layout: StateLayout) -> PhysicalCoordinates:
	"""Snapshot a structural declaration as immutable data, without model inference."""
	from contracts.dynamics.coordinates import StateLayout

	if not isinstance(layout, StateLayout):
		raise TypeError('`layout` must declare component names and position/velocity roles.')
	description: PhysicalCoordinates = PhysicalCoordinates(layout.component_names, layout.position_names, layout.velocity_names)
	if layout.state_dimension != description.state_dimension:
		raise ValueError('The declared coordinate names must match the physical dimension.')
	return description


__all__ = ['PhysicalCoordinates', 'CoordinateValues', 'describe_coordinates']
