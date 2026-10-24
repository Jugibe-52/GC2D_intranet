"""Size-checked temporary physical copies, independent of models and methods."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, ClassVar, TYPE_CHECKING

from contracts.arrays import array_namespace

if TYPE_CHECKING:
	from contracts.arrays import BackendArray
	from types import ModuleType


@dataclass(frozen=True, slots=True)
class SingleFormulation:
	"""One temporary copy of a physical array with shape (physical_size, *samples)."""

	physical_size: int
	copies: ClassVar[int] = 1

	def __post_init__(self) -> None:
		if isinstance(self.physical_size, bool) or not isinstance(self.physical_size, Integral) or self.physical_size <= 0:
			raise ValueError('The physical size must be a positive integer.')
		object.__setattr__(self, 'physical_size', int(self.physical_size))

	@property
	def spatial_size(self) -> int:
		"""Number of workspace entries across all copies."""
		return self.copies * self.physical_size

	def _array(self, state: Any, size: int) -> Any:
		"""Validate static dimensions without converting traced array values."""
		value: BackendArray = array_namespace(state).asarray(state)
		if value.ndim == 0 or value.shape[0] != size:
			raise ValueError(f"Expected a packed state with leading dimension {size}.")
		return value

	def lift(self, physical: Any) -> Any:
		"""Build one or two equal physical copies, without auxiliary coordinates.

	The input has shape ``(d*N, *sample_axes)``. NumPy output owns its data,
	and its copy blocks can be mutated independently; JAX remains functional.
	"""
		value: BackendArray = self._array(physical, self.physical_size)
		return self.join_copies(*(value for _ in range(self.copies)))

	def join_copies(self, *physical: Any) -> Any:
		"""Pack explicitly supplied copies, including shifted projection candidates."""
		if len(physical) != self.copies:
			raise ValueError(f"{type(self).__name__} requires {self.copies} physical copies.")
		xp: ModuleType = array_namespace(*physical)
		values: tuple[Any, ...] = tuple(self._array(xp.asarray(value), self.physical_size) for value in physical)
		if any(value.shape != values[0].shape for value in values[1:]):
			raise ValueError("Physical copies must have matching sample axes.")
		return xp.concatenate(values, axis=0)

	def split_copies(self, spatial: Any) -> tuple[Any, ...]:
		"""Read physical copy blocks from a spatial array without projecting them."""
		value: BackendArray = self._array(spatial, self.spatial_size)
		return tuple(value[index * self.physical_size:(index + 1) * self.physical_size]
			for index in range(self.copies))


@dataclass(frozen=True, slots=True)
class DoubleFormulation(SingleFormulation):
	"""Two independent temporary copies; projection belongs to the method."""

	copies: ClassVar[int] = 2


__all__ = ['SingleFormulation', 'DoubleFormulation']
