"""Owned physical initial values and their structural coordinate layout."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from contracts.configuration import StateLayout
from formulations.layout import FCStateLayout, GCStateLayout


def _owned_initial_values(values: ArrayLike, layout: StateLayout) -> tuple[np.ndarray, StateLayout, int]:
	"""Prepare a finite packed vector and its independent structural layout."""
	if not isinstance(layout, StateLayout):
		raise TypeError("`layout` must implement StateLayout.")
	owned_layout: StateLayout = deepcopy(layout)
	value: np.ndarray = np.array(values, dtype=float, copy=True)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The initial state must be a finite non-empty vector.")
	owned_layout.validate_packed_state_layout(value)
	count: int = owned_layout.particle_count(value)
	if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
		or count <= 0 or value.size != count * owned_layout.state_dimension):
		raise ValueError("The layout must describe complete particle blocks.")
	value.setflags(write=False)
	return value, owned_layout, int(count)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class InitialConditions:
	"""Immutable physical input, packed as ``(d*N,)`` in its supplied units.

	GC uses ``[x, y]`` blocks and FC uses ``[x, y, vx, vy]``. This object
	contains neither a clock nor dynamics. Public arrays and layouts are copies.
	"""

	_values: np.ndarray = field(repr=False)
	_layout: StateLayout = field(repr=False)
	particle_count: int

	def __init__(self, values: ArrayLike, *, layout: StateLayout) -> None:
		value: np.ndarray
		owned_layout: StateLayout
		count: int
		value, owned_layout, count = _owned_initial_values(values, layout)
		object.__setattr__(self, '_values', value)
		object.__setattr__(self, '_layout', owned_layout)
		object.__setattr__(self, 'particle_count', count)

	@classmethod
	def gc(cls, *, x: ArrayLike, y: ArrayLike) -> InitialConditions:
		"""Construct one-dimensional GC component arrays in their supplied units."""
		return cls(GCStateLayout.pack_components(np.asarray(x), np.asarray(y)), layout=GCStateLayout())

	@classmethod
	def fc(cls, *, x: ArrayLike, y: ArrayLike, vx: ArrayLike, vy: ArrayLike) -> InitialConditions:
		"""Construct FC position and velocity blocks without rescaling them."""
		return cls(FCStateLayout.pack_components(*(np.asarray(v) for v in (x, y, vx, vy))), layout=FCStateLayout())

	@property
	def values(self) -> np.ndarray:
		"""Return an independent packed array suitable for numerical work."""
		return self._values.copy()

	@property
	def layout(self) -> StateLayout:
		"""Return a structural snapshot without exposing mutable layout internals."""
		return deepcopy(self._layout)

	def components(self) -> tuple[np.ndarray, ...]:
		"""Return independent component arrays using the owned convention."""
		return self.layout.split(self.values)

	def positions(self) -> tuple[np.ndarray, np.ndarray]:
		"""Return the two physical position components."""
		return self.layout.positions(self.values)

	def velocities(self) -> tuple[np.ndarray, np.ndarray]:
		"""Return named velocity components; GC has no stored velocities."""
		components: tuple[np.ndarray, ...] = self.components()
		if not hasattr(components, 'vx') or not hasattr(components, 'vy'):
			raise ValueError("This initial layout has no named velocity components.")
		return np.asarray(components.vx), np.asarray(components.vy)

	def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
		return _restore_initial, (self._values, self._layout)


def _restore_initial(values: np.ndarray, layout: StateLayout) -> InitialConditions:
	"""Reapply ownership and validation after serialization."""
	return InitialConditions(values, layout=layout)


__all__ = ["InitialConditions"]
