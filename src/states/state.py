"""Owned instantaneous physical states, independent of their preparation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from contracts.configuration import StateLayout


@dataclass(frozen=True, slots=True, init=False, eq=False)
class State:
	"""Store one finite component-major NumPy vector and its layout snapshot.

	Values use the dynamics' units without conversion. Layouts, rather than this
	container, handle arrays with trailing sample axes and backend tracing.
	"""

	_values: np.ndarray = field(repr=False)
	_layout: StateLayout = field(repr=False)

	def __init__(self, values: np.ndarray, *, layout: StateLayout) -> None:
		"""Validate and capture independent copies of values and interpretation."""
		if not isinstance(layout, StateLayout):
			raise TypeError("`layout` must implement StateLayout.")
		owned_layout = deepcopy(layout)
		value = np.array(values, dtype=float, copy=True)
		if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
			raise ValueError("A state must be a finite non-empty one-dimensional vector.")
		owned_layout.validate_packed_state_layout(value)
		count = owned_layout.particle_count(value)
		if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
			or count <= 0 or value.size != count * owned_layout.state_dimension):
			raise ValueError("The layout must describe complete particle blocks.")
		value.setflags(write=False)
		object.__setattr__(self, "_values", value)
		object.__setattr__(self, "_layout", owned_layout)

	@classmethod
	def from_components(cls, *components: np.ndarray, layout: StateLayout) -> State:
		"""Pack component arrays in layout order and capture one physical state."""
		# Packing is optional for external layouts in the structural protocol.
		packer = getattr(layout, "pack_components", None)
		if not callable(packer):
			raise TypeError("This layout does not provide pack_components.")
		return cls(packer(*components), layout=layout)

	@property
	def values(self) -> np.ndarray:
		"""Return an independent copy of the packed vector."""
		return self._values.copy()

	@property
	def initial_state(self) -> np.ndarray:
		"""Provide the same snapshot to InitialValueProblem's source protocol."""
		return self.values

	@property
	def layout(self) -> StateLayout:
		"""Return an independent copy of the component interpretation."""
		return deepcopy(self._layout)

	@property
	def particle_count(self) -> int:
		"""Return the number of particles in this snapshot."""
		return self._layout.particle_count(self._values)

	def split(self) -> tuple[np.ndarray, ...]:
		"""Return independent component data, preserving named built-in tuples."""
		return self._layout.split(self.values)

	def positions(self) -> tuple[np.ndarray, np.ndarray]:
		"""Return the spatial coordinate blocks of this snapshot."""
		return self._layout.positions(self.values)

	def velocities(self) -> tuple[np.ndarray, np.ndarray]:
		"""Return named velocity blocks when the layout defines them."""
		components: Any = self.split()
		if not hasattr(components, "vx") or not hasattr(components, "vy"):
			raise TypeError("This state's layout does not define velocity components.")
		return components.vx, components.vy


__all__ = ["State"]
