"""Validated physical initial arrays bound to their dynamics and layout."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
from dynamics.protocols import DynamicalSystem
from contracts.configuration import StateLayout
from formulations.layout import GCStateLayout, FCStateLayout


@dataclass(frozen=True, slots=True, init=False)
class InitialValueProblem:
	"""Own physical initial data; the request independently selects its time.

	Omitted layouts use the canonical GC/FC component order for dimensions 2/4.
	Other coordinate conventions require an explicit structural layout.
	"""

	dynamics: DynamicalSystem
	_initial_state: np.ndarray = field(init=False, repr=False)
	_layout: StateLayout = field(init=False, repr=False)
	particle_count: int = field(init=False)

	def __init__(self, dynamics: DynamicalSystem, initial_state: np.ndarray,
	             *, layout: StateLayout | None = None) -> None:
		if not isinstance(dynamics, DynamicalSystem):
			raise TypeError("`dynamics` must implement DynamicalSystem.")
		if layout is None:
			if dynamics.state_dimension == 2:
				layout = GCStateLayout()
			elif dynamics.state_dimension == 4:
				layout = FCStateLayout()
			else:
				raise TypeError("Supply an explicit layout for these dynamics.")
		if not isinstance(layout, StateLayout):
			raise TypeError("`layout` must implement StateLayout.")
		layout = deepcopy(layout)
		value = np.array(initial_state, dtype=float, copy=True)
		if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
			raise ValueError("The initial state must be a finite non-empty vector.")
		layout.validate_packed_state_layout(value)
		if layout.state_dimension != dynamics.state_dimension:
			raise TypeError("The initial layout is incompatible with the dynamics.")
		count = layout.particle_count(value)
		if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
			or count <= 0 or value.size != count * layout.state_dimension):
			raise ValueError("The layout must describe complete particle blocks.")
		value.setflags(write=False)
		object.__setattr__(self, 'dynamics', dynamics)
		object.__setattr__(self, '_initial_state', value)
		object.__setattr__(self, '_layout', layout)
		object.__setattr__(self, 'particle_count', int(count))

	@property
	def initial_state(self) -> np.ndarray:
		"""Independent initial snapshot, safe to pass to mutable numerical work."""
		return self._initial_state.copy()

	@property
	def layout(self) -> StateLayout:
		return deepcopy(self._layout)


__all__ = ["InitialValueProblem"]
