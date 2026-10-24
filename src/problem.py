"""Owned physical initial conditions associated with compatible dynamics."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from contracts.dynamics.protocols import DynamicalSystem
from contracts.dynamics.coordinates import StateLayout
from dynamics.coordinates import PhysicalCoordinates, describe_coordinates
from states.initial import InitialConditions


@dataclass(frozen=True, slots=True, init=False)
class InitialValueProblem:
	"""Own physical initial data; the request independently selects its time.

	Omitted layouts use the dynamics coordinate declaration. External systems may
	declare coordinates or supply an explicit structural description.
	"""

	dynamics: DynamicalSystem
	initial_conditions: InitialConditions = field(init=False)
	particle_count: int = field(init=False)

	def __init__(self, dynamics: DynamicalSystem, initial_state: np.ndarray | InitialConditions,
	             *, layout: StateLayout | None = None) -> None:
		if not isinstance(dynamics, DynamicalSystem):
			raise TypeError("`dynamics` must implement DynamicalSystem.")
		if isinstance(initial_state, InitialConditions):
			if layout is not None:
				raise ValueError("InitialConditions already owns its layout; omit `layout`.")
			initial: InitialConditions = initial_state
		else:
			initial = self._from_array(dynamics, initial_state, layout)
		declared: StateLayout | None = getattr(dynamics, 'coordinates', None)
		if declared is not None and initial.layout != describe_coordinates(declared):
			raise TypeError('The initial coordinate order or roles are incompatible with the dynamics.')
		if initial.layout.state_dimension != dynamics.state_dimension:
			raise TypeError("The initial layout is incompatible with the dynamics.")
		object.__setattr__(self, 'dynamics', dynamics)
		object.__setattr__(self, 'initial_conditions', initial)
		object.__setattr__(self, 'particle_count', initial.particle_count)

	@staticmethod
	def _from_array(dynamics: DynamicalSystem, values: np.ndarray, layout: StateLayout | None) -> InitialConditions:
		"""Normalize the established array constructor through the same owner."""
		if layout is None:
			layout = getattr(dynamics, 'coordinates', None)
			if layout is None:
				raise TypeError('Supply an explicit layout or declare dynamics.coordinates.')
		return InitialConditions(values, layout=layout)

	@property
	def initial_state(self) -> np.ndarray:
		"""Independent initial snapshot, safe to pass to mutable numerical work."""
		return self.initial_conditions.values

	@property
	def layout(self) -> PhysicalCoordinates:
		return self.initial_conditions.layout


__all__ = ["InitialValueProblem"]
