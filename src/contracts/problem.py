"""Owned physical initial conditions associated with compatible dynamics."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from dynamics.protocols import DynamicalSystem
from contracts.configuration import StateLayout
from formulations.layout import GCStateLayout, FCStateLayout
from states.initial import InitialConditions


@dataclass(frozen=True, slots=True, init=False)
class InitialValueProblem:
	"""Own physical initial data; the request independently selects its time.

	Omitted layouts use the canonical GC/FC component order for dimensions 2/4.
	Other coordinate conventions require an explicit structural layout.
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
		if initial.layout.state_dimension != dynamics.state_dimension:
			raise TypeError("The initial layout is incompatible with the dynamics.")
		object.__setattr__(self, 'dynamics', dynamics)
		object.__setattr__(self, 'initial_conditions', initial)
		object.__setattr__(self, 'particle_count', initial.particle_count)

	@staticmethod
	def _from_array(dynamics: DynamicalSystem, values: np.ndarray, layout: StateLayout | None) -> InitialConditions:
		"""Normalize the established array constructor through the same owner."""
		if layout is None:
			if dynamics.state_dimension == 2:
				layout = GCStateLayout()
			elif dynamics.state_dimension == 4:
				layout = FCStateLayout()
			else:
				raise TypeError("Supply an explicit layout for these dynamics.")
		return InitialConditions(values, layout=layout)

	@property
	def initial_state(self) -> np.ndarray:
		"""Independent initial snapshot, safe to pass to mutable numerical work."""
		return self.initial_conditions.values

	@property
	def layout(self) -> StateLayout:
		return self.initial_conditions.layout


__all__ = ["InitialValueProblem"]
