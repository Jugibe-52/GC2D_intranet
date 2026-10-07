"""Initial-value problems assembled from dynamics and initial configurations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from dynamics.protocols import DynamicalSystem

from contracts.configuration import InitialConfiguration, StateLayout


@dataclass(frozen=True, slots=True)
class InitialValueProblem:
	"""Bind dynamics to a validated snapshot of one initial configuration.

	The original configuration remains available as provenance. Later edits to
	that source cannot change this problem's physical state or layout.
	"""

	dynamics: DynamicalSystem
	initial_configuration: InitialConfiguration
	_initial_state: np.ndarray = field(init=False, repr=False, compare=False)
	_layout: StateLayout = field(init=False, repr=False, compare=False)
	_particle_count: int = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Validate capabilities, layout compatibility, and initial state."""
		if not isinstance(self.dynamics, DynamicalSystem):
			raise TypeError("`dynamics` must implement DynamicalSystem.")
		configuration = self.initial_configuration
		if not isinstance(configuration, InitialConfiguration):
			raise TypeError(
				"`initial_configuration` must be an InitialConfiguration instance."
			)
		layout = deepcopy(configuration.layout)
		if not isinstance(layout, StateLayout):
			raise TypeError("`initial_configuration.layout` must implement StateLayout.")
		state = configuration.initial_state
		if state is None:
			raise ValueError("The initial configuration has no initial state.")
		value = np.array(layout.validate_packed_state_layout(state), dtype=float, copy=True)
		if value.ndim != 1 or not np.all(np.isfinite(value)):
			raise ValueError("The initial state must be a finite one-dimensional vector.")
		if layout.state_dimension != self.dynamics.state_dimension:
			raise TypeError(
				"The initial configuration layout is incompatible with the dynamics."
			)
		count = layout.particle_count(value)
		if (
			isinstance(count, (bool, np.bool_))
			or not isinstance(count, (int, np.integer))
			or count <= 0
			or value.size != count * layout.state_dimension
		):
			raise ValueError("The initial layout must describe complete particle blocks.")
		value.setflags(write=False)
		object.__setattr__(self, "_initial_state", value)
		object.__setattr__(self, "_layout", layout)
		object.__setattr__(self, "_particle_count", int(count))
		# Physical parameters belong exclusively to dynamics. The configuration
		# contributes only state geometry and packed-layout behavior.

	@property
	def initial_state(self) -> np.ndarray:
		"""Return an independent copy of the validated physical initial state."""
		return self._initial_state.copy()

	@property
	def layout(self) -> StateLayout:
		"""Return an independent layout snapshot for physical-state interpretation."""
		return deepcopy(self._layout)

	@property
	def particle_count(self) -> int:
		"""Return the number of particles represented by the initial state."""
		return self._particle_count


__all__ = ["InitialValueProblem"]
