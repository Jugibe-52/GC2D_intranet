"""Initial-value problems assembled from dynamics and initial configurations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from classes.dynamics import DynamicalSystem

from .configuration import InitialConfiguration


def _validate_shared_parameters(
	dynamics: DynamicalSystem,
	configuration: InitialConfiguration,
) -> None:
	"""Reject contradictory transitional copies of physical parameters."""
	for name in ("rho", "eta"):
		if not hasattr(dynamics, name) or not hasattr(configuration, name):
			continue
		try:
			dynamics_scalar = float(getattr(dynamics, name))
			configuration_scalar = float(getattr(configuration, name))
		except (TypeError, ValueError) as error:
			raise TypeError(f"`{name}` must be a scalar physical parameter.") from error
		if dynamics_scalar != configuration_scalar:
			raise ValueError(
				f"Conflicting `{name}` values in dynamics and initial configuration."
			)


@dataclass(frozen=True, slots=True)
class InitialValueProblem:
	"""Bind physical dynamics to one validated initial configuration."""

	dynamics: DynamicalSystem
	initial_configuration: InitialConfiguration

	def __post_init__(self) -> None:
		"""Validate capabilities, layout compatibility, and initial state."""
		if not isinstance(self.dynamics, DynamicalSystem):
			raise TypeError("`dynamics` must implement DynamicalSystem.")
		configuration = self.initial_configuration
		if not isinstance(configuration, InitialConfiguration):
			raise TypeError(
				"`initial_configuration` must be an InitialConfiguration instance."
			)
		state = configuration.initial_state
		if state is None:
			raise ValueError("The initial configuration has no initial state.")
		value = configuration.validate_packed_state(state)
		if value.ndim != 1 or not np.all(np.isfinite(value)):
			raise ValueError("The initial state must be a finite one-dimensional vector.")
		if configuration.state_dimension != self.dynamics.state_dimension:
			raise TypeError(
				"The initial configuration layout is incompatible with the dynamics."
			)
		# GC/FC configuration aliases still expose their historical physical
		# parameters. Until those compatibility fields can be removed, require
		# their metadata to agree with the authoritative dynamics object.
		_validate_shared_parameters(self.dynamics, configuration)

	@property
	def initial_state(self) -> np.ndarray:
		"""Return an independent copy of the validated physical initial state."""
		state = self.initial_configuration.initial_state
		assert state is not None
		return state

	@property
	def particle_count(self) -> int:
		"""Return the number of particles represented by the initial state."""
		return self.initial_configuration.particle_count(self.initial_state)


__all__ = ["InitialValueProblem"]
