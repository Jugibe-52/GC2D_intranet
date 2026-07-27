"""General fixed-step classical fourth-order Runge--Kutta method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from classes.dynamics import DynamicalSystem, ExtendedHamiltonianSystem

from .._fixed import integrate_fixed_grid
from .._result import IntegrationData
from ..formulations.base import generalized_energy_error
from ..problem import InitialValueProblem
from ..request import SimulationRequest


def _checked_vector_field(
	dynamics: DynamicalSystem,
	t: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate and shape-check one physical vector field."""
	derivative = np.asarray(dynamics.vector_field(t, state))
	if derivative.shape != state.shape:
		raise ValueError("The vector field changed the physical state shape.")
	return derivative


@dataclass(frozen=True, slots=True)
class RK4:
	"""Classical RK4 with an output-independent uniform main grid."""

	track_energy: bool = False
	progress: bool = False

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate any compatible physical vector field."""
		dynamics = problem.dynamics
		if not isinstance(dynamics, DynamicalSystem):
			raise TypeError("RK4 requires DynamicalSystem.")
		if self.track_energy and not isinstance(
			dynamics,
			ExtendedHamiltonianSystem,
		):
			raise TypeError("Energy tracking requires ExtendedHamiltonianSystem.")

		physical_initial = problem.initial_state
		physical_size = physical_initial.size
		particle_count = problem.particle_count
		initial_state = (
			physical_initial
			if not self.track_energy
			else np.concatenate((physical_initial, np.zeros(particle_count)))
		)

		def derivative(t: float, value: np.ndarray) -> np.ndarray:
			physical = value[:physical_size]
			physical_derivative = _checked_vector_field(dynamics, t, physical)
			if not self.track_energy:
				return physical_derivative
			assert isinstance(dynamics, ExtendedHamiltonianSystem)
			momentum_derivative = np.asarray(
				dynamics.extended_momentum_derivative(t, physical)
			)
			if momentum_derivative.shape != (particle_count,):
				raise ValueError(
					"The extended-momentum derivative changed its shape."
				)
			return np.concatenate((physical_derivative, momentum_derivative))

		def advance(
			t: float,
			value: np.ndarray,
			step: float,
			_step_index: int,
			_observe: bool,
		) -> np.ndarray:
			k1 = derivative(t, value)
			k2 = derivative(t + step / 2, value + step * k1 / 2)
			k3 = derivative(t + step / 2, value + step * k2 / 2)
			k4 = derivative(t + step, value + step * k3)
			return np.asarray(value + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6)

		history, step_count = integrate_fixed_grid(
			initial_state,
			request,
			advance,
			progress=bool(self.progress),
			label="RK4",
		)
		states = history[:physical_size]
		diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
			"step_count": step_count,
		}
		momentum = history[physical_size:] if self.track_energy else None
		if momentum is not None:
			diagnostics["extended_momentum"] = momentum
			diagnostics["energy_error"] = generalized_energy_error(
				request.output_times,
				states,
				momentum,
				dynamics,
			)
		return IntegrationData(
			t=request.output_times,
			states=states,
			diagnostics=diagnostics,
		)


__all__ = ["RK4"]
