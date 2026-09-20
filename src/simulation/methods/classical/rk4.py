"""General fixed-step classical fourth-order Runge--Kutta method."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics import DynamicalSystem, ExtendedHamiltonianSystem

from ...integration import IntegrationMethod, StepInfo, StepResult
from ..._result import DiagnosticValue
from ...formulations.base import generalized_energy_error
from ...observation import IntegrationStep, StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest


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


@dataclass(slots=True)
class RK4(IntegrationMethod[None]):
	"""Classical RK4 with an output-independent uniform main grid."""

	track_energy: bool = False
	progress: bool = False
	step_observer: StepObserver | None = None

	# Resources owned by one run; excluded from constructor options.
	dynamics: DynamicalSystem = field(init=False, repr=False, compare=False)
	physical_size: int = field(init=False, repr=False, compare=False)
	particle_count: int = field(init=False, repr=False, compare=False)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Bind the RK4 map on the physical or energy-augmented workspace."""
		self.dynamics = problem.dynamics
		if not isinstance(self.dynamics, DynamicalSystem):
			raise TypeError("RK4 requires DynamicalSystem.")
		if self.track_energy and not isinstance(
			self.dynamics,
			ExtendedHamiltonianSystem,
		):
			raise TypeError("Energy tracking requires ExtendedHamiltonianSystem.")
		physical_initial = problem.initial_state
		self.physical_size = physical_initial.size
		self.particle_count = problem.particle_count
		initial_state = (
			physical_initial
			if not self.track_energy
			else np.concatenate((physical_initial, np.zeros(self.particle_count)))
		)
		self.initial_state = initial_state

	def _derivative(self, t: float, value: np.ndarray) -> np.ndarray:
		physical = value[:self.physical_size]
		physical_derivative = _checked_vector_field(self.dynamics, t, physical)
		if not self.track_energy:
			return physical_derivative
		assert isinstance(self.dynamics, ExtendedHamiltonianSystem)
		momentum_derivative = np.asarray(
			self.dynamics.extended_momentum_derivative(t, physical)
		)
		if momentum_derivative.shape != (self.particle_count,):
			raise ValueError(
				"The extended-momentum derivative changed its shape."
			)
		return np.concatenate((physical_derivative, momentum_derivative))

	def _apply_step(self, t: float, candidate: np.ndarray, step: float) -> np.ndarray:
		"""Evaluate the established four RK stages in their original arithmetic order."""
		k1 = self._derivative(t, candidate)
		k2 = self._derivative(t + step / 2, candidate + step * k1 / 2)
		k3 = self._derivative(t + step / 2, candidate + step * k2 / 2)
		k4 = self._derivative(t + step, candidate + step * k3)
		return np.asarray(candidate + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6)

	def advance(self, t: float, state: np.ndarray, step: float) -> StepResult[None]:
		"""Return one RK4 step; explicit stages have no nonlinear-work counters."""
		return StepResult(self._apply_step(t, state, step), {}, None)

	def build_observation(self, info: StepInfo, result: StepResult[None]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return self._apply_step(info.time, candidate, info.duration)
		return IntegrationStep(
			dynamics_name=type(self.dynamics).__name__, method_name=type(self).__name__,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=info.state_before.copy(),
			state_after=result.state.copy(), map_state=map_state, dynamics=self.dynamics,
		)

	def export_history(self, times: np.ndarray, history: np.ndarray) -> tuple[np.ndarray, dict[str, DiagnosticValue]]:
		states = history[:self.physical_size]
		diagnostics: dict[str, DiagnosticValue] = {}
		if self.track_energy:
			momentum = history[self.physical_size:]
			diagnostics['extended_momentum'] = momentum
			diagnostics['energy_error'] = generalized_energy_error(times, states, momentum, self.dynamics)
		return states, diagnostics


__all__ = ["RK4"]
