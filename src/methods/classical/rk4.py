"""General fixed-step classical fourth-order Runge--Kutta method."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics import DynamicalSystem, HamiltonianSystem

from integration.core import IntegrationMethod
from contracts.step import StepInfo, StepResult
from contracts.result import DiagnosticValue
from formulations.state import PhysicalFormulation
from contracts.observation import IntegrationStep, StepObserver
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest


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
	state_formulation: PhysicalFormulation = field(init=False, repr=False, compare=False)
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
			HamiltonianSystem,
		):
			raise TypeError("Energy tracking requires HamiltonianSystem.")
		physical_initial = problem.initial_state
		self.physical_size = physical_initial.size
		self.particle_count = problem.particle_count
		self.state_formulation = PhysicalFormulation(problem, request.t_span[0], self.track_energy)
		self.initial_state = self.state_formulation.initial_state

	def _physical_step(self, t: float, candidate: np.ndarray, step: float) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
		"""Return the physical RK4 map and its four accepted quadrature states."""
		k1 = _checked_vector_field(self.dynamics, t, candidate)
		z2 = candidate + step * k1 / 2
		k2 = _checked_vector_field(self.dynamics, t + step / 2, z2)
		z3 = candidate + step * k2 / 2
		k3 = _checked_vector_field(self.dynamics, t + step / 2, z3)
		z4 = candidate + step * k3
		k4 = _checked_vector_field(self.dynamics, t + step, z4)
		return np.asarray(candidate + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6), (candidate, z2, z3, z4)

	def _apply_step(self, t: float, candidate: np.ndarray, step: float) -> np.ndarray:
		return self._physical_step(t, candidate, step)[0]

	def advance(self, t: float, state: np.ndarray, step: float) -> StepResult[None]:
		"""Advance physical stages, then the passive energy quadrature if requested."""
		physical, stages = self._physical_step(t, self.state_formulation.physical(state), step)
		increment = None
		if self.track_energy:
			rates = [self.state_formulation.momentum_rate(t + c * step, z)
			         for c, z in zip((0., .5, .5, 1.), stages)]
			increment = step * (rates[0] + 2 * rates[1] + 2 * rates[2] + rates[3]) / 6
		return StepResult(self.state_formulation.finish(state, physical, t + step, increment), {}, None)

	def build_observation(self, info: StepInfo, result: StepResult[None]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return self._apply_step(info.time, candidate, info.duration)
		return IntegrationStep(
			dynamics_name=type(self.dynamics).__name__, method_name=type(self).__name__,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=self.state_formulation.physical(info.state_before).copy(),
			state_after=self.state_formulation.physical(result.state).copy(), map_state=map_state, dynamics=self.dynamics,
		)



__all__ = ["RK4"]
