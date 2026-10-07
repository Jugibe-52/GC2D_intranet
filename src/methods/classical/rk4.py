"""General fixed-step classical fourth-order Runge--Kutta method."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np

from dynamics import DynamicalSystem

from methods._compiled import CompiledFixedMethod
from contracts.step import StepInfo, StepResult
from formulations.state import PhysicalFormulation
from contracts.observation import IntegrationStep, StepObserver
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from methods.classical._rk4_core import physical_step, momentum_increment


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
class RK4(CompiledFixedMethod[None]):
	"""Classical RK4 with an output-independent uniform main grid."""

	track_energy: bool = False
	progress: bool = False
	step_observer: StepObserver | None = None

	# Resources owned by one run; excluded from constructor options.
	state_formulation: PhysicalFormulation = field(init=False, repr=False, compare=False)
	dynamics: DynamicalSystem = field(init=False, repr=False, compare=False)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Bind the RK4 map on the physical or energy-augmented workspace."""
		self.dynamics = problem.dynamics
		self.state_formulation = PhysicalFormulation(problem, request.t_span[0], self.track_energy)
		self.initial_state = self.state_formulation.initial_state
		self.metadata = {"execution_backend": "scipy", "execution_device": "cpu",
		                 "execution_device_index": 0}

	def _physical_step(self, t: float, candidate: np.ndarray, step: float) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
		"""Return the physical RK4 map and its four accepted quadrature states."""
		return physical_step(partial(_checked_vector_field, self.dynamics), t, candidate, step)

	def advance(self, t: float, state: np.ndarray, step: float) -> StepResult[None]:
		"""Advance physical stages, then the passive energy quadrature if requested."""
		physical, stages = self._physical_step(t, self.state_formulation.physical(state), step)
		increment = None
		if self.track_energy:
			increment = momentum_increment(self.state_formulation.momentum_rate, t, step, stages)
		return StepResult(self.state_formulation.finish(state, physical, t + step, increment), {}, None)

	def build_observation(self, info: StepInfo, result: StepResult[None]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return self._physical_step(info.time, candidate, info.duration)[0]
		return IntegrationStep(
			dynamics_name=type(self.dynamics).__name__, method_name=type(self).__name__,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=self.state_formulation.physical(info.state_before).copy(),
			state_after=self.state_formulation.physical(result.state).copy(), map_state=map_state, dynamics=self.dynamics,
		)


__all__ = ["RK4"]
