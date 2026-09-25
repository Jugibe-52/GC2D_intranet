"""General fixed-step classical explicit Euler method."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics import DynamicalSystem

from formulations.state import PhysicalFormulation
from integration.core import IntegrationMethod
from contracts.step import StepInfo, StepResult
from contracts.result import DiagnosticValue
from contracts.observation import IntegrationStep, StepObserver
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest


def _checked_vector_field(
	dynamics: DynamicalSystem,
	time: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate one finite vector field with the physical state shape."""
	derivative = np.asarray(dynamics.vector_field(time, state), dtype=float)
	if derivative.shape != state.shape or not np.all(np.isfinite(derivative)):
		raise ValueError("The vector field changed shape or became non-finite.")
	return derivative


@dataclass(slots=True)
class ExplicitEuler(IntegrationMethod[None]):
	"""Classical forward Euler, ``z_next = z + h f(t, z)``."""

	track_energy: bool = False
	progress: bool = False
	step_observer: StepObserver | None = None

	# Resources owned by one run; excluded from constructor options.
	state_formulation: PhysicalFormulation = field(init=False, repr=False, compare=False)
	dynamics: DynamicalSystem = field(init=False, repr=False, compare=False)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Bind the Euler map, its physical observation and output extraction."""
		self.dynamics = problem.dynamics
		if not isinstance(self.dynamics, DynamicalSystem):
			raise TypeError("ExplicitEuler requires DynamicalSystem.")
		self.state_formulation = PhysicalFormulation(problem, request.t_span[0], self.track_energy)
		self.initial_state = self.state_formulation.initial_state

	def _apply_step(self, time: float, state: np.ndarray, step: float) -> np.ndarray:
		"""Apply one forward Euler step to an independent physical state."""
		value = np.asarray(state, dtype=float)
		return np.asarray(value + step * _checked_vector_field(self.dynamics, time, value), dtype=float)

	def advance(self, time: float, state: np.ndarray, step: float) -> StepResult[None]:
		"""Return one forward Euler step without collection or observation."""
		physical = self.state_formulation.physical(state)
		increment = step * self.state_formulation.momentum_rate(time, physical) if self.track_energy else None
		after = self.state_formulation.finish(state, self._apply_step(time, physical, step), time + step, increment)
		return StepResult(after, {}, None)

	def build_observation(self, info: StepInfo, result: StepResult[None]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return self._apply_step(info.time, candidate, info.duration)
		return IntegrationStep(
			dynamics_name=type(self.dynamics).__name__, method_name=type(self).__name__,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=self.state_formulation.physical(info.state_before).copy(),
			state_after=self.state_formulation.physical(result.state).copy(), map_state=map_state, dynamics=self.dynamics,
		)


__all__ = ["ExplicitEuler"]
