"""General fixed-step classical explicit Euler method."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics import DynamicalSystem

from ...integration import IntegrationMethod, StepInfo, StepResult
from ..._result import DiagnosticValue
from ...observation import IntegrationStep, StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest


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

	progress: bool = False
	step_observer: StepObserver | None = None

	# Resources owned by one run; excluded from constructor options.
	dynamics: DynamicalSystem = field(init=False, repr=False, compare=False)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Bind the Euler map, its physical observation and output extraction."""
		self.dynamics = problem.dynamics
		if not isinstance(self.dynamics, DynamicalSystem):
			raise TypeError("ExplicitEuler requires DynamicalSystem.")
		self.initial_state = problem.initial_state

	def _apply_step(self, time: float, state: np.ndarray, step: float) -> np.ndarray:
		"""Apply one forward Euler step to an independent physical state."""
		value = np.asarray(state, dtype=float)
		return np.asarray(value + step * _checked_vector_field(self.dynamics, time, value), dtype=float)

	def advance(self, time: float, state: np.ndarray, step: float) -> StepResult[None]:
		"""Return one forward Euler step without collection or observation."""
		return StepResult(self._apply_step(time, state, step), {}, None)

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
		return history, {}


__all__ = ["ExplicitEuler"]
