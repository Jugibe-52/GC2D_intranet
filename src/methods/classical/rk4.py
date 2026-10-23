"""General fixed-step classical fourth-order Runge--Kutta method."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np

from contracts.dynamics.protocols import DynamicalSystem

from methods.base import FixedStepMethod
from contracts.methods.step import StepInfo, StepResult
from formulations.state import SingleFormulation
from contracts.methods.observation import IntegrationStep, StepObserver
from problem import InitialValueProblem
from request import SimulationRequest
from methods.classical._rk4_core import physical_step, momentum_increment

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from typing import Any


def _checked_vector_field(
	dynamics: DynamicalSystem,
	t: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate and shape-check one physical vector field."""
	derivative: np.ndarray = np.asarray(dynamics.vector_field(t, state))
	if derivative.shape != state.shape:
		raise ValueError("The vector field changed the physical state shape.")
	return derivative


@dataclass(slots=True)
class RK4(FixedStepMethod[None]):
	"""Classical RK4 with an output-independent fixed step and a clipped endpoint."""

	jax_mode = "device_resident"

	track_energy: bool = False
	progress: bool = False
	step_observer: StepObserver | None = None

	# Resources owned by one run; excluded from constructor options.
	state_formulation: SingleFormulation = field(init=False, repr=False, compare=False)
	dynamics: DynamicalSystem = field(init=False, repr=False, compare=False)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Bind the RK4 map on the physical or energy-augmented workspace."""
		self.dynamics = problem.dynamics
		self.state_formulation = SingleFormulation(problem.layout, problem.particle_count, self.track_energy)
		self.initial_state = self._initialize_state(problem, request)
		self.metadata = {}

	def _physical_step(self, t: float, candidate: np.ndarray, step: float) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
		"""Return the physical RK4 map and its four accepted quadrature states."""
		return physical_step(partial(_checked_vector_field, self.dynamics), t, candidate, step)

	def advance(self, t: float, state: np.ndarray, step: float) -> StepResult[None]:
		"""Advance physical stages, then the passive energy quadrature if requested."""
		physical: np.ndarray
		stages: tuple[np.ndarray, ...]
		physical, stages = self._physical_step(t, self.state_formulation.physical(state), step)
		increment: np.ndarray | None = None
		if self.track_energy:
			increment = momentum_increment(self._momentum_rate, t, step, stages)
		return StepResult(self._finish_state(state, physical, t + step, increment), {}, None)

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
