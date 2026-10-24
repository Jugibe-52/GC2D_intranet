"""General fixed-step classical fourth-order Runge--Kutta method."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import numpy as np

from methods.base import FixedStepMethod
from contracts.methods.step import StepInfo, StepResult
from contracts.methods.observation import IntegrationStep
from contracts.methods.quadrature import StageQuadrature
from request import SimulationRequest
from methods.classical._rk4_core import physical_step
from methods.classical._jacobians import _checked_vector_field


@dataclass(slots=True)
class RK4(FixedStepMethod[None]):
	"""Classical RK4 with an output-independent fixed step and a clipped endpoint."""

	jax_mode = 'device_resident'
	progress: bool = False

	def initialize(self, request: SimulationRequest) -> None:
		"""RK4 needs no prepared resources beyond the supplied field."""

	def _physical_step(self, time: float, candidate: np.ndarray, step: float) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
		"""Return the physical map and its four accepted quadrature states."""
		return physical_step(partial(_checked_vector_field, self.f), time, candidate, step)

	def advance(self, time: float, state: np.ndarray, step: float, *, observe: bool = True) -> StepResult[None]:
		"""Advance physical stages and optionally expose their signed weights."""
		physical: np.ndarray
		stages: tuple[np.ndarray, ...]
		physical, stages = self._physical_step(time, self._checked_state(state), step)
		quadrature: StageQuadrature | None = None
		if self.capture_quadrature:
			quadrature = StageQuadrature((time, time + step / 2, time + step / 2, time + step),
				stages, (step / 6, step / 3, step / 3, step / 6))
		return StepResult(physical, {}, None, quadrature)

	def build_observation(self, info: StepInfo, result: StepResult[None]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return self._physical_step(info.time, candidate, info.duration)[0]
		return IntegrationStep(
			dynamics_name='', method_name=type(self).__name__,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=info.state_before.copy(),
			state_after=result.state.copy(), map_state=map_state,
		)


__all__ = ['RK4']
