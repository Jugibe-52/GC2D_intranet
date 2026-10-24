"""General fixed-step classical explicit Euler method."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from methods.base import FixedStepMethod
from contracts.methods.step import StepInfo, StepResult
from contracts.methods.observation import IntegrationStep
from contracts.methods.quadrature import StageQuadrature
from request import SimulationRequest
from methods.classical._jacobians import _checked_vector_field


@dataclass(slots=True)
class ExplicitEuler(FixedStepMethod[None]):
	"""Classical forward Euler, ``z_next = z + h f(t, z)``."""

	jax_mode = 'device_resident'
	progress: bool = False

	def initialize(self, request: SimulationRequest) -> None:
		"""Euler needs no prepared resources beyond the supplied field."""

	def _apply_step(self, time: float, state: np.ndarray, step: float) -> np.ndarray:
		"""Apply one forward Euler step to an independent physical state."""
		value: np.ndarray = np.asarray(state, dtype=float)
		return np.asarray(value + step * _checked_vector_field(self.f, time, value), dtype=float)

	def advance(self, time: float, state: np.ndarray, step: float, *, observe: bool = True) -> StepResult[None]:
		"""Return a physical step and its optional passive quadrature point."""
		physical: np.ndarray = self._checked_state(state)
		after: np.ndarray = self._apply_step(time, physical, step)
		quadrature: StageQuadrature | None = (
			StageQuadrature((time,), (physical,), (step,)) if self.capture_quadrature else None
		)
		return StepResult(after, {}, None, quadrature)

	def build_observation(self, info: StepInfo, result: StepResult[None]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return self._apply_step(info.time, candidate, info.duration)
		return IntegrationStep(
			dynamics_name='', method_name=type(self).__name__,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=info.state_before.copy(),
			state_after=result.state.copy(), map_state=map_state,
		)


__all__ = ['ExplicitEuler']
