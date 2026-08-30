"""General fixed-step classical explicit Euler method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import DynamicalSystem

from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
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


@dataclass(frozen=True, slots=True)
class ExplicitEuler:
	"""Classical forward Euler, ``z_next = z + h f(t, z)``."""

	progress: bool = False
	step_observer: StepObserver | None = None

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate any compatible physical vector field on a fixed grid."""
		dynamics = problem.dynamics
		if not isinstance(dynamics, DynamicalSystem):
			raise TypeError("ExplicitEuler requires DynamicalSystem.")

		def advance(
			time: float,
			state: np.ndarray,
			step: float,
			step_index: int,
			observe: bool,
		) -> np.ndarray:
			def apply_step(candidate: np.ndarray) -> np.ndarray:
				"""Apply the forward Euler map to a diagnostic candidate."""
				value = np.asarray(candidate, dtype=float)
				return np.asarray(
					value + step * _checked_vector_field(dynamics, time, value),
					dtype=float,
				)

			state_before = np.asarray(state, dtype=float)
			state_after = apply_step(state_before)
			if observe and self.step_observer is not None:
				self.step_observer(
					IntegrationStep(
						dynamics_name=type(dynamics).__name__,
						method_name=type(self).__name__,
						step_index=step_index,
						start_time=time,
						time=time + step,
						duration=step,
						state_before=state_before.copy(),
						state_after=state_after.copy(),
						map_state=apply_step,
						dynamics=dynamics,
					)
				)
			return np.asarray(state_after, dtype=float)

		history, step_count = integrate_fixed_grid(
			problem.initial_state,
			request,
			advance,
			progress=bool(self.progress),
			label="ExplicitEuler",
		)
		return IntegrationData(
			t=request.output_times,
			states=history,
			diagnostics={"step_count": step_count},
		)


__all__ = ["ExplicitEuler"]
