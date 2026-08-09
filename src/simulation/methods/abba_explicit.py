"""Explicit endpoint-time ABBA integration with projection by averaging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import DynamicalSystem

from .._fixed import integrate_fixed_grid
from .._result import IntegrationData
from ..observation import IntegrationStep, StepObserver
from ..problem import InitialValueProblem
from ..request import SimulationRequest


@dataclass(frozen=True, slots=True)
class _ExplicitABBAStep:
	"""Physical average and off-diagonal copy separation after one ABBA map."""

	state: np.ndarray
	copy_separation_norm: float


def _checked_vector_field(
	dynamics: DynamicalSystem,
	t: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate a finite vector field without allowing a layout change."""
	result = np.asarray(dynamics.vector_field(t, state), dtype=float)
	if result.shape != state.shape or not np.all(np.isfinite(result)):
		raise ValueError("The vector field changed shape or became non-finite.")
	return result


def _explicit_abba_step(
	dynamics: DynamicalSystem,
	t: float,
	state: np.ndarray,
	step: float,
) -> _ExplicitABBAStep:
	"""Apply endpoint-time A-B-B-A and project both copies by their mean."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The ABBA physical state must be a finite, non-empty vector.")
	half_step = step / 2.0
	final_time = t + step

	# Each complete step starts from the physical diagonal u=v=y_n.
	u_first = value + half_step * _checked_vector_field(dynamics, t, value)
	v_first = value + half_step * _checked_vector_field(dynamics, t, u_first)
	v_final = v_first + half_step * _checked_vector_field(
		dynamics,
		final_time,
		u_first,
	)
	u_final = u_first + half_step * _checked_vector_field(
		dynamics,
		final_time,
		v_final,
	)
	separation = u_final - v_final
	return _ExplicitABBAStep(
		state=np.asarray((u_final + v_final) / 2.0),
		copy_separation_norm=float(np.linalg.norm(separation, ord=np.inf)),
	)


@dataclass(frozen=True, slots=True)
class ExplicitABBA:
	"""Second-order explicit ABBA method with arithmetic mean projection.

	The method duplicates the physical state, applies the endpoint-time A-B-B-A
	shears, and averages the two final copies. The average is an inexpensive
	Euclidean projection, but it does not guarantee a symplectic physical map.
	"""

	progress: bool = False
	step_observer: StepObserver | None = None

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate one planar physical problem and retain copy separation."""
		dynamics = problem.dynamics
		if not isinstance(dynamics, DynamicalSystem):
			raise TypeError("ExplicitABBA requires DynamicalSystem.")
		if dynamics.state_dimension != 2:
			raise TypeError("ExplicitABBA requires planar two-component dynamics.")

		copy_separation_norms: list[float] = []

		def advance(
			t: float,
			state: np.ndarray,
			step: float,
			step_index: int,
			observe: bool,
		) -> np.ndarray:
			def apply_step(candidate: np.ndarray) -> np.ndarray:
				"""Apply the same fixed-time explicit map to one candidate state."""
				return _explicit_abba_step(dynamics, t, candidate, step).state

			state_before = np.asarray(state, dtype=float)
			result = _explicit_abba_step(dynamics, t, state_before, step)
			if observe:
				copy_separation_norms.append(result.copy_separation_norm)
				if self.step_observer is not None:
					self.step_observer(
						IntegrationStep(
							dynamics_name=type(dynamics).__name__,
							method_name=type(self).__name__,
							step_index=step_index,
							time=t + step,
							duration=step,
							state_before=state_before.copy(),
							state_after=result.state.copy(),
							map_state=apply_step,
						)
					)
			return result.state

		history, step_count = integrate_fixed_grid(
			problem.initial_state,
			request,
			advance,
			progress=bool(self.progress),
			label=type(self).__name__,
		)
		return IntegrationData(
			t=request.output_times,
			states=np.asarray(history),
			diagnostics={
				"step_count": step_count,
				"copy_separation_norms": np.asarray(
					copy_separation_norms,
					dtype=float,
				),
				"projection_kind": "arithmetic_mean",
				"vector_field_evaluations_per_step": 4,
			},
		)


__all__ = ["ExplicitABBA"]
