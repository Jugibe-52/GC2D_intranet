"""Midpoint ABBA integration with arithmetic-mean diagonal projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import DynamicalSystem, GuidingCenterDynamics

from .._fully_extended import _integrate_abba_fully_extended_midpoint
from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
from ...observation import IntegrationStep, StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from ._configuration import (
	StateExtension,
	_state_dimension_diagnostics,
	_validate_state_extension,
)
from ._core import _ABBAStages, _evaluate_unprojected_stages
from ._implicit import _shared_time_kappa_increment_from_stages


@dataclass(frozen=True, slots=True)
class _ABBA2MidpointStep:
	"""Physical average and off-diagonal copy separation after one ABBA map."""

	state: np.ndarray
	copy_separation_norm: float
	stages: _ABBAStages


def _midpoint_abba_step(
	dynamics: DynamicalSystem,
	t: float,
	state: np.ndarray,
	step: float,
) -> _ABBA2MidpointStep:
	"""Apply endpoint-time A-B-B-A and project both copies by their mean."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The ABBA physical state must be a finite, non-empty vector.")
	# Midpoint projection starts both copies on the physical diagonal, then uses
	# exactly the same endpoint-time A-B-B-A map as the implicit formulations.
	stages = _evaluate_unprojected_stages(
		dynamics,
		t,
		value,
		value,
		step,
	)
	return _ABBA2MidpointStep(
		state=np.asarray((stages.u_final + stages.v_final) / 2.0),
		copy_separation_norm=float(
			np.linalg.norm(stages.residual, ord=np.inf)
		),
		stages=stages,
	)


@dataclass(frozen=True, slots=True)
class ABBA2Midpoint:
	"""Second-order midpoint ABBA with three state-extension configurations.

	The method duplicates the state selected by ``state_extension``, applies the
	endpoint-time A-B-B-A shears, and averages the two final copies. The average
	is inexpensive but does not guarantee a symplectic physical map. Midpoint has
	no residual-formulation or nonlinear-solver axis.
	"""

	state_extension: StateExtension = "physical"
	progress: bool = False
	step_observer: StepObserver | None = None

	def __post_init__(self) -> None:
		"""Validate the only configuration axis used by midpoint ABBA."""
		object.__setattr__(
			self,
			"state_extension",
			_validate_state_extension(self.state_extension),
		)

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate one planar problem and retain the final copy separation."""
		if self.state_extension == "fully_extended":
			return _integrate_abba_fully_extended_midpoint(self, problem, request)
		dynamics = problem.dynamics
		if not isinstance(dynamics, DynamicalSystem):
			raise TypeError("ABBA2Midpoint requires DynamicalSystem.")
		if dynamics.state_dimension != 2:
			raise TypeError("ABBA2Midpoint requires planar two-component dynamics.")
		shared_time_extension = self.state_extension == "shared_time"
		if shared_time_extension:
			if not isinstance(dynamics, GuidingCenterDynamics):
				raise TypeError(
					"ABBA2Midpoint requires GuidingCenterDynamics for shared_time."
				)
			if np.asarray(problem.initial_state).shape != (2,):
				raise ValueError(
					"ABBA2Midpoint requires exactly one GC particle for shared_time."
				)

		copy_separation_norms: list[float] = []

		def advance(
			t: float,
			state: np.ndarray,
			step: float,
			step_index: int,
			observe: bool,
		) -> np.ndarray:
			def apply_step(candidate: np.ndarray) -> np.ndarray:
				"""Apply the same fixed-time midpoint map to one candidate state."""
				return _midpoint_abba_step(dynamics, t, candidate, step).state

			if shared_time_extension:
				extended_before = np.asarray(state, dtype=float)
				if extended_before.shape != (4,) or not np.all(np.isfinite(extended_before)):
					raise ValueError(
						"The accepted shared-time state must use finite (x,y,t,kappa) order."
					)
				time_tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(t))
				if not np.isclose(
					float(extended_before[2]),
					t,
					rtol=0.0,
					atol=float(time_tolerance),
				):
					raise RuntimeError(
						"The shared-time extension and integration-grid times diverged."
					)
				state_before = extended_before[:2]
			else:
				extended_before = None
				state_before = np.asarray(state, dtype=float)
			result = _midpoint_abba_step(dynamics, t, state_before, step)
			if observe:
				copy_separation_norms.append(result.copy_separation_norm)
				if self.step_observer is not None:
					self.step_observer(
						IntegrationStep(
							dynamics_name=type(dynamics).__name__,
							method_name=type(self).__name__,
							step_index=step_index,
							start_time=t,
							time=t + step,
							duration=step,
							state_before=state_before.copy(),
							state_after=result.state.copy(),
							map_state=apply_step,
							dynamics=dynamics,
						)
					)
			if not shared_time_extension:
				return result.state
			assert extended_before is not None
			assert isinstance(dynamics, GuidingCenterDynamics)
			kappa_after = extended_before[3] + _shared_time_kappa_increment_from_stages(
				dynamics,
				t,
				step,
				result.stages,
			)
			return np.concatenate((result.state, (t + step, kappa_after)))

		initial_state = problem.initial_state
		if shared_time_extension:
			initial_state = np.concatenate(
				(initial_state, (float(request.t_span[0]), 0.0))
			)
		history, step_count = integrate_fixed_grid(
			initial_state,
			request,
			advance,
			progress=bool(self.progress),
			label=type(self).__name__,
		)
		diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
			"step_count": step_count,
			"copy_separation_norms": np.asarray(
				copy_separation_norms,
				dtype=float,
			),
			"projection_kind": "arithmetic_mean",
			"state_extension": self.state_extension,
			"vector_field_evaluations_per_step": 4,
		}
		diagnostics.update(
			_state_dimension_diagnostics(
				self.state_extension,
				particle_count=problem.initial_state.size // dynamics.state_dimension,
			)
		)
		diagnostics["nonlinear_unknown_dimension"] = 0
		if shared_time_extension:
			diagnostics.update(
				{
					"extended_time": np.asarray(history[2]),
					"extended_kappa": np.asarray(history[3]),
					"extended_momentum_normalization": "kappa_equals_k_over_2",
				}
			)
		return IntegrationData(
			t=request.output_times,
			states=np.asarray(history[:2] if shared_time_extension else history),
			diagnostics=diagnostics,
		)


__all__ = ["ABBA2Midpoint"]
