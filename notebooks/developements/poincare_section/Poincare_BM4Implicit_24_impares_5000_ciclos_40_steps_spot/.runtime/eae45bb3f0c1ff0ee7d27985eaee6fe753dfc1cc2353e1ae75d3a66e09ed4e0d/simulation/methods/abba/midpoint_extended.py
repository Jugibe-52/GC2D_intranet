"""Fully extended arithmetic-mean ABBA2, outside the implicit solver family."""
from __future__ import annotations
from typing import Protocol
import numpy as np
from dynamics import GuidingCenterDynamics
from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
from ...observation import IntegrationStep, StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from .maps.extended import _abba_base_map, _checked_extended_state, _synchronized_extended_time
from .state import _fully_extended_energy_diagnostics

class _ABBAFullyExtendedMidpointMethod(Protocol):
	"""Configuration consumed by the full-state arithmetic-mean runtime."""

	@property
	def progress(self) -> bool: ...

	@property
	def step_observer(self) -> StepObserver | None: ...


def _integrate_abba_fully_extended_midpoint(
	method: _ABBAFullyExtendedMidpointMethod,
	problem: InitialValueProblem,
	request: SimulationRequest,
) -> IntegrationData:
	"""Integrate full-state ABBA2 with arithmetic-mean diagonal projection."""
	method_name = type(method).__name__
	if not isinstance(problem.dynamics, GuidingCenterDynamics):
		raise TypeError(f"{method_name} requires GuidingCenterDynamics.")
	physical_initial = np.asarray(problem.initial_state, dtype=float)
	if physical_initial.shape != (2,):
		raise ValueError(f"{method_name} requires exactly one GC particle.")
	dynamics = problem.dynamics
	initial_extended = np.concatenate(
		(physical_initial, (float(request.t_span[0]), 0.0))
	)
	copy_separation_norms: list[float] = []

	def midpoint_step(state: np.ndarray, step: float) -> tuple[np.ndarray, float]:
		value = _checked_extended_state(state, duplicated=False)
		mapped = np.asarray(
			_abba_base_map(dynamics, step).map_state(np.concatenate((value, value))),
			dtype=float,
		)
		accepted_state = _synchronized_extended_time(
			np.asarray((mapped[:4] + mapped[4:]) / 2.0),
			float(value[2] + step),
			context="The fully extended midpoint map",
		)
		return (
			accepted_state,
			float(np.linalg.norm(mapped[:4] - mapped[4:], ord=np.inf)),
		)

	def advance(
		time: float,
		state: np.ndarray,
		step: float,
		step_index: int,
		observe: bool,
	) -> np.ndarray:
		value = _synchronized_extended_time(
			state,
			time,
			context="The internal state",
		)
		state_after, separation = midpoint_step(value, step)
		if observe:
			copy_separation_norms.append(separation)
			if method.step_observer is not None:
				def map_state(candidate: np.ndarray) -> np.ndarray:
					return midpoint_step(candidate, step)[0]

				method.step_observer(
					IntegrationStep(
						dynamics_name=type(dynamics).__name__,
						method_name=method_name,
						step_index=step_index,
						start_time=time,
						time=time + step,
						duration=step,
						state_before=value.copy(),
						state_after=state_after.copy(),
						map_state=map_state,
						dynamics=dynamics,
					)
				)
		return state_after

	extended_history, step_count = integrate_fixed_grid(
		initial_extended,
		request,
		advance,
		progress=bool(method.progress),
		label=method_name,
	)
	extended_history[2] = request.output_times
	diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
		"step_count": step_count,
		"copy_separation_norms": np.asarray(copy_separation_norms, dtype=float),
		"projection_kind": "arithmetic_mean",
		"state_extension": "fully_extended",
		"track_energy": True,
		"accepted_internal_state_dimension": 4,
		"base_splitting_state_dimension": 8,
		"observer_state_dimension": 4,
		"observer_state_kind": "accepted_internal_map",
		"nonlinear_unknown_dimension": 0,
		"unprojected_abba_maps_per_step": 1,
		"vector_field_evaluations_per_step": 4,
	}
	diagnostics.update(_fully_extended_energy_diagnostics(dynamics, extended_history))
	return IntegrationData(
		t=request.output_times,
		states=np.asarray(extended_history[:2]),
		diagnostics=diagnostics,
	)


__all__: list[str] = []
