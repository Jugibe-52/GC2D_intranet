"""Fourth-order triple-jump composition of reduced implicit ABBA steps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import GuidingCenterJacobianSystem

from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
from ...observation import (
	ImplicitABBA4IntegrationStep,
	ImplicitABBACompositionIntegrationStep,
	ImplicitABBAIntegrationStep,
)
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from .._nonlinear import NonlinearSolver
from ._implicit import _ImplicitABBA
from ._projection import (
	_ProjectedStep,
	_checked_vector_field_jacobian,
	_solve_projected_step,
)


_CUBE_ROOT_TWO = float(np.cbrt(2.0))
_GAMMA = 1.0 / (2.0 - _CUBE_ROOT_TWO)
_DELTA = -_CUBE_ROOT_TWO / (2.0 - _CUBE_ROOT_TWO)
_ABBA4_COEFFICIENTS = np.asarray((_GAMMA, _DELTA, _GAMMA), dtype=float)
_SUBSTEP_FORMULATION = "reduced_multiplier"
_COMPOSITION_FORMULATION = "abba4_implicit_1_triple_jump"


@dataclass(frozen=True, slots=True)
class _AcceptedSubstep:
	"""One signed projected-ABBA solve inside an accepted outer step."""

	start_time: float
	duration: float
	state_before: np.ndarray
	result: _ProjectedStep


@dataclass(frozen=True, slots=True)
class _ComposedABBAStep:
	"""Final physical state and the accepted nonlinear composition substeps."""

	state: np.ndarray
	substeps: tuple[_AcceptedSubstep, ...]


def _solve_composed_abba_step(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	coefficients: np.ndarray,
	method_name: str,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver,
) -> _ComposedABBAStep:
	"""Compose complete reduced implicit-ABBA maps with signed durations."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError(
			f"The {method_name} physical state must be a finite, non-empty vector."
		)
	composition = np.asarray(coefficients, dtype=float)
	if (
		composition.ndim != 1
		or composition.size == 0
		or not np.all(np.isfinite(composition))
	):
		raise ValueError("ABBA composition coefficients must be finite and non-empty.")
	current_time = float(t)
	current_state = value
	accepted: list[_AcceptedSubstep] = []
	for coefficient in composition:
		duration = float(coefficient * step)
		state_before = np.asarray(current_state, dtype=float)
		result = _solve_projected_step(
			dynamics,
			current_time,
			state_before,
			duration,
			absolute_tolerance=absolute_tolerance,
			relative_tolerance=relative_tolerance,
			max_iterations=max_iterations,
			nonlinear_solver=nonlinear_solver,
		)
		accepted.append(
			_AcceptedSubstep(
				start_time=current_time,
				duration=duration,
				state_before=state_before.copy(),
				result=result,
			)
		)
		current_state = result.state
		current_time += duration

	expected_time = float(t + step)
	tolerance = float(
		64.0
		* np.finfo(float).eps
		* max(1.0, abs(float(t)), abs(expected_time), abs(float(step)))
	)
	if not np.isclose(current_time, expected_time, rtol=0.0, atol=tolerance):
		raise RuntimeError(
			f"The {method_name} composition coefficients do not sum to one."
		)
	return _ComposedABBAStep(
		state=np.asarray(current_state),
		substeps=tuple(accepted),
	)


def _solve_abba4_step(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver,
) -> _ComposedABBAStep:
	"""Compose the three signed reduced implicit-ABBA maps of ABBA4."""
	return _solve_composed_abba_step(
		dynamics,
		t,
		state,
		step,
		coefficients=_ABBA4_COEFFICIENTS,
		method_name="ABBA4Implicit1",
		absolute_tolerance=absolute_tolerance,
		relative_tolerance=relative_tolerance,
		max_iterations=max_iterations,
		nonlinear_solver=nonlinear_solver,
	)


def _substep_observation(
	*,
	dynamics: GuidingCenterJacobianSystem,
	method_name: str,
	step_index: int,
	accepted: _AcceptedSubstep,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver,
) -> ImplicitABBAIntegrationStep:
	"""Build one immutable diagnostic snapshot for a composed signed substep."""
	state_before = accepted.state_before
	result = accepted.result
	start_time = accepted.start_time
	duration = accepted.duration

	def map_state(candidate: np.ndarray) -> np.ndarray:
		"""Apply this fixed-time signed projected-ABBA substep."""
		return _solve_projected_step(
			dynamics,
			start_time,
			candidate,
			duration,
			absolute_tolerance=absolute_tolerance,
			relative_tolerance=relative_tolerance,
			max_iterations=max_iterations,
			nonlinear_solver=nonlinear_solver,
		).state

	state_scale = max(1.0, float(np.linalg.norm(state_before, ord=np.inf)))
	tolerance = absolute_tolerance + relative_tolerance * state_scale
	multiplier_norm = float(np.linalg.norm(result.multiplier, ord=np.inf))
	return ImplicitABBAIntegrationStep(
		dynamics_name=type(dynamics).__name__,
		method_name=method_name,
		step_index=step_index,
		time=start_time + duration,
		duration=duration,
		state_before=state_before.copy(),
		state_after=result.state.copy(),
		map_state=map_state,
		formulation_name=_SUBSTEP_FORMULATION,
		start_time=start_time,
		nonlinear_solver=nonlinear_solver,
		newton_iterations=result.iterations,
		residual_evaluations=result.residual_evaluations,
		newton_residual_norm=result.residual_norm,
		newton_tolerance=tolerance,
		projection_multiplier_norm=multiplier_norm,
		dynamics=dynamics,
		multiplier=result.multiplier.copy(),
		u_initial=result.stages.u_initial.copy(),
		v_initial=result.stages.v_initial.copy(),
		u_first=result.stages.u_first.copy(),
		v_final=result.stages.v_final.copy(),
		u_final=result.stages.u_final.copy(),
	)


def _integrate_composed_implicit_abba(
	method: _ImplicitABBA,
	problem: InitialValueProblem,
	request: SimulationRequest,
	*,
	coefficients: np.ndarray,
	composition_formulation: str,
	observation_type: type[ImplicitABBACompositionIntegrationStep],
) -> IntegrationData:
	"""Run one symmetric ABBA composition and aggregate its nonlinear solves."""
	dynamics = problem.dynamics
	method_name = type(method).__name__
	if not isinstance(dynamics, GuidingCenterJacobianSystem):
		raise TypeError(f"{method_name} requires GuidingCenterJacobianSystem.")
	if dynamics.state_dimension != 2:
		raise TypeError(f"{method_name} requires planar two-component dynamics.")
	if method.nonlinear_solver == "newton":
		_checked_vector_field_jacobian(
			dynamics,
			request.t_span[0],
			problem.initial_state,
		)

	iteration_rows: list[list[int]] = []
	residual_evaluation_rows: list[list[int]] = []
	residual_norm_rows: list[list[float]] = []
	tolerance_rows: list[list[float]] = []
	multiplier_norm_rows: list[list[float]] = []

	def solve_step(t: float, state: np.ndarray, step: float) -> _ComposedABBAStep:
		"""Solve one fixed-time outer composition."""
		return _solve_composed_abba_step(
			dynamics,
			t,
			state,
			step,
			coefficients=coefficients,
			method_name=method_name,
			absolute_tolerance=method.newton_absolute_tolerance,
			relative_tolerance=method.newton_relative_tolerance,
			max_iterations=method.newton_max_iterations,
			nonlinear_solver=method.nonlinear_solver,
		)

	def advance(
		t: float,
		state: np.ndarray,
		step: float,
		step_index: int,
		observe: bool,
	) -> np.ndarray:
		state_before = np.asarray(state, dtype=float)

		def map_state(candidate: np.ndarray) -> np.ndarray:
			"""Apply the same complete composed map to a diagnostic state."""
			return solve_step(t, candidate, step).state

		result = solve_step(t, state_before, step)
		if observe:
			substeps = tuple(
				_substep_observation(
					dynamics=dynamics,
					method_name=method_name,
					step_index=step_index,
					accepted=accepted,
					absolute_tolerance=method.newton_absolute_tolerance,
					relative_tolerance=method.newton_relative_tolerance,
					max_iterations=method.newton_max_iterations,
					nonlinear_solver=method.nonlinear_solver,
				)
				for accepted in result.substeps
			)
			iterations = [substep.newton_iterations for substep in substeps]
			residual_evaluations = [
				substep.residual_evaluations for substep in substeps
			]
			residual_norms = [substep.newton_residual_norm for substep in substeps]
			tolerances = [substep.newton_tolerance for substep in substeps]
			multiplier_norms = [
				substep.projection_multiplier_norm for substep in substeps
			]
			iteration_rows.append(iterations)
			residual_evaluation_rows.append(residual_evaluations)
			residual_norm_rows.append(residual_norms)
			tolerance_rows.append(tolerances)
			multiplier_norm_rows.append(multiplier_norms)
			if method.step_observer is not None:
				worst_substep = int(
					np.argmax(np.asarray(residual_norms) / np.asarray(tolerances))
				)
				method.step_observer(
					observation_type(
						dynamics_name=type(dynamics).__name__,
						method_name=method_name,
						step_index=step_index,
						time=t + step,
						duration=step,
						state_before=state_before.copy(),
						state_after=result.state.copy(),
						map_state=map_state,
						formulation_name=composition_formulation,
						start_time=t,
						nonlinear_solver=method.nonlinear_solver,
						newton_iterations=sum(iterations),
						residual_evaluations=sum(residual_evaluations),
						newton_residual_norm=residual_norms[worst_substep],
						newton_tolerance=tolerances[worst_substep],
						projection_multiplier_norm=max(multiplier_norms),
						dynamics=dynamics,
						composition_coefficients=coefficients.copy(),
						substeps=substeps,
					)
				)
		return result.state

	history, step_count = integrate_fixed_grid(
		problem.initial_state,
		request,
		advance,
		progress=bool(method.progress),
		label=method_name,
	)
	iterations = np.asarray(iteration_rows, dtype=int)
	residual_evaluations = np.asarray(residual_evaluation_rows, dtype=int)
	residual_norms = np.asarray(residual_norm_rows, dtype=float)
	tolerances = np.asarray(tolerance_rows, dtype=float)
	multiplier_norms = np.asarray(multiplier_norm_rows, dtype=float)
	diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
		"step_count": step_count,
		"implicit_substeps_per_step": int(coefficients.size),
		"nonlinear_solves_per_step": int(coefficients.size),
		"composition_coefficients": coefficients.copy(),
		"nonlinear_solver": method.nonlinear_solver,
		"nonlinear_iterations": np.sum(iterations, axis=1),
		"residual_evaluations": np.sum(residual_evaluations, axis=1),
		"nonlinear_residual_norms": np.max(residual_norms, axis=1),
		"nonlinear_tolerances": np.max(tolerances, axis=1),
		"projection_multiplier_norms": np.max(multiplier_norms, axis=1),
		"substep_nonlinear_iterations": iterations,
		"substep_residual_evaluations": residual_evaluations,
		"substep_nonlinear_residual_norms": residual_norms,
		"substep_nonlinear_tolerances": tolerances,
		"substep_projection_multiplier_norms": multiplier_norms,
		"nonlinear_absolute_tolerance": method.newton_absolute_tolerance,
		"nonlinear_relative_tolerance": method.newton_relative_tolerance,
		"nonlinear_max_iterations": method.newton_max_iterations,
		"newton_iterations": np.sum(iterations, axis=1),
		"newton_residual_norms": np.max(residual_norms, axis=1),
		"newton_absolute_tolerance": method.newton_absolute_tolerance,
		"newton_relative_tolerance": method.newton_relative_tolerance,
		"newton_max_iterations": method.newton_max_iterations,
		"projection_solver_formulation": composition_formulation,
		"substep_projection_solver_formulation": _SUBSTEP_FORMULATION,
	}
	return IntegrationData(
		t=request.output_times,
		states=np.asarray(history),
		diagnostics=diagnostics,
	)


def _integrate_abba4_implicit_1(
	method: ABBA4Implicit1,
	problem: InitialValueProblem,
	request: SimulationRequest,
) -> IntegrationData:
	"""Run the fourth-order composition and aggregate three solves per step."""
	return _integrate_composed_implicit_abba(
		method,
		problem,
		request,
		coefficients=_ABBA4_COEFFICIENTS,
		composition_formulation=_COMPOSITION_FORMULATION,
		observation_type=ImplicitABBA4IntegrationStep,
	)


@dataclass(frozen=True, slots=True)
class ABBA4Implicit1(_ImplicitABBA):
	"""Fourth-order symmetric composition of three reduced implicit ABBA maps.

	One complete step applies signed substeps ``(gamma h, delta h, gamma h)``;
	the middle substep runs backward in time. Every substep solves its own reduced
	projection multiplier equation with exact ``2 x 2`` Newton blocks or with
	good Broyden updates.
	"""

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate a planar GC problem with the fourth-order composition."""
		return _integrate_abba4_implicit_1(self, problem, request)


__all__ = ["ABBA4Implicit1"]
