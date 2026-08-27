"""One reduced Hairer projection around an unprojected ABBA4 composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import GuidingCenterJacobianSystem

from .._fixed import integrate_fixed_grid
from .._result import IntegrationData
from ..observation import (
	ABBA4SingleProjectionIntegrationStep,
	UnprojectedABBAIntegrationStep,
)
from ..problem import InitialValueProblem
from ..request import SimulationRequest
from ._implicit_abba import _ImplicitABBA
from ._nonlinear import NonlinearSolver, _solve_broyden
from ._projected_abba import (
	_ABBAStages,
	_checked_vector_field_jacobian,
	_differentiate_stages,
	_evaluate_unprojected_stages,
)


_CUBE_ROOT_TWO = float(np.cbrt(2.0))
_GAMMA = 1.0 / (2.0 - _CUBE_ROOT_TWO)
_DELTA = -_CUBE_ROOT_TWO / (2.0 - _CUBE_ROOT_TWO)
_ABBA4_COEFFICIENTS = np.asarray((_GAMMA, _DELTA, _GAMMA), dtype=float)
_FORMULATION = "abba4_single_projection_implicit_1_reduced"
_BASE_COMPOSITION = "unprojected_abba4_triple_jump"


@dataclass(frozen=True, slots=True)
class _SingleProjectionBaseEvaluation:
	"""One complete unprojected ABBA4 map at a candidate multiplier."""

	u_final: np.ndarray
	v_final: np.ndarray
	residual: np.ndarray
	substeps: tuple[_ABBAStages, ...]


@dataclass(frozen=True, slots=True)
class _SingleProjectionResidualEvaluation:
	"""ABBA4 residual and its exact independent-particle Newton blocks."""

	u_final: np.ndarray
	v_final: np.ndarray
	residual: np.ndarray
	jacobian: np.ndarray
	base_jacobian: np.ndarray
	substeps: tuple[_ABBAStages, ...]


@dataclass(frozen=True, slots=True)
class _ABBA4SingleProjectionStep:
	"""Accepted physical state and one outer projection solve's diagnostics."""

	state: np.ndarray
	multiplier: np.ndarray
	substeps: tuple[_ABBAStages, ...]
	iterations: int
	residual_evaluations: int
	residual_norm: float


def _validated_state(
	dynamics: GuidingCenterJacobianSystem,
	state: np.ndarray,
) -> np.ndarray:
	"""Return one finite packed planar GC state."""
	value = np.asarray(state, dtype=float)
	if (
		value.ndim != 1
		or value.size == 0
		or value.size % dynamics.state_dimension
		or not np.all(np.isfinite(value))
	):
		raise ValueError(
			"The ABBA4 single-projection state must be a finite, non-empty "
			"packed GC vector."
		)
	return value


def _evaluate_single_projection_base(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	multiplier: np.ndarray,
) -> _SingleProjectionBaseEvaluation:
	"""Apply three continuous signed ABBA maps with no intermediate projection."""
	value = _validated_state(dynamics, state)
	mu = np.asarray(multiplier, dtype=float)
	if mu.shape != value.shape or not np.all(np.isfinite(mu)):
		raise ValueError(
			"The ABBA4 single-projection multiplier must match the physical state."
		)
	if not np.isfinite(t) or not np.isfinite(step):
		raise ValueError("The ABBA4 single-projection time and step must be finite.")

	# Hairer's normal embedding N=G^T displaces the two physical copies once,
	# before the whole fourth-order base composition.
	u_current = value + mu
	v_current = value - mu
	current_time = float(t)
	substeps: list[_ABBAStages] = []
	for coefficient in _ABBA4_COEFFICIENTS:
		duration = float(coefficient * step)
		stages = _evaluate_unprojected_stages(
			dynamics,
			current_time,
			u_current,
			v_current,
			duration,
		)
		substeps.append(stages)
		u_current = stages.u_final
		v_current = stages.v_final
		current_time += duration

	expected_time = float(t + step)
	time_tolerance = float(
		64.0
		* np.finfo(float).eps
		* max(1.0, abs(float(t)), abs(expected_time), abs(float(step)))
	)
	if not np.isclose(
		current_time,
		expected_time,
		rtol=0.0,
		atol=time_tolerance,
	):
		raise RuntimeError("The ABBA4 composition coefficients do not sum to one.")

	# The final normal correction is (+mu, -mu), so diagonal equality is
	# u_final - v_final + 2 mu = 0.
	residual = u_current - v_current + 2.0 * mu
	return _SingleProjectionBaseEvaluation(
		u_final=np.asarray(u_current),
		v_final=np.asarray(v_current),
		residual=np.asarray(residual),
		substeps=tuple(substeps),
	)


def _differentiate_single_projection_base(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	base: _SingleProjectionBaseEvaluation,
) -> _SingleProjectionResidualEvaluation:
	"""Form ``K = G M_3 M_2 M_1 N + 2 I`` from exact ABBA tangents."""
	value = _validated_state(dynamics, state)
	particle_count = value.size // dynamics.state_dimension
	identity_2 = np.broadcast_to(
		np.eye(dynamics.state_dimension),
		(particle_count, dynamics.state_dimension, dynamics.state_dimension),
	)
	base_jacobian = np.broadcast_to(
		np.eye(2 * dynamics.state_dimension),
		(
			particle_count,
			2 * dynamics.state_dimension,
			2 * dynamics.state_dimension,
		),
	).copy()
	current_time = float(t)
	for coefficient, stages in zip(
		_ABBA4_COEFFICIENTS,
		base.substeps,
		strict=True,
	):
		duration = float(coefficient * step)
		factor = _differentiate_stages(
			dynamics,
			current_time,
			value,
			duration,
			stages,
		).abba_jacobian
		base_jacobian = factor @ base_jacobian
		current_time += duration

	dimension = dynamics.state_dimension
	top_left = base_jacobian[:, :dimension, :dimension]
	top_right = base_jacobian[:, :dimension, dimension:]
	bottom_left = base_jacobian[:, dimension:, :dimension]
	bottom_right = base_jacobian[:, dimension:, dimension:]
	jacobian = (
		top_left
		- top_right
		- bottom_left
		+ bottom_right
		+ 2.0 * identity_2
	)
	if not np.all(np.isfinite(jacobian)):
		raise ValueError(
			"The ABBA4 single-projection residual Jacobian is non-finite."
		)
	return _SingleProjectionResidualEvaluation(
		u_final=base.u_final,
		v_final=base.v_final,
		residual=base.residual,
		jacobian=np.asarray(jacobian),
		base_jacobian=np.asarray(base_jacobian),
		substeps=base.substeps,
	)


def _evaluate_single_projection_residual(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	multiplier: np.ndarray,
) -> _SingleProjectionResidualEvaluation:
	"""Evaluate the reduced outer residual and its exact analytic Jacobian."""
	base = _evaluate_single_projection_base(
		dynamics,
		t,
		state,
		step,
		multiplier,
	)
	return _differentiate_single_projection_base(
		dynamics,
		t,
		state,
		step,
		base,
	)


def _projected_state(
	base: _SingleProjectionBaseEvaluation,
	multiplier: np.ndarray,
) -> np.ndarray:
	"""Return the neutral mean of the two symmetrically corrected copies."""
	first_copy = base.u_final + multiplier
	second_copy = base.v_final - multiplier
	return np.asarray((first_copy + second_copy) / 2.0)


def _solve_abba4_single_projection_step(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver = "newton",
) -> _ABBA4SingleProjectionStep:
	"""Solve one reduced multiplier around the complete unprojected ABBA4 map."""
	value = _validated_state(dynamics, state)
	multiplier = np.zeros_like(value)
	state_scale = max(1.0, float(np.linalg.norm(value, ord=np.inf)))
	threshold = absolute_tolerance + relative_tolerance * state_scale
	context = (
		"ABBA4 single-projection implicit formulation 1 at "
		f"t={t:.16g} with step={step:.16g}"
	)

	if nonlinear_solver == "broyden":
		def residual_function(
			candidate: np.ndarray,
		) -> tuple[np.ndarray, _SingleProjectionBaseEvaluation]:
			"""Evaluate the one outer projection residual for Broyden."""
			base = _evaluate_single_projection_base(
				dynamics,
				t,
				value,
				step,
				candidate,
			)
			return base.residual, base

		result = _solve_broyden(
			residual_function,
			multiplier,
			4.0 * np.eye(value.size),
			tolerance=threshold,
			max_iterations=max_iterations,
			context=context,
		)
		return _ABBA4SingleProjectionStep(
			state=_projected_state(result.payload, result.unknown),
			multiplier=result.unknown,
			substeps=result.payload.substeps,
			iterations=result.iterations,
			residual_evaluations=result.residual_evaluations,
			residual_norm=float(np.linalg.norm(result.residual, ord=np.inf)),
		)
	if nonlinear_solver != "newton":
		raise ValueError(
			"Unknown nonlinear solver for ABBA4 single-projection formulation 1."
		)

	for iteration in range(max_iterations + 1):
		base = _evaluate_single_projection_base(
			dynamics,
			t,
			value,
			step,
			multiplier,
		)
		residual_norm = float(np.linalg.norm(base.residual, ord=np.inf))
		if residual_norm <= threshold:
			return _ABBA4SingleProjectionStep(
				state=_projected_state(base, multiplier),
				multiplier=multiplier.copy(),
				substeps=base.substeps,
				iterations=iteration,
				residual_evaluations=iteration + 1,
				residual_norm=residual_norm,
			)
		if iteration == max_iterations:
			break

		evaluation = _differentiate_single_projection_base(
			dynamics,
			t,
			value,
			step,
			base,
		)
		residual_blocks = evaluation.residual.reshape(
			dynamics.state_dimension,
			-1,
		).T
		try:
			correction_blocks = np.linalg.solve(
				evaluation.jacobian,
				residual_blocks[..., None],
			)[..., 0]
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				"The ABBA4 single-projection Jacobian is singular at "
				f"t={t:.16g} with step={step:.16g}."
			) from exc
		multiplier = multiplier - correction_blocks.T.reshape(-1)

	raise RuntimeError(
		"ABBA4 single-projection implicit formulation 1 did not converge at "
		f"t={t:.16g} with step={step:.16g}: residual norm "
		f"{residual_norm:.3e} exceeds {threshold:.3e} after "
		f"{max_iterations} Newton iterations."
	)


def _observed_substeps(
	start_time: float,
	step: float,
	stages: tuple[_ABBAStages, ...],
) -> tuple[UnprojectedABBAIntegrationStep, ...]:
	"""Copy the three continuous unprojected maps into public snapshots."""
	current_time = float(start_time)
	result: list[UnprojectedABBAIntegrationStep] = []
	for coefficient, substep in zip(
		_ABBA4_COEFFICIENTS,
		stages,
		strict=True,
	):
		duration = float(coefficient * step)
		result.append(
			UnprojectedABBAIntegrationStep(
				start_time=current_time,
				time=current_time + duration,
				duration=duration,
				u_initial=substep.u_initial.copy(),
				v_initial=substep.v_initial.copy(),
				u_first=substep.u_first.copy(),
				v_final=substep.v_final.copy(),
				u_final=substep.u_final.copy(),
			)
		)
		current_time += duration
	return tuple(result)


def _integrate_abba4_single_projection_implicit_1(
	method: ABBA4SingleProjectionImplicit1,
	problem: InitialValueProblem,
	request: SimulationRequest,
) -> IntegrationData:
	"""Integrate with three unprojected ABBA maps and one outer solve per step."""
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

	iteration_counts: list[int] = []
	residual_evaluation_counts: list[int] = []
	residual_norms: list[float] = []
	tolerances: list[float] = []
	multiplier_norms: list[float] = []

	def solve_step(
		t: float,
		state: np.ndarray,
		step: float,
	) -> _ABBA4SingleProjectionStep:
		"""Apply the fixed-time single-projection step."""
		return _solve_abba4_single_projection_step(
			dynamics,
			t,
			state,
			step,
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
		result = solve_step(t, state_before, step)
		if observe:
			state_scale = max(1.0, float(np.linalg.norm(state_before, ord=np.inf)))
			tolerance = (
				method.newton_absolute_tolerance
				+ method.newton_relative_tolerance * state_scale
			)
			multiplier_norm = float(
				np.linalg.norm(result.multiplier, ord=np.inf)
			)
			iteration_counts.append(result.iterations)
			residual_evaluation_counts.append(result.residual_evaluations)
			residual_norms.append(result.residual_norm)
			tolerances.append(tolerance)
			multiplier_norms.append(multiplier_norm)
			if method.step_observer is not None:
				def map_state(candidate: np.ndarray) -> np.ndarray:
					"""Apply this same outer map to a diagnostic candidate."""
					return solve_step(t, candidate, step).state

				method.step_observer(
					ABBA4SingleProjectionIntegrationStep(
						dynamics_name=type(dynamics).__name__,
						method_name=method_name,
						step_index=step_index,
						time=t + step,
						duration=step,
						state_before=state_before.copy(),
						state_after=result.state.copy(),
						map_state=map_state,
						formulation_name=_FORMULATION,
						start_time=t,
						nonlinear_solver=method.nonlinear_solver,
						newton_iterations=result.iterations,
						residual_evaluations=result.residual_evaluations,
						newton_residual_norm=result.residual_norm,
						newton_tolerance=tolerance,
						projection_multiplier_norm=multiplier_norm,
						dynamics=dynamics,
						multiplier=result.multiplier.copy(),
						composition_coefficients=(
							_ABBA4_COEFFICIENTS.copy()
						),
						substeps=_observed_substeps(
							t,
							step,
							result.substeps,
						),
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
	iterations = np.asarray(iteration_counts, dtype=int)
	residual_evaluations = np.asarray(residual_evaluation_counts, dtype=int)
	residuals = np.asarray(residual_norms, dtype=float)
	tolerance_values = np.asarray(tolerances, dtype=float)
	multipliers = np.asarray(multiplier_norms, dtype=float)
	diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
		"step_count": step_count,
		"implicit_substeps_per_step": 1,
		"nonlinear_solves_per_step": 1,
		"unprojected_abba_maps_per_step": int(_ABBA4_COEFFICIENTS.size),
		"unprojected_abba_maps_per_residual_evaluation": int(
			_ABBA4_COEFFICIENTS.size
		),
		"composition_coefficients": _ABBA4_COEFFICIENTS.copy(),
		"base_composition": _BASE_COMPOSITION,
		"projection_placement": "around_complete_base_composition",
		"nonlinear_solver": method.nonlinear_solver,
		"nonlinear_iterations": iterations,
		"residual_evaluations": residual_evaluations,
		"nonlinear_residual_norms": residuals,
		"nonlinear_tolerances": tolerance_values,
		"projection_multiplier_norms": multipliers,
		# One-column forms preserve the composed-method comparison schema while
		# making the single nonlinear solve per outer step explicit.
		"substep_nonlinear_iterations": iterations[:, None],
		"substep_residual_evaluations": residual_evaluations[:, None],
		"substep_nonlinear_residual_norms": residuals[:, None],
		"substep_nonlinear_tolerances": tolerance_values[:, None],
		"substep_projection_multiplier_norms": multipliers[:, None],
		"nonlinear_absolute_tolerance": method.newton_absolute_tolerance,
		"nonlinear_relative_tolerance": method.newton_relative_tolerance,
		"nonlinear_max_iterations": method.newton_max_iterations,
		"newton_iterations": iterations,
		"newton_residual_norms": residuals,
		"newton_absolute_tolerance": method.newton_absolute_tolerance,
		"newton_relative_tolerance": method.newton_relative_tolerance,
		"newton_max_iterations": method.newton_max_iterations,
		"projection_solver_formulation": _FORMULATION,
	}
	return IntegrationData(
		t=request.output_times,
		states=np.asarray(history),
		diagnostics=diagnostics,
	)


@dataclass(frozen=True, slots=True)
class ABBA4SingleProjectionImplicit1(_ImplicitABBA):
	"""Fourth-order ABBA triple jump with one reduced symmetric projection.

	The signed ``(gamma h, delta h, gamma h)`` ABBA maps evolve two independent
	physical copies continuously. A single multiplier is solved around the whole
	composition, so no projection returns the copies to the diagonal between its
	three constituent maps.
	"""

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate a planar GC problem with one outer projection per step."""
		return _integrate_abba4_single_projection_implicit_1(
			self,
			problem,
			request,
		)


__all__ = ["ABBA4SingleProjectionImplicit1"]
