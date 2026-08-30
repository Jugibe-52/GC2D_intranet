"""One configurable projection around an unprojected ABBA4 composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import GuidingCenterDynamics, GuidingCenterJacobianSystem

from .._fully_extended import _integrate_abba_fully_extended
from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
from ...observation import (
	ABBA4ImplicitSingleProjectionIntegrationStep,
	UnprojectedABBAIntegrationStep,
)
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from .._nonlinear import NonlinearSolver, _solve_broyden
from ._coefficients import _ABBA4_COEFFICIENTS
from ._configuration import (
	ProjectionFormulation,
	_state_dimension_diagnostics,
)
from ._core import _ABBAStages, _evaluate_unprojected_stages
from ._implicit import (
	_ABBAImplicitConfig,
	_shared_time_kappa_increment_from_stages,
)
from ._projection_common import (
	_checked_vector_field_jacobian,
	_differentiate_stages,
)


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


def _solve_reduced_abba4_single_projection_step(
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
		"ABBA4 single reduced-multiplier projection at "
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
		raise ValueError("Unknown nonlinear solver for ABBA4 single projection.")

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
		"ABBA4 single reduced-multiplier projection did not converge at "
		f"t={t:.16g} with step={step:.16g}: residual norm "
		f"{residual_norm:.3e} exceeds {threshold:.3e} after "
		f"{max_iterations} Newton iterations."
	)


def _simultaneous_residual_blocks(
	base: _SingleProjectionBaseEvaluation,
	multiplier: np.ndarray,
	first_output: np.ndarray,
	second_output: np.ndarray,
	state_dimension: int,
) -> np.ndarray:
	"""Return particle-major output, multiplier, and diagonal defects."""
	first_defect = first_output - multiplier - base.u_final
	second_defect = second_output + multiplier - base.v_final
	constraint = first_output - second_output
	particle_blocks = lambda vector: vector.reshape(state_dimension, -1).T
	return np.concatenate(
		(
			particle_blocks(first_defect),
			particle_blocks(second_defect),
			particle_blocks(constraint),
		),
		axis=-1,
	)


def _simultaneous_newton_jacobian(
	evaluation: _SingleProjectionResidualEvaluation,
) -> np.ndarray:
	"""Assemble exact simultaneous blocks around the complete ABBA4 base map."""
	particle_count = evaluation.base_jacobian.shape[0]
	identity_2 = np.broadcast_to(np.eye(2), (particle_count, 2, 2))
	identity_4 = np.broadcast_to(np.eye(4), (particle_count, 4, 4))
	zero_2 = np.zeros((particle_count, 2, 2), dtype=float)
	normal = np.concatenate((identity_2, -identity_2), axis=-2)
	constraint = np.concatenate((identity_2, -identity_2), axis=-1)
	top_right = -(identity_4 + evaluation.base_jacobian) @ normal
	return np.concatenate(
		(
			np.concatenate((identity_4, top_right), axis=-1),
			np.concatenate((constraint, zero_2), axis=-1),
		),
		axis=-2,
	)


def _solve_simultaneous_abba4_single_projection_step(
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
	"""Solve output copies and one multiplier around the complete ABBA4 map."""
	value = _validated_state(dynamics, state)
	multiplier = np.zeros_like(value)
	base = _evaluate_single_projection_base(dynamics, t, value, step, multiplier)
	first_output = base.u_final.copy()
	second_output = base.v_final.copy()
	threshold = absolute_tolerance + relative_tolerance * max(
		1.0,
		float(np.linalg.norm(value, ord=np.inf)),
	)
	context = (
		"ABBA4 single simultaneous state-multiplier projection at "
		f"t={t:.16g} with step={step:.16g}"
	)

	if nonlinear_solver == "broyden":
		physical_size = value.size
		internal_size = 2 * physical_size
		identity_internal = np.eye(internal_size)
		identity_physical = np.eye(physical_size)
		normal = np.concatenate((identity_physical, -identity_physical), axis=0)
		constraint = np.concatenate((identity_physical, -identity_physical), axis=1)
		initial_jacobian = np.block(
			[
				[identity_internal, -2.0 * normal],
				[constraint, np.zeros((physical_size, physical_size))],
			]
		)

		def residual_function(
			unknown: np.ndarray,
		) -> tuple[
			np.ndarray,
			tuple[_SingleProjectionBaseEvaluation, np.ndarray, np.ndarray, np.ndarray],
		]:
			first = unknown[:physical_size]
			second = unknown[physical_size:internal_size]
			candidate_multiplier = unknown[internal_size:]
			candidate_base = _evaluate_single_projection_base(
				dynamics,
				t,
				value,
				step,
				candidate_multiplier,
			)
			residual = np.concatenate(
				(
					first - candidate_multiplier - candidate_base.u_final,
					second + candidate_multiplier - candidate_base.v_final,
					first - second,
				)
			)
			return residual, (candidate_base, first, second, candidate_multiplier)

		result = _solve_broyden(
			residual_function,
			np.concatenate((first_output, second_output, multiplier)),
			initial_jacobian,
			tolerance=threshold,
			max_iterations=max_iterations,
			context=context,
			initial_evaluation=(
				np.concatenate(
					(
						np.zeros(internal_size),
						first_output - second_output,
					)
				),
				(base, first_output, second_output, multiplier),
			),
		)
		base, first_output, second_output, multiplier = result.payload
		return _ABBA4SingleProjectionStep(
			state=np.asarray((first_output + second_output) / 2.0),
			multiplier=np.asarray(multiplier).copy(),
			substeps=base.substeps,
			iterations=result.iterations,
			residual_evaluations=result.residual_evaluations,
			residual_norm=float(np.linalg.norm(result.residual, ord=np.inf)),
		)
	if nonlinear_solver != "newton":
		raise ValueError("Unknown nonlinear solver for ABBA4 single projection.")

	for iteration in range(max_iterations + 1):
		residual_blocks = _simultaneous_residual_blocks(
			base,
			multiplier,
			first_output,
			second_output,
			dynamics.state_dimension,
		)
		residual_norm = float(np.max(np.abs(residual_blocks)))
		if residual_norm <= threshold:
			return _ABBA4SingleProjectionStep(
				state=np.asarray((first_output + second_output) / 2.0),
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
		try:
			increments = np.linalg.solve(
				_simultaneous_newton_jacobian(evaluation),
				-residual_blocks[..., None],
			)[..., 0]
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				"The simultaneous ABBA4 single-projection Jacobian is singular at "
				f"t={t:.16g} with step={step:.16g}."
			) from exc
		first_output = first_output + increments[..., :2].T.reshape(-1)
		second_output = second_output + increments[..., 2:4].T.reshape(-1)
		multiplier = multiplier + increments[..., 4:].T.reshape(-1)
		base = _evaluate_single_projection_base(
			dynamics,
			t,
			value,
			step,
			multiplier,
		)

	raise RuntimeError(
		f"{context} did not converge: residual {residual_norm:.3e} exceeds "
		f"{threshold:.3e} after {max_iterations} iterations."
	)


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
	projection_formulation: ProjectionFormulation = "reduced_multiplier",
) -> _ABBA4SingleProjectionStep:
	"""Select one formulation around the complete unprojected ABBA4 map."""
	solver = (
		_solve_reduced_abba4_single_projection_step
		if projection_formulation == "reduced_multiplier"
		else _solve_simultaneous_abba4_single_projection_step
	)
	return solver(
		dynamics,
		t,
		state,
		step,
		absolute_tolerance=absolute_tolerance,
		relative_tolerance=relative_tolerance,
		max_iterations=max_iterations,
		nonlinear_solver=nonlinear_solver,
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


def _integrate_abba4_implicit_single_projection(
	method: ABBA4ImplicitSingleProjection,
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
	shared_time_extension = method.state_extension == "shared_time"
	if shared_time_extension:
		if not isinstance(dynamics, GuidingCenterDynamics):
			raise TypeError(
				f"{method_name} requires GuidingCenterDynamics for shared_time."
			)
		if np.asarray(problem.initial_state).shape != (2,):
			raise ValueError(
				f"{method_name} requires exactly one GC particle for shared_time."
			)
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
			projection_formulation=method.projection_formulation,
		)

	def advance(
		t: float,
		state: np.ndarray,
		step: float,
		step_index: int,
		observe: bool,
	) -> np.ndarray:
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
					ABBA4ImplicitSingleProjectionIntegrationStep(
						dynamics_name=type(dynamics).__name__,
						method_name=method_name,
						step_index=step_index,
						time=t + step,
						duration=step,
						state_before=state_before.copy(),
						state_after=result.state.copy(),
						map_state=map_state,
						formulation_name=method.projection_formulation,
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
		if not shared_time_extension:
			return result.state
		assert extended_before is not None
		assert isinstance(dynamics, GuidingCenterDynamics)
		kappa_after = float(extended_before[3])
		current_time = float(t)
		for coefficient, stages in zip(
			_ABBA4_COEFFICIENTS,
			result.substeps,
			strict=True,
		):
			duration = float(coefficient * step)
			kappa_after += _shared_time_kappa_increment_from_stages(
				dynamics,
				current_time,
				duration,
				stages,
			)
			current_time += duration
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
		"projection_formulation": method.projection_formulation,
		"state_extension": method.state_extension,
	}
	diagnostics.update(
		_state_dimension_diagnostics(
			method.state_extension,
			method.projection_formulation,
			particle_count=problem.initial_state.size // dynamics.state_dimension,
		)
	)
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


@dataclass(frozen=True, slots=True)
class ABBA4ImplicitSingleProjection(_ABBAImplicitConfig):
	"""Fourth-order ABBA triple jump with one symmetric outer projection.

	The signed ``(gamma h, delta h, gamma h)`` ABBA maps evolve two independent
	physical copies continuously. A single multiplier is solved around the whole
	composition, so no projection returns the copies to the diagonal between its
	three constituent maps. Both projection formulations, both nonlinear solvers,
	and all three state extensions apply to this one outer projection.
	"""

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate a planar GC problem with one outer projection per step."""
		if self.state_extension == "fully_extended":
			return _integrate_abba_fully_extended(
				self,
				problem,
				request,
				variant="abba4_single_projection",
				projection_formulation=self.projection_formulation,
			)
		return _integrate_abba4_implicit_single_projection(
			self,
			problem,
			request,
		)


__all__ = ["ABBA4ImplicitSingleProjection"]
