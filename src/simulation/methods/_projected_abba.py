"""ABBA integration with Hairer's symmetric projection for GC dynamics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dynamics import GuidingCenterJacobianSystem

from .._fixed import integrate_fixed_grid
from .._result import IntegrationData
from ..observation import IntegrationStep, StepObserver
from ..problem import InitialValueProblem
from ..request import SimulationRequest


@dataclass(frozen=True, slots=True)
class _ProjectedStep:
	"""Converged physical state and nonlinear-solve diagnostics for one step."""

	state: np.ndarray
	multiplier: np.ndarray
	ideal_state_jacobian: np.ndarray | None
	iterations: int
	residual_norm: float


@dataclass(frozen=True, slots=True)
class _ResidualEvaluation:
	"""ABBA stages and the exact reduced residual Jacobian at one multiplier."""

	u_final: np.ndarray
	v_final: np.ndarray
	residual: np.ndarray
	jacobian: np.ndarray
	abba_jacobian: np.ndarray


@dataclass(frozen=True, slots=True)
class _ABBAStages:
	"""State points traversed by one explicit endpoint-time ABBA map."""

	u_initial: np.ndarray
	v_initial: np.ndarray
	u_first: np.ndarray
	v_final: np.ndarray
	u_final: np.ndarray
	residual: np.ndarray


def _positive_finite(value: float, name: str) -> float:
	"""Normalize a strictly positive finite solver parameter."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be positive and finite.")
	result = float(value)
	if not np.isfinite(result) or result <= 0:
		raise ValueError(f"`{name}` must be positive and finite.")
	return result


def _positive_integer(value: int, name: str) -> int:
	"""Normalize a strictly positive integer solver parameter."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 1
	):
		raise ValueError(f"`{name}` must be a positive integer.")
	return int(value)


def _checked_vector_field(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate one GC vector field without allowing a layout change."""
	result = np.asarray(dynamics.vector_field(t, state), dtype=float)
	if result.shape != state.shape or not np.all(np.isfinite(result)):
		raise ValueError("The GC vector field changed shape or became non-finite.")
	return result


def _checked_vector_field_jacobian(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate batched exact GC Jacobians with one matrix per particle."""
	result = np.asarray(
		dynamics.particle_vector_field_jacobians(t, state),
		dtype=float,
	)
	particle_count = state.size // dynamics.state_dimension
	expected_shape = (particle_count, dynamics.state_dimension, dynamics.state_dimension)
	if result.shape != expected_shape or not np.all(np.isfinite(result)):
		raise ValueError(
			"The GC vector-field Jacobian changed shape or became non-finite."
		)
	return result


def _evaluate_stages(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	multiplier: np.ndarray,
) -> _ABBAStages:
	"""Apply the four explicit ABBA stages and evaluate equation (11)."""
	half_step = step / 2.0
	final_time = t + step

	# G^T mu displaces the two copies in opposite physical directions.
	u_initial = state + multiplier
	v_initial = state - multiplier
	u_first = u_initial + half_step * _checked_vector_field(
		dynamics,
		t,
		v_initial,
	)

	v_first = v_initial + half_step * _checked_vector_field(
		dynamics,
		t,
		u_first,
	)

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

	residual = u_final - v_final + 2.0 * multiplier
	return _ABBAStages(
		u_initial=u_initial,
		v_initial=v_initial,
		u_first=u_first,
		v_final=v_final,
		u_final=u_final,
		residual=residual,
	)


def _differentiate_stages(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	stages: _ABBAStages,
) -> _ResidualEvaluation:
	"""Evaluate equations (15) and (17) at the traversed ABBA stage points."""
	half_step = step / 2.0
	final_time = t + step
	w_1 = _checked_vector_field_jacobian(dynamics, t, stages.v_initial)
	w_2 = _checked_vector_field_jacobian(dynamics, t, stages.u_first)
	w_3 = _checked_vector_field_jacobian(dynamics, final_time, stages.u_first)
	w_4 = _checked_vector_field_jacobian(dynamics, final_time, stages.v_final)
	particle_count = state.size // dynamics.state_dimension
	identity = np.broadcast_to(
		np.eye(dynamics.state_dimension),
		(particle_count, dynamics.state_dimension, dynamics.state_dimension),
	)
	central_jacobian = w_2 + w_3
	# Matrix products retain their order because the stage Jacobians need not commute.
	jacobian = (
		4.0 * identity
		- half_step * (w_1 + w_2 + w_3 + w_4)
		+ half_step**2
		* (w_4 @ central_jacobian + central_jacobian @ w_1)
		- half_step**3 * (w_4 @ central_jacobian @ w_1)
	)
	top_left = identity + half_step**2 * (w_4 @ central_jacobian)
	top_right = (
		half_step * (w_1 + w_4)
		+ half_step**3 * (w_4 @ central_jacobian @ w_1)
	)
	bottom_left = half_step * central_jacobian
	bottom_right = identity + half_step**2 * (central_jacobian @ w_1)
	abba_jacobian = np.concatenate(
		(
			np.concatenate((top_left, top_right), axis=-1),
			np.concatenate((bottom_left, bottom_right), axis=-1),
		),
		axis=-2,
	)
	return _ResidualEvaluation(
		u_final=stages.u_final,
		v_final=stages.v_final,
		residual=stages.residual,
		jacobian=jacobian,
		abba_jacobian=abba_jacobian,
	)


def _evaluate_residual(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	multiplier: np.ndarray,
) -> _ResidualEvaluation:
	"""Apply ABBA and evaluate its exact residual and map Jacobians."""
	stages = _evaluate_stages(dynamics, t, state, step, multiplier)
	return _differentiate_stages(dynamics, t, state, step, stages)


def _dense_component_major_jacobian(blocks: np.ndarray) -> np.ndarray:
	"""Expand independent particle Jacobians into the packed physical layout."""
	particle_count = blocks.shape[0]
	result = np.zeros((2 * particle_count, 2 * particle_count), dtype=float)
	for particle in range(particle_count):
		indices = (particle, particle_count + particle)
		result[np.ix_(indices, indices)] = blocks[particle]
	return result


def _ideal_projected_state_jacobian(
	evaluation: _ResidualEvaluation,
) -> np.ndarray:
	"""Differentiate the exact-root projected map by the implicit theorem.

	This is the tangent of the mathematical method defined by ``R(mu) = 0``. It
	does not differentiate a finite Newton iteration or its stopping decision.
	The implementation evaluates ``P - Q solve(K, L)`` without forming an inverse.
	"""
	particle_count = evaluation.jacobian.shape[0]
	identity = np.broadcast_to(np.eye(2), (particle_count, 2, 2))
	abba_jacobian = evaluation.abba_jacobian
	top_left = abba_jacobian[..., :2, :2]
	top_right = abba_jacobian[..., :2, 2:]
	bottom_left = abba_jacobian[..., 2:, :2]
	bottom_right = abba_jacobian[..., 2:, 2:]
	# Equations (5)--(7) of ABBA_semiimplicit in block form.
	residual_state_jacobian = (
		top_left + top_right - bottom_left - bottom_right
	)
	direct_state_jacobian = top_left + top_right
	implicit_weight = top_left - top_right + identity
	try:
		residual_response = np.linalg.solve(
			evaluation.jacobian,
			residual_state_jacobian,
		)
	except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				"The ABBA projection Jacobian is singular while differentiating the step."
			) from exc
	physical_blocks = direct_state_jacobian - implicit_weight @ residual_response
	return _dense_component_major_jacobian(physical_blocks)


def _solve_projected_step(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	compute_ideal_state_jacobian: bool = True,
) -> _ProjectedStep:
	"""Solve implicit formulation 1 with exact reduced Newton steps."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The ABBA physical state must be a finite, non-empty vector.")
	multiplier = np.zeros_like(value)
	state_scale = max(1.0, float(np.linalg.norm(value, ord=np.inf)))
	threshold = absolute_tolerance + relative_tolerance * state_scale

	for iteration in range(max_iterations + 1):
		stages = _evaluate_stages(
			dynamics,
			t,
			value,
			step,
			multiplier,
		)
		residual_norm = float(np.linalg.norm(stages.residual, ord=np.inf))
		if residual_norm <= threshold:
			# Both projected copies agree to the requested tolerance. Their mean is
			# the numerically neutral representative of the physical diagonal state.
			first_copy = stages.u_final + multiplier
			second_copy = stages.v_final - multiplier
			projected_state = (first_copy + second_copy) / 2.0
			ideal_state_jacobian = None
			if compute_ideal_state_jacobian:
				evaluation = _differentiate_stages(
					dynamics,
					t,
					value,
					step,
					stages,
				)
				ideal_state_jacobian = _ideal_projected_state_jacobian(evaluation)
			return _ProjectedStep(
				state=np.asarray(projected_state),
				multiplier=multiplier.copy(),
				ideal_state_jacobian=ideal_state_jacobian,
				iterations=iteration,
				residual_norm=residual_norm,
			)
		if iteration == max_iterations:
			break
		evaluation = _differentiate_stages(
			dynamics,
			t,
			value,
			step,
			stages,
		)
		# The packed residual is component-major; Newton systems are independent
		# two-dimensional solves for the individual GC particles.
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
				"The ABBA projection Jacobian is singular at "
				f"t={t:.16g} with step={step:.16g}."
			) from exc
		correction = correction_blocks.T.reshape(-1)
		multiplier = multiplier - correction

	raise RuntimeError(
		"ABBA implicit formulation 1 did not converge at "
		f"t={t:.16g} with step={step:.16g}: "
		f"residual norm {residual_norm:.3e} exceeds {threshold:.3e} after "
		f"{max_iterations} Newton iterations."
	)


def _particle_blocks(vector: np.ndarray, dimension: int) -> np.ndarray:
	"""View one component-major vector as particle-major coordinate blocks."""
	return vector.reshape(dimension, -1).T


def _packed_particle_blocks(blocks: np.ndarray) -> np.ndarray:
	"""Pack particle-major coordinate blocks into component-major order."""
	return blocks.T.reshape(-1)


def _simultaneous_residual_blocks(
	stages: _ABBAStages,
	multiplier: np.ndarray,
	first_output: np.ndarray,
	second_output: np.ndarray,
	state_dimension: int,
) -> np.ndarray:
	"""Evaluate the two equation-(21) defects for every independent particle.

	Each returned row contains ``(d_u, d_v, g)`` in physical coordinate blocks,
	where ``d`` is the four-dimensional step-equation defect and ``g=u-v`` is
	the two-dimensional diagonal constraint.
	"""
	first_defect = first_output - multiplier - stages.u_final
	second_defect = second_output + multiplier - stages.v_final
	constraint_defect = first_output - second_output
	return np.concatenate(
		(
			_particle_blocks(first_defect, state_dimension),
			_particle_blocks(second_defect, state_dimension),
			_particle_blocks(constraint_defect, state_dimension),
		),
		axis=-1,
	)


def _simultaneous_newton_jacobian(
	evaluation: _ResidualEvaluation,
) -> np.ndarray:
	"""Assemble equation (21) as one exact 6-by-6 system per GC particle."""
	particle_count = evaluation.abba_jacobian.shape[0]
	identity_2 = np.broadcast_to(np.eye(2), (particle_count, 2, 2))
	identity_4 = np.broadcast_to(np.eye(4), (particle_count, 4, 4))
	zero_2 = np.zeros((particle_count, 2, 2), dtype=float)
	# N = G^T maps the multiplier to opposite displacements of both copies.
	normal = np.concatenate((identity_2, -identity_2), axis=-2)
	constraint = np.concatenate((identity_2, -identity_2), axis=-1)
	top_right = -(identity_4 + evaluation.abba_jacobian) @ normal
	return np.concatenate(
		(
			np.concatenate((identity_4, top_right), axis=-1),
			np.concatenate((constraint, zero_2), axis=-1),
		),
		axis=-2,
	)


def _solve_simultaneous_projected_step(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	compute_ideal_state_jacobian: bool = True,
) -> _ProjectedStep:
	"""Solve implicit formulation 2 using the simultaneous equation (21)."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The ABBA physical state must be a finite, non-empty vector.")
	multiplier = np.zeros_like(value)
	state_scale = max(1.0, float(np.linalg.norm(value, ord=np.inf)))
	threshold = absolute_tolerance + relative_tolerance * state_scale

	# Start from the uncorrected ABBA output. This makes d=0 initially while
	# retaining the generally non-zero diagonal constraint g in equation (21).
	stages = _evaluate_stages(dynamics, t, value, step, multiplier)
	first_output = stages.u_final.copy()
	second_output = stages.v_final.copy()

	for iteration in range(max_iterations + 1):
		residual_blocks = _simultaneous_residual_blocks(
			stages,
			multiplier,
			first_output,
			second_output,
			dynamics.state_dimension,
		)
		residual_norm = float(np.max(np.abs(residual_blocks)))
		if residual_norm <= threshold:
			# The simultaneous unknown is constrained to the physical diagonal. The
			# mean removes its finite-tolerance antisymmetric round-off component.
			projected_state = (first_output + second_output) / 2.0
			ideal_state_jacobian = None
			if compute_ideal_state_jacobian:
				evaluation = _differentiate_stages(
					dynamics,
					t,
					value,
					step,
					stages,
				)
				ideal_state_jacobian = _ideal_projected_state_jacobian(evaluation)
			return _ProjectedStep(
				state=np.asarray(projected_state),
				multiplier=multiplier.copy(),
				ideal_state_jacobian=ideal_state_jacobian,
				iterations=iteration,
				residual_norm=residual_norm,
			)
		if iteration == max_iterations:
			break

		evaluation = _differentiate_stages(
			dynamics,
			t,
			value,
			step,
			stages,
		)
		newton_jacobian = _simultaneous_newton_jacobian(evaluation)
		try:
			increments = np.linalg.solve(
				newton_jacobian,
				-residual_blocks[..., None],
			)[..., 0]
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				"The simultaneous ABBA projection Jacobian is singular at "
				f"t={t:.16g} with step={step:.16g}."
			) from exc

		# Equation (21) orders each particle increment as (Delta Y, Delta mu),
		# with Delta Y split into first-copy and second-copy coordinate pairs.
		first_output = first_output + _packed_particle_blocks(increments[..., :2])
		second_output = second_output + _packed_particle_blocks(increments[..., 2:4])
		multiplier = multiplier + _packed_particle_blocks(increments[..., 4:])
		stages = _evaluate_stages(dynamics, t, value, step, multiplier)

	raise RuntimeError(
		"ABBA implicit formulation 2 did not converge at "
		f"t={t:.16g} with step={step:.16g}: "
		f"simultaneous residual norm {residual_norm:.3e} exceeds "
		f"{threshold:.3e} after {max_iterations} Newton iterations."
	)


def _integrate_projected_abba(
	problem: InitialValueProblem,
	request: SimulationRequest,
	*,
	method_name: str,
	step_solver: Callable[..., _ProjectedStep],
	solver_formulation: str,
	newton_absolute_tolerance: float,
	newton_relative_tolerance: float,
	newton_max_iterations: int,
	progress: bool,
	step_observer: StepObserver | None,
	exact_tangent: bool,
) -> IntegrationData:
	"""Integrate projected ABBA with optional exact tangent propagation."""
	dynamics = problem.dynamics
	if not isinstance(dynamics, GuidingCenterJacobianSystem):
		raise TypeError(f"{method_name} requires GuidingCenterJacobianSystem.")
	if dynamics.state_dimension != 2:
		raise TypeError(f"{method_name} requires planar two-component dynamics.")
	# Preflight the exact Hessian capability before the integration grid advances.
	_checked_vector_field_jacobian(
		dynamics,
		request.t_span[0],
		problem.initial_state,
	)

	iteration_counts: list[int] = []
	residual_norms: list[float] = []
	multiplier_norms: list[float] = []
	# The dense matrix follows the packed component-major physical state layout.
	accumulated_state_jacobian = np.eye(problem.initial_state.size)

	def advance(
		t: float,
		state: np.ndarray,
		step: float,
		step_index: int,
		observe: bool,
	) -> np.ndarray:
		nonlocal accumulated_state_jacobian

		def apply_step(candidate: np.ndarray) -> np.ndarray:
			"""Apply this fixed-time projected ABBA map to one candidate."""
			return step_solver(
				dynamics,
				t,
				candidate,
				step,
				absolute_tolerance=newton_absolute_tolerance,
				relative_tolerance=newton_relative_tolerance,
				max_iterations=newton_max_iterations,
				compute_ideal_state_jacobian=False,
			).state

		state_before = np.asarray(state, dtype=float)
		result = step_solver(
			dynamics,
			t,
			state_before,
			step,
			absolute_tolerance=newton_absolute_tolerance,
			relative_tolerance=newton_relative_tolerance,
			max_iterations=newton_max_iterations,
			compute_ideal_state_jacobian=exact_tangent and observe,
		)
		if observe:
			state_jacobian = result.ideal_state_jacobian
			if exact_tangent:
				if state_jacobian is None:
					raise RuntimeError(
						"Exact ABBA tangent propagation did not produce a state Jacobian."
					)
				accumulated_state_jacobian = (
					state_jacobian @ accumulated_state_jacobian
				)
			iteration_counts.append(result.iterations)
			residual_norms.append(result.residual_norm)
			multiplier_norms.append(
				float(np.linalg.norm(result.multiplier, ord=np.inf))
			)
			if step_observer is not None:
				step_observer(
					IntegrationStep(
						dynamics_name=type(dynamics).__name__,
						method_name=method_name,
						step_index=step_index,
						time=t + step,
						duration=step,
						state_before=state_before.copy(),
						state_after=result.state.copy(),
						map_state=apply_step,
						state_jacobian=(
							None
							if state_jacobian is None
							else state_jacobian.copy()
						),
					)
				)
		return result.state

	history, step_count = integrate_fixed_grid(
		problem.initial_state,
		request,
		advance,
		progress=bool(progress),
		label=method_name,
	)
	diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
		"step_count": step_count,
		"newton_iterations": np.asarray(iteration_counts, dtype=int),
		"newton_residual_norms": np.asarray(residual_norms, dtype=float),
		"projection_multiplier_norms": np.asarray(
			multiplier_norms,
			dtype=float,
		),
		"newton_absolute_tolerance": newton_absolute_tolerance,
		"newton_relative_tolerance": newton_relative_tolerance,
		"newton_max_iterations": newton_max_iterations,
		"projection_solver_formulation": solver_formulation,
	}
	if exact_tangent:
		diagnostics.update(
			{
				"state_jacobian_kind": "exact_implicit_function",
				"final_state_jacobian": accumulated_state_jacobian.copy(),
			}
		)
	return IntegrationData(
		t=request.output_times,
		states=np.asarray(history),
		diagnostics=diagnostics,
	)

__all__: list[str] = []
