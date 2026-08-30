"""Hairer symmetric-projection kernels for endpoint-time A-B-B-A maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import GuidingCenterJacobianSystem

from .._nonlinear import NonlinearSolver, _solve_broyden
from ._core import _ABBAStages, _evaluate_unprojected_stages


@dataclass(frozen=True, slots=True)
class _ProjectedStep:
	"""Converged physical state and nonlinear-solve diagnostics for one step."""

	state: np.ndarray
	multiplier: np.ndarray
	stages: _ABBAStages
	iterations: int
	residual_evaluations: int
	residual_norm: float


@dataclass(frozen=True, slots=True)
class _ResidualEvaluation:
	"""ABBA stages and the exact reduced residual Jacobian at one multiplier."""

	u_final: np.ndarray
	v_final: np.ndarray
	residual: np.ndarray
	jacobian: np.ndarray
	abba_jacobian: np.ndarray


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
	"""Apply the four midpoint ABBA stages and evaluate equation (11)."""
	# G^T mu displaces the two copies in opposite physical directions.
	u_initial = state + multiplier
	v_initial = state - multiplier
	unprojected = _evaluate_unprojected_stages(
		dynamics,
		t,
		u_initial,
		v_initial,
		step,
	)
	return _ABBAStages(
		u_initial=unprojected.u_initial,
		v_initial=unprojected.v_initial,
		u_first=unprojected.u_first,
		v_final=unprojected.v_final,
		u_final=unprojected.u_final,
		residual=unprojected.residual + 2.0 * multiplier,
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


def _solve_projected_step(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver = "newton",
) -> _ProjectedStep:
	"""Solve implicit formulation 1 with Newton or good Broyden steps."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The ABBA physical state must be a finite, non-empty vector.")
	multiplier = np.zeros_like(value)
	state_scale = max(1.0, float(np.linalg.norm(value, ord=np.inf)))
	threshold = absolute_tolerance + relative_tolerance * state_scale
	if nonlinear_solver == "broyden":
		result = _solve_broyden(
			lambda candidate: (
				(stages := _evaluate_stages(
					dynamics,
					t,
					value,
					step,
					candidate,
				)).residual,
				stages,
			),
			multiplier,
			4.0 * np.eye(value.size),
			tolerance=threshold,
			max_iterations=max_iterations,
			context=(
				"ABBA implicit formulation 1 at "
				f"t={t:.16g} with step={step:.16g}"
			),
		)
		stages = result.payload
		first_copy = stages.u_final + result.unknown
		second_copy = stages.v_final - result.unknown
		return _ProjectedStep(
			state=np.asarray((first_copy + second_copy) / 2.0),
			multiplier=result.unknown,
			stages=stages,
			iterations=result.iterations,
			residual_evaluations=result.residual_evaluations,
			residual_norm=float(np.linalg.norm(result.residual, ord=np.inf)),
		)
	if nonlinear_solver != "newton":
		raise ValueError("Unknown nonlinear solver for implicit ABBA.")

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
			return _ProjectedStep(
				state=np.asarray(projected_state),
				multiplier=multiplier.copy(),
				stages=stages,
				iterations=iteration,
				residual_evaluations=iteration + 1,
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
	nonlinear_solver: NonlinearSolver = "newton",
) -> _ProjectedStep:
	"""Solve simultaneous equation (21) with Newton or good Broyden steps."""
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
	if nonlinear_solver == "broyden":
		physical_size = value.size
		internal_size = 2 * physical_size
		identity_internal = np.eye(internal_size)
		identity_physical = np.eye(physical_size)
		normal = np.concatenate(
			(identity_physical, -identity_physical), axis=0
		)
		constraint = np.concatenate(
			(identity_physical, -identity_physical), axis=1
		)
		initial_jacobian = np.block(
			[
				[identity_internal, -2.0 * normal],
				[constraint, np.zeros((physical_size, physical_size))],
			]
		)

		def residual_function(
			unknown: np.ndarray,
		) -> tuple[np.ndarray, tuple[_ABBAStages, np.ndarray, np.ndarray, np.ndarray]]:
			"""Evaluate equation (21) for one simultaneous Broyden iterate."""
			first = unknown[:physical_size]
			second = unknown[physical_size:internal_size]
			candidate_multiplier = unknown[internal_size:]
			candidate_stages = _evaluate_stages(
				dynamics,
				t,
				value,
				step,
				candidate_multiplier,
			)
			residual = np.concatenate(
				(
					first - candidate_multiplier - candidate_stages.u_final,
					second + candidate_multiplier - candidate_stages.v_final,
					first - second,
				)
			)
			return residual, (
				candidate_stages,
				first,
				second,
				candidate_multiplier,
			)

		result = _solve_broyden(
			residual_function,
			np.concatenate((first_output, second_output, multiplier)),
			initial_jacobian,
			tolerance=threshold,
			max_iterations=max_iterations,
			context=(
				"ABBA implicit formulation 2 at "
				f"t={t:.16g} with step={step:.16g}"
			),
			initial_evaluation=(
				np.concatenate(
					(
						np.zeros(internal_size),
						first_output - second_output,
					)
				),
				(stages, first_output, second_output, multiplier),
			),
		)
		stages, first_output, second_output, multiplier = result.payload
		return _ProjectedStep(
			state=np.asarray((first_output + second_output) / 2.0),
			multiplier=multiplier.copy(),
			stages=stages,
			iterations=result.iterations,
			residual_evaluations=result.residual_evaluations,
			residual_norm=float(np.linalg.norm(result.residual, ord=np.inf)),
		)
	if nonlinear_solver != "newton":
		raise ValueError("Unknown nonlinear solver for implicit ABBA.")

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
			return _ProjectedStep(
				state=np.asarray(projected_state),
				multiplier=multiplier.copy(),
				stages=stages,
				iterations=iteration,
				residual_evaluations=iteration + 1,
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


__all__: list[str] = []
