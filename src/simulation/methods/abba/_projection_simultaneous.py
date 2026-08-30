"""Simultaneous state-multiplier formulation of implicit ABBA projection."""

from __future__ import annotations

import numpy as np

from dynamics import GuidingCenterJacobianSystem

from .._nonlinear import NonlinearSolver, _solve_broyden
from ._core import _ABBAStages
from ._projection_common import (
	_ProjectedStep,
	_ResidualEvaluation,
	_differentiate_stages,
	_evaluate_displaced_stages,
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
	"""Evaluate simultaneous state-multiplier defects for every particle.

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
	"""Assemble one exact simultaneous 6-by-6 system per GC particle."""
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


def _solve_simultaneous_state_multiplier_step(
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
	"""Solve the simultaneous state-multiplier projection."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The ABBA physical state must be a finite, non-empty vector.")
	multiplier = np.zeros_like(value)
	state_scale = max(1.0, float(np.linalg.norm(value, ord=np.inf)))
	threshold = absolute_tolerance + relative_tolerance * state_scale

	# Start from the uncorrected ABBA output. This makes d=0 initially while
	# retaining the generally non-zero diagonal constraint in the coupled system.
	stages = _evaluate_displaced_stages(dynamics, t, value, step, multiplier)
	first_output = stages.u_final.copy()
	second_output = stages.v_final.copy()
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
		) -> tuple[np.ndarray, tuple[_ABBAStages, np.ndarray, np.ndarray, np.ndarray]]:
			"""Evaluate the coupled defects for one simultaneous Broyden iterate."""
			first = unknown[:physical_size]
			second = unknown[physical_size:internal_size]
			candidate_multiplier = unknown[internal_size:]
			candidate_stages = _evaluate_displaced_stages(
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
				"ABBA simultaneous state-multiplier projection at "
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
		stages = _evaluate_displaced_stages(
			dynamics,
			t,
			value,
			step,
			multiplier,
		)

	raise RuntimeError(
		"ABBA simultaneous state-multiplier projection did not converge at "
		f"t={t:.16g} with step={step:.16g}: "
		f"simultaneous residual norm {residual_norm:.3e} exceeds "
		f"{threshold:.3e} after {max_iterations} Newton iterations."
	)


__all__: list[str] = []
