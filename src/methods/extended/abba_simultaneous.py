"""Simultaneous state-multiplier formulation of implicit ABBA projection."""

from __future__ import annotations

import numpy as np
from methods.extended.projection import solve_projection
from methods.extended.composition import ABBA2
from methods.extended.abba_maps import uncoupled_maps
from methods._nonlinear import SolverOptions

from dynamics import GuidingCenterJacobianSystem

from methods._nonlinear import NonlinearSolver, _solve_broyden, _solve_newton
from methods.extended.abba_maps import _ProjectedStep
from methods.extended.abba_maps import (
	_ABBAStages,
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
    """Adapt the shared spatial projection to the historical ABBA step view."""
    result = solve_projection(uncoupled_maps(dynamics, state), ABBA2,
        SolverOptions(nonlinear_solver, absolute_tolerance, relative_tolerance, max_iterations),
        'simultaneous_state_multiplier', t, state, step)
    return _ProjectedStep(result.state, result.multiplier, result.trace.maps[0].stages,
                          result.iterations, result.residual_evaluations, result.residual_norm)


__all__: list[str] = []
