"""Reduced-multiplier formulation of Hairer's implicit ABBA projection."""

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


def _evaluate_stages(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	multiplier: np.ndarray,
) -> _ABBAStages:
	"""Apply ABBA and assemble the reduced projection residual."""
	displaced = _evaluate_displaced_stages(
		dynamics,
		t,
		state,
		step,
		multiplier,
	)
	return _ABBAStages(
		u_initial=displaced.u_initial,
		v_initial=displaced.v_initial,
		u_first=displaced.u_first,
		v_final=displaced.v_final,
		u_final=displaced.u_final,
		residual=displaced.residual + 2.0 * multiplier,
	)


def _evaluate_residual(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	multiplier: np.ndarray,
) -> _ResidualEvaluation:
	"""Apply ABBA and evaluate its exact reduced residual Jacobian."""
	stages = _evaluate_stages(dynamics, t, state, step, multiplier)
	return _differentiate_stages(dynamics, t, state, step, stages)


def _solve_reduced_multiplier_step(
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
        'reduced_multiplier', t, state, step)
    return _ProjectedStep(result.state, result.multiplier, result.trace.maps[0].stages,
                          result.iterations, result.residual_evaluations, result.residual_norm)


__all__: list[str] = []
