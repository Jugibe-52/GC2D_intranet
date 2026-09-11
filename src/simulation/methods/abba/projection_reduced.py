"""Reduced-multiplier formulation of Hairer's implicit ABBA projection."""

from __future__ import annotations

import numpy as np

from dynamics import GuidingCenterJacobianSystem

from .._nonlinear import NonlinearSolver, _solve_broyden, _solve_newton
from ._projection_common import _ProjectedStep
from .maps.physical import (
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
	"""Solve the reduced-multiplier projection with Newton or good Broyden."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The ABBA physical state must be a finite, non-empty vector.")
	multiplier = np.zeros_like(value)
	state_scale = max(1.0, float(np.linalg.norm(value, ord=np.inf)))
	threshold = absolute_tolerance + relative_tolerance * state_scale
	if nonlinear_solver == "broyden":
		result = _solve_broyden(
			lambda candidate: (
				(
					stages := _evaluate_stages(
						dynamics,
						t,
						value,
						step,
						candidate,
					)
				).residual,
				stages,
			),
			multiplier,
			4.0 * np.eye(value.size),
			tolerance=threshold,
			max_iterations=max_iterations,
			context=(
				"ABBA reduced-multiplier projection at "
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

	def evaluate_newton(candidate: np.ndarray) -> tuple[np.ndarray, _ABBAStages]:
		base = _evaluate_stages(dynamics, t, value, step, candidate)
		return base.residual, base

	def update(candidate: np.ndarray, residual: np.ndarray, base: _ABBAStages) -> np.ndarray:
		evaluation = _differentiate_stages(dynamics, t, value, step, base)
		blocks = evaluation.residual.reshape(dynamics.state_dimension, -1).T
		try:
			correction = np.linalg.solve(evaluation.jacobian, blocks[..., None])[..., 0]
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				f"The ABBA projection Jacobian is singular at t={t:.16g} with step={step:.16g}."
			) from exc
		return np.asarray(candidate - correction.T.reshape(-1))

	newton_result = _solve_newton(
		evaluate_newton, multiplier, update, tolerance=threshold,
		max_iterations=max_iterations,
		context=f"ABBA reduced projection at t={t:.16g} with step={step:.16g}",
	)
	base = newton_result.payload
	first_copy = base.u_final + newton_result.unknown
	second_copy = base.v_final - newton_result.unknown
	return _ProjectedStep(
		state=np.asarray((first_copy + second_copy) / 2.0),
		multiplier=newton_result.unknown, stages=base,
		iterations=newton_result.iterations,
		residual_evaluations=newton_result.residual_evaluations,
		residual_norm=float(np.linalg.norm(newton_result.residual, ord=np.inf)),
	)


__all__: list[str] = []
