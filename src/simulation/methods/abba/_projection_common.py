"""Stage and tangent kernels shared by both implicit ABBA projections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import GuidingCenterJacobianSystem

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


def _evaluate_displaced_stages(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	multiplier: np.ndarray,
) -> _ABBAStages:
	"""Apply ABBA to copies displaced by one projection multiplier."""
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
		residual=unprojected.residual,
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


__all__: list[str] = []
