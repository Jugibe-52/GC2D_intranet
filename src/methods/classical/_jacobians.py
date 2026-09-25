"""Shared vector-field and Jacobian operations for implicit classical methods."""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np

from dynamics import DynamicalSystem, GuidingCenterJacobianSystem


JacobianMethod: TypeAlias = Literal["auto", "analytic", "finite_difference"]
ResolvedJacobianMethod: TypeAlias = Literal["analytic", "finite_difference"]


def _checked_vector_field(
	dynamics: DynamicalSystem,
	time: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate one finite vector field without allowing a shape change."""
	result = np.asarray(dynamics.vector_field(time, state), dtype=float)
	if result.shape != state.shape or not np.all(np.isfinite(result)):
		raise ValueError("The vector field must be finite and preserve state shape.")
	return result


def _dense_finite_difference_jacobian(
	dynamics: DynamicalSystem,
	time: float,
	state: np.ndarray,
	*,
	relative_step: float,
) -> np.ndarray:
	"""Differentiate a vector field with scale-aware centered differences."""
	dimension = state.size
	jacobian = np.empty((dimension, dimension), dtype=float)
	for column in range(dimension):
		increment = relative_step * max(1.0, abs(float(state[column])))
		perturbation = np.zeros_like(state)
		perturbation[column] = increment
		forward = _checked_vector_field(dynamics, time, state + perturbation)
		backward = _checked_vector_field(dynamics, time, state - perturbation)
		jacobian[:, column] = (forward - backward) / (2.0 * increment)
	if not np.all(np.isfinite(jacobian)):
		raise ValueError("The finite-difference vector-field Jacobian is non-finite.")
	return jacobian


def _checked_analytic_jacobians(
	dynamics: GuidingCenterJacobianSystem,
	time: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Return one exact finite ``2 x 2`` block per independent GC particle."""
	if state.size % 2:
		raise ValueError("Analytic GC Jacobians require an even physical state size.")
	particle_count = state.size // 2
	result = np.asarray(
		dynamics.particle_vector_field_jacobians(time, state),
		dtype=float,
	)
	expected_shape = (particle_count, 2, 2)
	if result.shape != expected_shape or not np.all(np.isfinite(result)):
		raise ValueError(
			"The analytic GC Jacobian must be finite and have shape "
			f"{expected_shape}."
		)
	return result


def _resolved_jacobian_method(
	dynamics: DynamicalSystem,
	requested: JacobianMethod,
	*,
	initial_time: float,
	initial_state: np.ndarray,
) -> ResolvedJacobianMethod:
	"""Select analytic GC blocks or finite differences from available capabilities."""
	if requested == "auto":
		if (
			not isinstance(dynamics, GuidingCenterJacobianSystem)
			or dynamics.state_dimension != 2
		):
			return "finite_difference"
		try:
			_checked_analytic_jacobians(dynamics, initial_time, initial_state)
		except (TypeError, ValueError, NotImplementedError):
			return "finite_difference"
		return "analytic"
	if requested == "analytic" and (
		not isinstance(dynamics, GuidingCenterJacobianSystem)
		or dynamics.state_dimension != 2
	):
		raise TypeError(
			"`newton_jacobian_method='analytic'` requires "
			"planar GuidingCenterJacobianSystem dynamics."
		)
	return requested
