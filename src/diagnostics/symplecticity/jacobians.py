"""Step-Jacobian calculations used by physical symplecticity observers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from dynamics import GuidingCenterJacobianSystem
from simulation.observation import (
	ImplicitABBAIntegrationStep,
	IntegrationStep,
	StateMap,
)


StepJacobianMethod: TypeAlias = Literal[
	"finite_difference",
	"implicit_function",
	"stage_increment",
]
STEP_JACOBIAN_METHODS: tuple[StepJacobianMethod, ...] = (
	"finite_difference",
	"implicit_function",
	"stage_increment",
)


def central_difference_jacobian(
	map_state: StateMap,
	state: np.ndarray,
	*,
	relative_step: float | None = None,
) -> np.ndarray:
	"""Differentiate a packed state map using centered finite differences."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The differentiated state must be a finite, non-empty vector.")
	scale = (
		float(np.cbrt(np.finfo(float).eps))
		if relative_step is None
		else float(relative_step)
	)
	if not np.isfinite(scale) or scale <= 0:
		raise ValueError("`relative_step` must be positive and finite.")

	dimension = value.size
	jacobian = np.empty((dimension, dimension), dtype=float)
	for column in range(dimension):
		increment = scale * max(1.0, abs(float(value[column])))
		perturbation = np.zeros_like(value)
		perturbation[column] = increment
		forward = np.asarray(map_state(value + perturbation), dtype=float)
		backward = np.asarray(map_state(value - perturbation), dtype=float)
		if forward.shape != value.shape or backward.shape != value.shape:
			raise ValueError("The differentiated map changed the state shape.")
		jacobian[:, column] = (forward - backward) / (2 * increment)
	if not np.all(np.isfinite(jacobian)):
		raise ValueError("The numerical Jacobian contains non-finite values.")
	return jacobian


@dataclass(frozen=True, slots=True)
class _ImplicitABBABlocks:
	"""Per-particle blocks shared by both analytic tangent factorizations."""

	identity: np.ndarray
	w_1: np.ndarray
	w_2: np.ndarray
	w_3: np.ndarray
	w_4: np.ndarray
	top_left: np.ndarray
	top_right: np.ndarray
	bottom_left: np.ndarray
	bottom_right: np.ndarray
	residual_multiplier_jacobian: np.ndarray
	residual_state_jacobian: np.ndarray


def _validated_stage_state(
	step: ImplicitABBAIntegrationStep,
	state: np.ndarray,
	name: str,
) -> np.ndarray:
	"""Validate one packed stage snapshot against the physical input layout."""
	value = np.asarray(state, dtype=float)
	if value.shape != step.state_before.shape or not np.all(np.isfinite(value)):
		raise ValueError(
			f"Implicit ABBA diagnostic stage `{name}` must be finite and have "
			f"shape {step.state_before.shape}."
		)
	return value


def _checked_vector_field_jacobians(
	dynamics: GuidingCenterJacobianSystem,
	time: float,
	state: np.ndarray,
	particle_count: int,
) -> np.ndarray:
	"""Evaluate one finite two-by-two field Jacobian per GC particle."""
	result = np.asarray(
		dynamics.particle_vector_field_jacobians(time, state),
		dtype=float,
	)
	expected_shape = (particle_count, 2, 2)
	if result.shape != expected_shape or not np.all(np.isfinite(result)):
		raise ValueError(
			"The GC vector-field Jacobian must be finite and have shape "
			f"{expected_shape}."
		)
	return result


def _implicit_abba_blocks(step: IntegrationStep) -> _ImplicitABBABlocks:
	"""Evaluate the converged ABBA stage blocks required by analytic tangents."""
	if not isinstance(step, ImplicitABBAIntegrationStep):
		raise TypeError(
			"Analytic implicit-ABBA Jacobians require "
			"ImplicitABBAIntegrationStep data."
		)
	dynamics = step.dynamics
	if not isinstance(dynamics, GuidingCenterJacobianSystem):
		raise TypeError(
			"Analytic implicit-ABBA Jacobians require "
			"GuidingCenterJacobianSystem dynamics."
		)
	state = np.asarray(step.state_before, dtype=float)
	if (
		state.ndim != 1
		or state.size == 0
		or state.size % 2
		or not np.all(np.isfinite(state))
	):
		raise ValueError(
			"Implicit ABBA tangent diagnostics require a finite component-major "
			"planar state."
		)
	if dynamics.state_dimension != 2:
		raise TypeError("Implicit ABBA tangent diagnostics require planar dynamics.")
	if (
		not np.isfinite(step.start_time)
		or not np.isfinite(step.time)
		or not np.isfinite(step.duration)
	):
		raise ValueError("The observed step times and duration must be finite.")

	particle_count = state.size // 2
	v_initial = _validated_stage_state(step, step.v_initial, "v_initial")
	u_first = _validated_stage_state(step, step.u_first, "u_first")
	v_final = _validated_stage_state(step, step.v_final, "v_final")
	w_1 = _checked_vector_field_jacobians(
		dynamics,
		step.start_time,
		v_initial,
		particle_count,
	)
	w_2 = _checked_vector_field_jacobians(
		dynamics,
		step.start_time,
		u_first,
		particle_count,
	)
	w_3 = _checked_vector_field_jacobians(
		dynamics,
		step.time,
		u_first,
		particle_count,
	)
	w_4 = _checked_vector_field_jacobians(
		dynamics,
		step.time,
		v_final,
		particle_count,
	)
	identity = np.broadcast_to(np.eye(2), (particle_count, 2, 2))
	half_step = step.duration / 2.0
	central = w_2 + w_3
	# Matrix order is fixed because stage Jacobians generally do not commute.
	top_left = identity + half_step**2 * (w_4 @ central)
	top_right = (
		half_step * (w_1 + w_4)
		+ half_step**3 * (w_4 @ central @ w_1)
	)
	bottom_left = half_step * central
	bottom_right = identity + half_step**2 * (central @ w_1)
	residual_multiplier_jacobian = (
		top_left
		- top_right
		- bottom_left
		+ bottom_right
		+ 2.0 * identity
	)
	residual_state_jacobian = (
		top_left + top_right - bottom_left - bottom_right
	)
	return _ImplicitABBABlocks(
		identity=identity,
		w_1=w_1,
		w_2=w_2,
		w_3=w_3,
		w_4=w_4,
		top_left=top_left,
		top_right=top_right,
		bottom_left=bottom_left,
		bottom_right=bottom_right,
		residual_multiplier_jacobian=residual_multiplier_jacobian,
		residual_state_jacobian=residual_state_jacobian,
	)


def _multiplier_state_jacobian(blocks: _ImplicitABBABlocks) -> np.ndarray:
	"""Return ``D mu_h = -solve(K, L)`` for each independent particle."""
	try:
		return -np.linalg.solve(
			blocks.residual_multiplier_jacobian,
			blocks.residual_state_jacobian,
		)
	except np.linalg.LinAlgError as exc:
		raise RuntimeError(
			"The ABBA projection Jacobian is singular while differentiating the step."
		) from exc


def _dense_component_major_jacobian(blocks: np.ndarray) -> np.ndarray:
	"""Expand independent particle blocks into the packed physical layout."""
	particle_count = blocks.shape[0]
	result = np.zeros((2 * particle_count, 2 * particle_count), dtype=float)
	for particle in range(particle_count):
		indices = (particle, particle_count + particle)
		result[np.ix_(indices, indices)] = blocks[particle]
	if not np.all(np.isfinite(result)):
		raise ValueError("The analytic physical step Jacobian is non-finite.")
	return result


def implicit_function_step_jacobian(step: IntegrationStep) -> np.ndarray:
	"""Calculate the ideal-root ABBA tangent as ``P - Q solve(K, L)``."""
	blocks = _implicit_abba_blocks(step)
	multiplier_jacobian = _multiplier_state_jacobian(blocks)
	direct = blocks.top_left + blocks.top_right
	implicit_weight = blocks.top_left - blocks.top_right + blocks.identity
	physical_blocks = direct + implicit_weight @ multiplier_jacobian
	return _dense_component_major_jacobian(physical_blocks)


def stage_increment_step_jacobian(step: IntegrationStep) -> np.ndarray:
	"""Calculate the same ideal-root tangent as ``I + h/4 sum(Dk_i)``."""
	blocks = _implicit_abba_blocks(step)
	multiplier_jacobian = _multiplier_state_jacobian(blocks)
	half_step = step.duration / 2.0
	dk_1 = blocks.w_1 @ (blocks.identity - multiplier_jacobian)
	du_first = blocks.identity + multiplier_jacobian + half_step * dk_1
	dk_2 = blocks.w_2 @ du_first
	dk_3 = blocks.w_3 @ du_first
	dv_final = (
		blocks.identity
		- multiplier_jacobian
		+ half_step * (dk_2 + dk_3)
	)
	dk_4 = blocks.w_4 @ dv_final
	physical_blocks = (
		blocks.identity + step.duration / 4.0 * (dk_1 + dk_2 + dk_3 + dk_4)
	)
	return _dense_component_major_jacobian(physical_blocks)


def calculate_step_jacobian(
	step: IntegrationStep,
	*,
	method: StepJacobianMethod = "finite_difference",
	relative_step: float | None = None,
) -> np.ndarray:
	"""Calculate one local step Jacobian with the selected diagnostic method."""
	if method not in STEP_JACOBIAN_METHODS:
		raise ValueError(
			"`method` must be 'finite_difference', 'implicit_function', or "
			"'stage_increment'."
		)
	if method == "finite_difference":
		return central_difference_jacobian(
			step.map_state,
			step.state_before,
			relative_step=relative_step,
		)
	if relative_step is not None:
		raise ValueError(
			"`relative_step` is used only when method='finite_difference'."
		)
	if method == "implicit_function":
		return implicit_function_step_jacobian(step)
	return stage_increment_step_jacobian(step)


__all__ = [
	"STEP_JACOBIAN_METHODS",
	"StepJacobianMethod",
	"calculate_step_jacobian",
	"central_difference_jacobian",
	"implicit_function_step_jacobian",
	"stage_increment_step_jacobian",
]
