"""Exact per-trajectory Jacobians for projected GC integration steps."""

from __future__ import annotations

import numpy as np

from diagnostics.abba_jacobian import particle_jacobian_blocks
from diagnostics.jacobians import implicit_function_step_jacobian
from dynamics import GuidingCenterJacobianSystem
from simulation import (
	ABBA_PROJECTION_FORMULATIONS,
	ABBA4ImplicitSingleProjectionIntegrationStep,
	ABBA4ImplicitIntegrationStep,
	ABBA2ImplicitIntegrationStep,
	ImplicitBM4IntegrationStep,
	IntegrationStage,
	IntegrationStep,
	UnprojectedABBAIntegrationStep,
	gc_coupling_matrix,
)


_BM4_STAGE_COUNT = 12


def _validated_step(
	step: IntegrationStep,
	*,
	method_name: str,
) -> tuple[GuidingCenterJacobianSystem, np.ndarray, np.ndarray, int]:
	"""Return one finite planar step and its exact generating dynamics."""
	if not isinstance(step, IntegrationStep):
		raise TypeError("Exact trajectory Jacobians require IntegrationStep data.")
	if step.method_name != method_name:
		raise TypeError(f"Expected a {method_name} integration step.")
	dynamics = step.dynamics
	if not isinstance(dynamics, GuidingCenterJacobianSystem):
		raise TypeError(
			"Exact trajectory Jacobians require GuidingCenterJacobianSystem dynamics."
		)
	if dynamics.state_dimension != 2:
		raise TypeError("Exact trajectory Jacobians require planar dynamics.")
	state_before = np.asarray(step.state_before, dtype=float)
	state_after = np.asarray(step.state_after, dtype=float)
	if (
		state_before.ndim != 1
		or state_before.size == 0
		or state_before.size % 2
		or state_after.shape != state_before.shape
		or not np.all(np.isfinite(state_before))
		or not np.all(np.isfinite(state_after))
	):
		raise ValueError("A physical GC step must contain finite 2N states.")
	if not all(
		np.isfinite(value)
		for value in (step.start_time, step.time, step.duration)
	):
		raise ValueError("Integration-step times and duration must be finite.")
	tolerance = float(
		64.0
		* np.finfo(float).eps
		* max(1.0, abs(step.start_time), abs(step.time), abs(step.duration))
	)
	if not np.isclose(
		step.start_time + step.duration,
		step.time,
		rtol=0.0,
		atol=tolerance,
	):
		raise ValueError("Integration-step start, duration, and final time disagree.")
	return dynamics, state_before, state_after, state_before.size // 2


def _checked_field(
	dynamics: GuidingCenterJacobianSystem,
	time: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate a finite vector field without changing packed layout."""
	result = np.asarray(dynamics.vector_field(time, state), dtype=float)
	if result.shape != state.shape or not np.all(np.isfinite(result)):
		raise ValueError("The GC vector field changed shape or became non-finite.")
	return result


def _checked_field_jacobians(
	dynamics: GuidingCenterJacobianSystem,
	time: float,
	state: np.ndarray,
	particle_count: int,
) -> np.ndarray:
	"""Evaluate one exact finite ``2 x 2`` field Jacobian per particle."""
	result = np.asarray(
		dynamics.particle_vector_field_jacobians(time, state),
		dtype=float,
	)
	expected = (particle_count, 2, 2)
	if result.shape != expected or not np.all(np.isfinite(result)):
		raise ValueError(
			"The GC vector-field Jacobian must be finite and have shape "
			f"{expected}."
		)
	return result


def _packed_planar_states(blocks: np.ndarray) -> np.ndarray:
	"""Pack particle-major ``(N, 2)`` values into component-major order."""
	values = np.asarray(blocks, dtype=float)
	if values.ndim != 2 or values.shape[1] != 2:
		raise ValueError("Planar particle blocks must have shape (N, 2).")
	return np.concatenate((values[:, 0], values[:, 1]))


def _doubled_particle_blocks(state: np.ndarray, particle_count: int) -> np.ndarray:
	"""Return doubled component-major state as particle-major ``(N, 4)``."""
	value = np.asarray(state, dtype=float)
	if value.shape != (4 * particle_count,):
		raise ValueError("The doubled GC state must contain 4N entries.")
	return np.column_stack(
		(
			value[:particle_count],
			value[particle_count : 2 * particle_count],
			value[2 * particle_count : 3 * particle_count],
			value[3 * particle_count :],
		)
	)


def abba2_midpoint_step_particle_jacobians(
	step: IntegrationStep,
) -> np.ndarray:
	"""Return the exact arithmetic-midpoint ABBA map Jacobian per particle."""
	dynamics, state, state_after, particle_count = _validated_step(
		step,
		method_name="ABBA2Midpoint",
	)
	half_step = step.duration / 2.0
	start_time = step.start_time
	final_time = step.time

	# Reconstruct the four explicit endpoint-time shears used by the accepted map.
	u_first = state + half_step * _checked_field(dynamics, start_time, state)
	v_first = state + half_step * _checked_field(dynamics, start_time, u_first)
	v_final = v_first + half_step * _checked_field(
		dynamics,
		final_time,
		u_first,
	)
	u_final = u_first + half_step * _checked_field(
		dynamics,
		final_time,
		v_final,
	)
	projected = (u_final + v_final) / 2.0
	if not np.array_equal(projected, state_after):
		raise ValueError(
			"The midpoint-ABBA stage reconstruction differs from the observed map."
		)

	w_1 = _checked_field_jacobians(
		dynamics,
		start_time,
		state,
		particle_count,
	)
	w_2 = _checked_field_jacobians(
		dynamics,
		start_time,
		u_first,
		particle_count,
	)
	w_3 = _checked_field_jacobians(
		dynamics,
		final_time,
		u_first,
		particle_count,
	)
	w_4 = _checked_field_jacobians(
		dynamics,
		final_time,
		v_final,
		particle_count,
	)
	identity = np.broadcast_to(np.eye(2), (particle_count, 2, 2))
	du_first = identity + half_step * w_1
	dv_first = identity + half_step * (w_2 @ du_first)
	dv_final = dv_first + half_step * (w_3 @ du_first)
	du_final = du_first + half_step * (w_4 @ dv_final)
	result = (du_final + dv_final) / 2.0
	if not np.all(np.isfinite(result)):
		raise ValueError("The explicit midpoint-ABBA Jacobian is non-finite.")
	return np.asarray(result, dtype=float)


def abba2_implicit_step_particle_jacobians(
	step: IntegrationStep,
) -> np.ndarray:
	"""Return exact ideal-root ABBA2 Jacobians for either solve formulation."""
	_, _, _, particle_count = _validated_step(
		step,
		method_name="ABBA2Implicit",
	)
	if not isinstance(step, ABBA2ImplicitIntegrationStep):
		raise TypeError(
			"ABBA2Implicit exact Jacobians require converged stage snapshots."
		)
	if step.formulation_name not in ABBA_PROJECTION_FORMULATIONS:
		raise TypeError("The observed step has an unknown ABBA2 formulation.")
	dense = implicit_function_step_jacobian(step)
	return particle_jacobian_blocks(dense, particle_count)


def abba4_implicit_step_particle_jacobians(
	step: IntegrationStep,
) -> np.ndarray:
	"""Compose the three exact ideal-root implicit-ABBA tangent factors."""
	dynamics, state, state_after, particle_count = _validated_step(
		step,
		method_name="ABBA4Implicit",
	)
	if not isinstance(step, ABBA4ImplicitIntegrationStep):
		raise TypeError(
			"ABBA4Implicit exact Jacobians require three converged substeps."
		)
	if step.formulation_name not in ABBA_PROJECTION_FORMULATIONS:
		raise TypeError("The observed step has an unknown ABBA4 formulation.")
	coefficients = np.asarray(step.composition_coefficients, dtype=float)
	root_two = float(np.cbrt(2.0))
	gamma = 1.0 / (2.0 - root_two)
	delta = -root_two / (2.0 - root_two)
	expected_coefficients = np.asarray((gamma, delta, gamma), dtype=float)
	coefficient_tolerance = float(
		64.0
		* np.finfo(float).eps
		* max(1.0, float(np.max(np.abs(expected_coefficients))))
	)
	if coefficients.shape != (3,) or not np.allclose(
		coefficients,
		expected_coefficients,
		rtol=0.0,
		atol=coefficient_tolerance,
	):
		raise ValueError("The ABBA4 composition coefficients are inconsistent.")
	substeps = tuple(step.substeps)
	if len(substeps) != 3:
		raise ValueError("One ABBA4 step must contain exactly three implicit substeps.")

	accumulated = np.broadcast_to(
		np.eye(2),
		(particle_count, 2, 2),
	).copy()
	current_state = state
	current_time = float(step.start_time)
	for index, (coefficient, substep) in enumerate(
		zip(coefficients, substeps, strict=True)
	):
		if not isinstance(substep, ABBA2ImplicitIntegrationStep):
			raise TypeError("Every ABBA4 substep must expose implicit ABBA stages.")
		expected_duration = float(coefficient * step.duration)
		tolerance = float(
			64.0
			* np.finfo(float).eps
			* max(
				1.0,
				abs(current_time),
				abs(expected_duration),
				abs(float(substep.start_time)),
			)
		)
		if (
			substep.step_index != step.step_index
			or substep.method_name != step.method_name
			or substep.formulation_name != step.formulation_name
			or substep.dynamics is not dynamics
			or not np.isclose(
				substep.start_time,
				current_time,
				rtol=0.0,
				atol=tolerance,
			)
			or not np.isclose(
				substep.duration,
				expected_duration,
				rtol=0.0,
				atol=tolerance,
			)
			or not np.array_equal(substep.state_before, current_state)
		):
			raise ValueError(f"ABBA4 substep {index} is inconsistent with the composition.")
		factor = particle_jacobian_blocks(
			implicit_function_step_jacobian(substep),
			particle_count,
		)
		accumulated = factor @ accumulated
		current_state = np.asarray(substep.state_after, dtype=float)
		current_time = float(substep.time)
	if not np.array_equal(current_state, state_after):
		raise ValueError("The ABBA4 substeps do not produce the observed final state.")
	if not np.isclose(
		current_time,
		step.time,
		rtol=0.0,
		atol=float(64.0 * np.finfo(float).eps * max(1.0, abs(step.time))),
	):
		raise ValueError("The ABBA4 signed substep times do not end at the outer time.")
	if not np.all(np.isfinite(accumulated)):
		raise ValueError("The exact ABBA4 physical Jacobian is non-finite.")
	return np.asarray(accumulated, dtype=float)


def _unprojected_abba_step_particle_jacobians(
	step: UnprojectedABBAIntegrationStep,
	dynamics: GuidingCenterJacobianSystem,
	particle_count: int,
) -> np.ndarray:
	"""Return one exact doubled ABBA tangent from stored stage points."""
	half_step = step.duration / 2.0
	w_1 = _checked_field_jacobians(
		dynamics,
		step.start_time,
		step.v_initial,
		particle_count,
	)
	w_2 = _checked_field_jacobians(
		dynamics,
		step.start_time,
		step.u_first,
		particle_count,
	)
	w_3 = _checked_field_jacobians(
		dynamics,
		step.time,
		step.u_first,
		particle_count,
	)
	w_4 = _checked_field_jacobians(
		dynamics,
		step.time,
		step.v_final,
		particle_count,
	)
	identity = np.broadcast_to(np.eye(2), (particle_count, 2, 2))
	central = w_2 + w_3
	top_left = identity + half_step**2 * (w_4 @ central)
	top_right = (
		half_step * (w_1 + w_4)
		+ half_step**3 * (w_4 @ central @ w_1)
	)
	bottom_left = half_step * central
	bottom_right = identity + half_step**2 * (central @ w_1)
	return np.concatenate(
		(
			np.concatenate((top_left, top_right), axis=-1),
			np.concatenate((bottom_left, bottom_right), axis=-1),
		),
		axis=-2,
	)


def abba4_implicit_single_projection_step_particle_jacobians(
	step: IntegrationStep,
) -> np.ndarray:
	"""Differentiate the one ideal projection around unprojected ABBA4."""
	dynamics, state, state_after, particle_count = _validated_step(
		step,
		method_name="ABBA4ImplicitSingleProjection",
	)
	if not isinstance(step, ABBA4ImplicitSingleProjectionIntegrationStep):
		raise TypeError(
			"ABBA4ImplicitSingleProjection exact Jacobians require converged "
			"outer-projection snapshots."
		)
	if step.formulation_name not in ABBA_PROJECTION_FORMULATIONS:
		raise TypeError(
			"The observed step has an unknown single-projection ABBA4 formulation."
		)
	root_two = float(np.cbrt(2.0))
	gamma = 1.0 / (2.0 - root_two)
	delta = -root_two / (2.0 - root_two)
	expected_coefficients = np.asarray((gamma, delta, gamma), dtype=float)
	coefficients = np.asarray(step.composition_coefficients, dtype=float)
	coefficient_tolerance = float(
		64.0
		* np.finfo(float).eps
		* max(1.0, float(np.max(np.abs(expected_coefficients))))
	)
	if coefficients.shape != (3,) or not np.allclose(
		coefficients,
		expected_coefficients,
		rtol=0.0,
		atol=coefficient_tolerance,
	):
		raise ValueError("The ABBA4 single-projection coefficients are inconsistent.")
	multiplier = np.asarray(step.multiplier, dtype=float)
	if multiplier.shape != state.shape or not np.all(np.isfinite(multiplier)):
		raise ValueError("The outer projection multiplier must be a finite 2N vector.")
	substeps = tuple(step.substeps)
	if len(substeps) != 3:
		raise ValueError("The unprojected ABBA4 base map must contain three substeps.")

	base_tangent = np.broadcast_to(
		np.eye(4),
		(particle_count, 4, 4),
	).copy()
	u_previous = state + multiplier
	v_previous = state - multiplier
	current_time = float(step.start_time)
	for index, (coefficient, substep) in enumerate(
		zip(coefficients, substeps, strict=True)
	):
		if not isinstance(substep, UnprojectedABBAIntegrationStep):
			raise TypeError("Every base-map entry must expose unprojected ABBA stages.")
		expected_duration = float(coefficient * step.duration)
		tolerance = float(
			64.0
			* np.finfo(float).eps
			* max(
				1.0,
				abs(current_time),
				abs(expected_duration),
				abs(float(substep.start_time)),
			)
		)
		if (
			not np.isclose(
				substep.start_time,
				current_time,
				rtol=0.0,
				atol=tolerance,
			)
			or not np.isclose(
				substep.duration,
				expected_duration,
				rtol=0.0,
				atol=tolerance,
			)
			or not np.isclose(
				substep.time,
				current_time + expected_duration,
				rtol=0.0,
				atol=tolerance,
			)
			or not np.array_equal(substep.u_initial, u_previous)
			or not np.array_equal(substep.v_initial, v_previous)
		):
			raise ValueError(
				f"Unprojected ABBA4 substep {index} is inconsistent with its base map."
			)
		factor = _unprojected_abba_step_particle_jacobians(
			substep,
			dynamics,
			particle_count,
		)
		base_tangent = factor @ base_tangent
		u_previous = np.asarray(substep.u_final, dtype=float)
		v_previous = np.asarray(substep.v_final, dtype=float)
		current_time = float(substep.time)

	projected = (u_previous + v_previous) / 2.0
	projection_tolerance = max(
		float(step.newton_tolerance),
		float(
			64.0
			* np.finfo(float).eps
			* max(1.0, float(np.linalg.norm(state_after, ord=np.inf)))
		),
	)
	if not np.allclose(
		projected,
		state_after,
		rtol=0.0,
		atol=projection_tolerance,
	):
		raise ValueError("The unprojected ABBA4 output and physical state disagree.")
	identity = np.broadcast_to(np.eye(2), (particle_count, 2, 2))
	top_left = base_tangent[:, :2, :2]
	top_right = base_tangent[:, :2, 2:]
	bottom_left = base_tangent[:, 2:, :2]
	bottom_right = base_tangent[:, 2:, 2:]
	residual_multiplier = (
		top_left
		- top_right
		- bottom_left
		+ bottom_right
		+ 2.0 * identity
	)
	residual_state = top_left + top_right - bottom_left - bottom_right
	try:
		multiplier_tangent = -np.linalg.solve(
			residual_multiplier,
			residual_state,
		)
	except np.linalg.LinAlgError as exc:
		raise RuntimeError(
			"The ABBA4 single-projection root is singular."
		) from exc
	result = (
		top_left
		+ top_right
		+ (top_left - top_right + identity) @ multiplier_tangent
	)
	if not np.all(np.isfinite(result)):
		raise ValueError("The exact ABBA4 single-projection Jacobian is non-finite.")
	return np.asarray(result, dtype=float)


def coupled_bm4_stage_particle_jacobians(
	stage: IntegrationStage,
	dynamics: GuidingCenterJacobianSystem,
	*,
	coupling_frequency: float,
) -> np.ndarray:
	"""Return one exact coupled-BM4 ``4 x 4`` stage factor per particle."""
	if not isinstance(stage, IntegrationStage):
		raise TypeError("Coupled BM4 Jacobians require IntegrationStage data.")
	if stage.dynamics is not dynamics:
		raise ValueError("The BM4 stage and Jacobian dynamics instances differ.")
	state_before = np.asarray(stage.state_before, dtype=float)
	state_after = np.asarray(stage.state_after, dtype=float)
	if (
		state_before.ndim != 1
		or state_before.size == 0
		or state_before.size % 4
		or state_after.shape != state_before.shape
		or not np.all(np.isfinite(state_before))
		or not np.all(np.isfinite(state_after))
	):
		raise ValueError("A coupled BM4 stage must contain finite 4N states.")
	particle_count = state_before.size // 4
	before = _doubled_particle_blocks(state_before, particle_count)
	after = _doubled_particle_blocks(state_after, particle_count)
	duration = float(stage.duration)
	coupling = gc_coupling_matrix(duration, coupling_frequency)
	identity = np.broadcast_to(np.eye(2), (particle_count, 2, 2))

	if stage.flow_name == "flow":
		# Direct stages shear first and apply the exact coupling rotation last.
		uncoupled_after = np.linalg.solve(coupling, after.T).T
		w_first = _checked_field_jacobians(
			dynamics,
			stage.time,
			_packed_planar_states(before[:, :2]),
			particle_count,
		)
		w_second = _checked_field_jacobians(
			dynamics,
			stage.time,
			_packed_planar_states(uncoupled_after[:, 2:]),
			particle_count,
		)
		shear = np.empty((particle_count, 4, 4), dtype=float)
		shear[:, :2, :2] = identity + duration**2 * (w_second @ w_first)
		shear[:, :2, 2:] = duration * w_second
		shear[:, 2:, :2] = duration * w_first
		shear[:, 2:, 2:] = identity
		result = np.einsum("ij,njk->nik", coupling, shear)
	elif stage.flow_name == "adjoint_flow":
		# Adjoint stages couple first and then apply the two triangular shears.
		coupled_before = before @ coupling.T
		w_first = _checked_field_jacobians(
			dynamics,
			stage.time,
			_packed_planar_states(coupled_before[:, 2:]),
			particle_count,
		)
		w_second = _checked_field_jacobians(
			dynamics,
			stage.time,
			_packed_planar_states(after[:, :2]),
			particle_count,
		)
		shear = np.empty((particle_count, 4, 4), dtype=float)
		shear[:, :2, :2] = identity
		shear[:, :2, 2:] = duration * w_first
		shear[:, 2:, :2] = duration * w_second
		shear[:, 2:, 2:] = identity + duration**2 * (w_second @ w_first)
		result = np.einsum("nij,jk->nik", shear, coupling)
	else:
		raise ValueError("A BM4 stage must be either 'flow' or 'adjoint_flow'.")
	if not np.all(np.isfinite(result)):
		raise ValueError("The exact coupled-BM4 stage Jacobian is non-finite.")
	return np.asarray(result, dtype=float)


def bm4_implicit_1_step_particle_jacobians(
	step: IntegrationStep,
) -> np.ndarray:
	"""Differentiate the ideal reduced Hairer projection by exact stages."""
	dynamics, state, state_after, particle_count = _validated_step(
		step,
		method_name="BM4Implicit1",
	)
	if not isinstance(step, ImplicitBM4IntegrationStep):
		raise TypeError("BM4Implicit1 exact Jacobians require base-stage snapshots.")
	if step.formulation_name != "bm4_implicit_1_reduced":
		raise TypeError("The observed step is not implicit BM4 formulation 1.")
	stages = tuple(step.base_stages)
	if len(stages) != _BM4_STAGE_COUNT:
		raise ValueError("A complete implicit BM4 base cycle has twelve stages.")
	multiplier = np.asarray(step.multiplier, dtype=float)
	if multiplier.shape != state.shape or not np.all(np.isfinite(multiplier)):
		raise ValueError("The projected-BM4 multiplier must be a finite 2N vector.")
	expected_input = np.concatenate((state + multiplier, state - multiplier))
	time_cursor = float(step.start_time)
	previous_after: np.ndarray | None = None
	base_tangent = np.broadcast_to(
		np.eye(4),
		(particle_count, 4, 4),
	).copy()
	for stage_index, stage in enumerate(stages):
		expected_flow = "adjoint_flow" if stage_index % 2 == 0 else "flow"
		if (
			stage.step_index != step.step_index
			or stage.stage_index != stage_index
			or stage.flow_name != expected_flow
			or stage.method_name != step.method_name
			or stage.formulation_name != "GCExtendedFormulation"
			or stage.dynamics is not dynamics
		):
			raise ValueError("Implicit BM4 stages are not one ordered base cycle.")
		before = np.asarray(stage.state_before, dtype=float)
		after = np.asarray(stage.state_after, dtype=float)
		if stage_index == 0 and not np.array_equal(before, expected_input):
			raise ValueError("The BM4 base cycle did not start at the projected input.")
		if previous_after is not None and not np.array_equal(before, previous_after):
			raise ValueError("Implicit BM4 stage snapshots are not continuous.")
		expected_time = (
			time_cursor + stage.duration
			if stage.flow_name == "flow"
			else time_cursor
		)
		tolerance = float(
			64.0
			* np.finfo(float).eps
			* max(1.0, abs(time_cursor), abs(expected_time), abs(stage.time))
		)
		if not np.isclose(stage.time, expected_time, rtol=0.0, atol=tolerance):
			raise ValueError("Implicit BM4 stage evaluation times are inconsistent.")
		factor = coupled_bm4_stage_particle_jacobians(
			stage,
			dynamics,
			coupling_frequency=step.coupling_frequency,
		)
		base_tangent = factor @ base_tangent
		time_cursor += stage.duration
		previous_after = after

	if not np.isclose(
		time_cursor,
		step.time,
		rtol=0.0,
		atol=float(64.0 * np.finfo(float).eps * max(1.0, abs(step.time))),
	):
		raise ValueError("The twelve BM4 stage durations do not equal the full step.")
	assert previous_after is not None
	physical_size = 2 * particle_count
	projected = (
		previous_after[:physical_size] + previous_after[physical_size:]
	) / 2.0
	if not np.allclose(projected, state_after, rtol=0.0, atol=2e-13):
		raise ValueError("The BM4 base output and physical projection disagree.")

	identity = np.eye(2)
	embedding = np.vstack((identity, identity))
	normal = np.vstack((identity, -identity))
	constraint = np.hstack((identity, -identity))
	projection = np.hstack((identity, identity)) / 2.0
	residual_multiplier = constraint @ base_tangent @ normal + 2.0 * identity
	residual_state = constraint @ base_tangent @ embedding
	try:
		multiplier_tangent = -np.linalg.solve(
			residual_multiplier,
			residual_state,
		)
	except np.linalg.LinAlgError as exc:
		raise RuntimeError(
			"The implicit BM4 projection Jacobian is singular."
		) from exc
	result = projection @ base_tangent @ (
		embedding + normal @ multiplier_tangent
	)
	if not np.all(np.isfinite(result)):
		raise ValueError("The exact implicit-BM4 physical Jacobian is non-finite.")
	return np.asarray(result, dtype=float)


__all__ = [
	"abba4_implicit_step_particle_jacobians",
	"abba4_implicit_single_projection_step_particle_jacobians",
	"bm4_implicit_1_step_particle_jacobians",
	"coupled_bm4_stage_particle_jacobians",
	"abba2_implicit_step_particle_jacobians",
	"abba2_midpoint_step_particle_jacobians",
]
