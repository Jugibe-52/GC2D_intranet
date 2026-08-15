"""Exact per-particle Jacobians for uncoupled midpoint-BM4 stages."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dynamics import GuidingCenterJacobianSystem
from simulation import IntegrationStage


MIDPOINT_BM4_STAGE_COUNT = 12


def _validated_stage_states(
	stage: IntegrationStage,
) -> tuple[np.ndarray, np.ndarray, int]:
	"""Return finite doubled GC states and their independent-particle count."""
	if not isinstance(stage, IntegrationStage):
		raise TypeError("Midpoint-BM4 Jacobians require IntegrationStage data.")
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
		raise ValueError(
			"Explicit midpoint-BM4 Jacobians require finite doubled planar "
			"states with 4N component-major entries."
		)
	if not np.isfinite(stage.time) or not np.isfinite(stage.duration):
		raise ValueError("The midpoint-BM4 stage time and duration must be finite.")
	return state_before, state_after, state_before.size // 4


def _checked_particle_vector_field_jacobians(
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
	expected_shape = (particle_count, 2, 2)
	if result.shape != expected_shape or not np.all(np.isfinite(result)):
		raise ValueError(
			"The GC vector-field Jacobian must be finite and have shape "
			f"{expected_shape}."
		)
	return result


def midpoint_bm4_stage_particle_jacobians(
	stage: IntegrationStage,
	dynamics: GuidingCenterJacobianSystem,
) -> np.ndarray:
	"""Return the exact uncoupled BM4 stage Jacobian for every GC particle.

	The doubled per-particle order is ``(u_x, u_y, v_x, v_y)``. Production
	states remain component-major, so the two field-evaluation states are sliced
	from the observed snapshots before assembling the independent ``4 x 4``
	blocks. No numerical differentiation or repeated stage map is used.
	"""
	if not isinstance(dynamics, GuidingCenterJacobianSystem):
		raise TypeError(
			"Explicit midpoint-BM4 Jacobians require "
			"GuidingCenterJacobianSystem dynamics."
		)
	if dynamics.state_dimension != 2:
		raise TypeError("Explicit midpoint-BM4 Jacobians require planar dynamics.")

	state_before, state_after, particle_count = _validated_stage_states(stage)
	physical_size = 2 * particle_count
	first_before = state_before[:physical_size]
	second_before = state_before[physical_size:]
	first_after = state_after[:physical_size]
	second_after = state_after[physical_size:]
	duration = float(stage.duration)
	identity = np.broadcast_to(np.eye(2), (particle_count, 2, 2))

	if stage.flow_name == "flow":
		# Direct stage: v' = v + s f(u), followed by u' = u + s f(v').
		w_first = _checked_particle_vector_field_jacobians(
			dynamics,
			stage.time,
			first_before,
			particle_count,
		)
		w_second = _checked_particle_vector_field_jacobians(
			dynamics,
			stage.time,
			second_after,
			particle_count,
		)
		top_left = identity + duration**2 * (w_second @ w_first)
		top_right = duration * w_second
		bottom_left = duration * w_first
		bottom_right = identity
	elif stage.flow_name == "adjoint_flow":
		# Adjoint stage: u' = u + s f(v), followed by v' = v + s f(u').
		w_first = _checked_particle_vector_field_jacobians(
			dynamics,
			stage.time,
			second_before,
			particle_count,
		)
		w_second = _checked_particle_vector_field_jacobians(
			dynamics,
			stage.time,
			first_after,
			particle_count,
		)
		top_left = identity
		top_right = duration * w_first
		bottom_left = duration * w_second
		bottom_right = identity + duration**2 * (w_second @ w_first)
	else:
		raise ValueError(
			"A midpoint-BM4 stage must be either 'flow' or 'adjoint_flow'."
		)

	result = np.empty((particle_count, 4, 4), dtype=float)
	result[:, :2, :2] = top_left
	result[:, :2, 2:] = top_right
	result[:, 2:, :2] = bottom_left
	result[:, 2:, 2:] = bottom_right
	if not np.all(np.isfinite(result)):
		raise ValueError("The explicit midpoint-BM4 stage Jacobian is non-finite.")
	return result


def midpoint_bm4_step_particle_jacobians(
	stages: Sequence[IntegrationStage],
	dynamics: GuidingCenterJacobianSystem,
) -> np.ndarray:
	"""Compose twelve exact stage factors into physical ``2 x 2`` blocks.

	The tangent begins with the diagonal embedding ``E: z -> (z, z)`` and is
	projected after the complete composition with ``P: (u, v) -> (u + v) / 2``.
	Factors multiply on the left in emitted stage order.
	"""
	observations = tuple(stages)
	if len(observations) != MIDPOINT_BM4_STAGE_COUNT:
		raise ValueError(
			"A complete midpoint-BM4 step must contain exactly twelve stages."
		)
	step_index = observations[0].step_index
	start_time = float(observations[0].time)
	time_cursor = start_time
	previous_state_after: np.ndarray | None = None
	particle_count: int | None = None
	for stage_index, stage in enumerate(observations):
		expected_flow = "adjoint_flow" if stage_index % 2 == 0 else "flow"
		if stage.dynamics is not dynamics:
			raise ValueError(
				"Midpoint-BM4 stages must expose the exact configured dynamics instance."
			)
		if (
			stage.step_index != step_index
			or stage.stage_index != stage_index
			or stage.flow_name != expected_flow
			or stage.method_name != "MidpointBM4"
			or stage.formulation_name != "GCStageProjectedFormulation"
			or stage.dynamics_name != type(dynamics).__name__
		):
			raise ValueError(
				"Midpoint-BM4 stages must be one ordered alternating complete step."
			)
		state_before, state_after, stage_particle_count = _validated_stage_states(
			stage
		)
		if particle_count is None:
			particle_count = stage_particle_count
			physical_size = 2 * particle_count
			if not np.allclose(
				state_before[:physical_size],
				state_before[physical_size:],
				rtol=0.0,
				atol=1e-13,
			):
				raise ValueError(
					"A midpoint-BM4 step must start from diagonal embedding."
				)
		elif stage_particle_count != particle_count:
			raise ValueError("The particle count changed within a midpoint-BM4 step.")
		if previous_state_after is not None and not np.array_equal(
			state_before,
			previous_state_after,
		):
			raise ValueError("Midpoint-BM4 stage state snapshots are not continuous.")
		expected_time = (
			time_cursor + stage.duration
			if stage.flow_name == "flow"
			else time_cursor
		)
		time_tolerance = float(
			64.0
			* np.finfo(float).eps
			* max(
				1.0,
				abs(start_time),
				abs(time_cursor),
				abs(expected_time),
			)
		)
		if not np.isclose(
			stage.time,
			expected_time,
			rtol=0.0,
			atol=time_tolerance,
		):
			raise ValueError("Midpoint-BM4 stage evaluation times are inconsistent.")
		time_cursor += stage.duration
		previous_state_after = state_after

	first_factor = midpoint_bm4_stage_particle_jacobians(
		observations[0],
		dynamics,
	)
	particle_count = first_factor.shape[0]
	identity = np.eye(2)
	embedding = np.vstack((identity, identity))
	tangent = np.broadcast_to(
		embedding,
		(particle_count, 4, 2),
	).copy()
	for stage_index, stage in enumerate(observations):
		factor = (
			first_factor
			if stage_index == 0
			else midpoint_bm4_stage_particle_jacobians(stage, dynamics)
		)
		if factor.shape != (particle_count, 4, 4):
			raise ValueError("The particle count changed within a midpoint-BM4 step.")
		tangent = factor @ tangent

	projection = np.hstack((identity, identity)) / 2.0
	result = projection @ tangent
	if not np.all(np.isfinite(result)):
		raise ValueError("The explicit midpoint-BM4 physical Jacobian is non-finite.")
	return np.asarray(result, dtype=float)


__all__ = [
	"MIDPOINT_BM4_STAGE_COUNT",
	"midpoint_bm4_stage_particle_jacobians",
	"midpoint_bm4_step_particle_jacobians",
]
