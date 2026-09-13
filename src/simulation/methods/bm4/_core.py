# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Private twelve-stage base map shared by the BM4 methods.

The base method alternates a first-order map with its adjoint using a
palindromic coefficient sequence.  If ``h`` is the complete step and ``b_j``
is one coefficient, stage ``j`` advances by the signed duration ``b_j h``.
The fourth coefficient is negative by design, so stage time is not monotone;
the full coefficient sequence nevertheless sums to one.

This module knows nothing about Hairer's projection.  It traverses exactly one
unprojected BM4 cycle on whatever internal vector the prepared formulation
accepts. The callers apply their diagonal projection around the entire cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from ...formulations.base import PreparedDirectAdjointFormulation
from ...observation import IntegrationStage, StageObserver


# The six independent coefficients sum to 1/2.  Mirroring them below produces
# the symmetric twelve-stage sequence and a total signed duration of one step.
_BM4_HALF_STAGES = np.asarray(
	[
		0.0792036964311957,
		0.1303114101821663,
		0.2228614958676077,
		-0.3667132690474257,
		0.3246481886897062,
		0.1096884778767498,
	],
	dtype=float,
)
_BM4_STAGES = np.concatenate((_BM4_HALF_STAGES, np.flip(_BM4_HALF_STAGES)))
# ``1`` selects the adjoint map and ``0`` the direct map.  Tiling [1, 0]
# therefore starts with the adjoint and ends with the direct map.
_BM4_ORDERS = np.tile(np.asarray([1, 0], dtype=int), _BM4_HALF_STAGES.size)


def _checked_map(
	mapper: object,
	duration: float,
	t: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Apply one prepared stage map without allowing a layout change.

	A prepared map follows the ``mapper(duration, time, state)`` protocol.  BM4
	requires every direct and adjoint stage to preserve the internal vector shape;
	numeric finiteness is checked by the implicit caller after the complete map.
	"""
	if not callable(mapper):
		raise TypeError("A formulation map is not callable.")
	result = np.asarray(mapper(duration, t, state))
	if result.shape != state.shape:
		raise ValueError("A formulation map changed the internal state shape.")
	return result


def _advance_composition(
	prepared: PreparedDirectAdjointFormulation,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	step_index: int,
	stage_observer: StageObserver | None,
	formulation_name: str,
	method_name: str,
) -> np.ndarray:
	"""Advance one complete unprojected BM4 cycle.

	Parameters
	----------
	prepared:
		Direct and adjoint maps already bound to one physical problem.
	t:
		Start time of the complete step.  The local stage clock is updated by
		each signed duration.
	state:
		Internal state accepted by ``prepared``.  For ``BM4Implicit`` this is the
		doubled vector ``(u, v)`` of shape ``(4*p,)`` for ``p`` particles.
	step:
		Duration of the complete BM4 cycle, before multiplication by stage
		coefficients.
	step_index, stage_observer, formulation_name, method_name:
		Diagnostic metadata.  These values do not modify the numerical map.

	Returns
	-------
	numpy.ndarray
		The internal state after all twelve direct/adjoint stages.  No projection
		has been applied.

	Notes
	-----
	Adjoint stages evaluate at the current stage clock.  Direct stages evaluate
	at the end of their signed substep.  This convention makes each direct map
	and its adjoint consistent for non-autonomous guiding-centre dynamics.
	"""
	for stage_index, (coefficient, order) in enumerate(
		zip(_BM4_STAGES, _BM4_ORDERS, strict=True)
	):
		# A negative BM4 coefficient deliberately integrates this subflow
		# backwards; it must also move the local clock backwards.
		duration = float(coefficient * step)
		if order == 0:
			selected_map = prepared.direct_map
			flow_name: Literal["flow", "adjoint_flow"] = "flow"
			evaluation_time = t + duration
		else:
			selected_map = prepared.adjoint_map
			flow_name = "adjoint_flow"
			evaluation_time = t

		# Default arguments freeze this stage's map, duration, and time.  The
		# observer may retain ``map_state`` and evaluate it after the loop has
		# advanced to later stages.
		def apply_stage(
			candidate: np.ndarray,
			_selected_map: object = selected_map,
			_duration: float = duration,
			_evaluation_time: float = evaluation_time,
		) -> np.ndarray:
			"""Apply this prepared map to a state of the expected shape."""
			return _checked_map(
				_selected_map,
				_duration,
				_evaluation_time,
				candidate,
			)

		state_before = state
		state = apply_stage(state_before)
		if stage_observer is not None:

			def map_state(
				candidate: np.ndarray,
				_apply_stage: Callable[[np.ndarray], np.ndarray] = apply_stage,
			) -> np.ndarray:
				"""Evaluate this fixed prepared map on a diagnostic candidate."""
				return _apply_stage(candidate)

			# Copies keep the stored record independent of later stages.  The
			# prepared maps used here return fresh arrays; shape checking alone would
			# not protect ``state_before`` from an in-place mapper.
			stage_observer(
				IntegrationStage(
					dynamics_name=prepared.dynamics_name,
					formulation_name=formulation_name,
					method_name=method_name,
					flow_name=flow_name,
					step_index=step_index,
					stage_index=stage_index,
					time=evaluation_time,
					duration=duration,
					state_before=np.asarray(state_before).copy(),
					state_after=np.asarray(state).copy(),
					map_state=map_state,
					dynamics=prepared.dynamics,
				)
			)
		# The twelve signed durations sum to ``step`` even though intermediate
		# times need not be monotone.
		t += duration
	return state

__all__: list[str] = []
