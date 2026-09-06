# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Fourth-order palindromic composition over direct/adjoint maps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from ...formulations.base import PreparedDirectAdjointFormulation
from ...observation import IntegrationStage, StageObserver


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
_BM4_ORDERS = np.tile(np.asarray([1, 0], dtype=int), _BM4_HALF_STAGES.size)


def _checked_map(
	mapper: object,
	duration: float,
	t: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Apply one formulation map while enforcing the shape invariant."""
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
	"""Apply one complete BM4 cycle with the established stage-time convention."""
	for stage_index, (coefficient, order) in enumerate(
		zip(_BM4_STAGES, _BM4_ORDERS, strict=True)
	):
		duration = float(coefficient * step)
		if order == 0:
			selected_map = prepared.direct_map
			flow_name: Literal["flow", "adjoint_flow"] = "flow"
			evaluation_time = t + duration
		else:
			selected_map = prepared.adjoint_map
			flow_name = "adjoint_flow"
			evaluation_time = t

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
		t += duration
	return state

__all__: list[str] = []
