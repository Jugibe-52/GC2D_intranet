# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Fourth-order palindromic composition over direct/adjoint maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .._fixed import integrate_fixed_grid
from .._result import IntegrationData
from ..formulations import DirectAdjointFormulation
from ..formulations.base import (
	PreparedDirectAdjointFormulation,
	generalized_energy_error,
)
from ..observation import IntegrationStage, StageObserver
from ..problem import InitialValueProblem
from ..request import SimulationRequest


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

		state_before = state
		state = _checked_map(
			selected_map,
			duration,
			evaluation_time,
			state_before,
		)
		if stage_observer is not None:

			def map_state(
				candidate: np.ndarray,
				_selected_map: object = selected_map,
				_duration: float = duration,
				_evaluation_time: float = evaluation_time,
			) -> np.ndarray:
				"""Evaluate this fixed prepared map on a diagnostic candidate."""
				return _checked_map(
					_selected_map,
					_duration,
					_evaluation_time,
					candidate,
				)

			stage_observer(
				IntegrationStage(
					system_name=prepared.observer_label,
					flow_name=flow_name,
					step_index=step_index,
					stage_index=stage_index,
					time=evaluation_time,
					duration=duration,
					state_before=np.asarray(state_before).copy(),
					state_after=np.asarray(state).copy(),
					map_state=map_state,
				)
			)
		t += duration
	return state


@dataclass(frozen=True, slots=True)
class BM4Composition:
	"""BM4 method configured with one reusable direct/adjoint formulation."""

	formulation: DirectAdjointFormulation
	track_energy: bool = False
	progress: bool = False
	stage_observer: StageObserver | None = None

	def __post_init__(self) -> None:
		if not isinstance(self.formulation, DirectAdjointFormulation):
			raise TypeError("`formulation` must implement DirectAdjointFormulation.")

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate prepared formulation maps and project physical output."""
		prepared = self.formulation.prepare(
			problem,
			track_energy=bool(self.track_energy),
		)

		def advance(
			t: float,
			state: np.ndarray,
			step: float,
			step_index: int,
			observe: bool,
		) -> np.ndarray:
			return _advance_composition(
				prepared,
				t,
				state,
				step,
				step_index=step_index,
				stage_observer=self.stage_observer if observe else None,
			)

		internal_history, step_count = integrate_fixed_grid(
			prepared.initial_internal_state,
			request,
			advance,
			progress=bool(self.progress),
			label=prepared.observer_label,
		)
		states, diagnostics = prepared.project(internal_history)
		diagnostics["step_count"] = step_count
		momentum_value = diagnostics.get("extended_momentum")
		momentum = (
			None if momentum_value is None else np.asarray(momentum_value)
		)
		if self.track_energy:
			diagnostics["energy_error"] = generalized_energy_error(
				request.output_times,
				states,
				momentum,
				problem.dynamics,
			)
		return IntegrationData(
			t=request.output_times,
			states=np.asarray(states),
			diagnostics=diagnostics,
		)


__all__ = ["BM4Composition"]
