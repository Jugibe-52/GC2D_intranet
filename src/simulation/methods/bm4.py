# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Fourth-order palindromic composition over direct/adjoint maps."""

from __future__ import annotations

from collections.abc import Callable
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
_InternalProjection = Callable[[np.ndarray], np.ndarray]


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
	stage_projection: _InternalProjection | None = None,
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
			"""Apply this prepared map and its optional end-of-stage projection."""
			mapped = _checked_map(
				_selected_map,
				_duration,
				_evaluation_time,
				candidate,
			)
			if stage_projection is None:
				return mapped
			projected = np.asarray(stage_projection(mapped))
			if projected.shape != mapped.shape:
				raise ValueError(
					"The end-of-stage projection changed the internal state shape."
				)
			return projected

		state_before = state
		state = apply_stage(state_before)
		if stage_observer is not None:

			def map_state(
				candidate: np.ndarray,
				_apply_stage: _InternalProjection = apply_stage,
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
		return _integrate_bm4(self, problem, request, project_each_stage=False)


@dataclass(frozen=True, slots=True)
class ProjectedBM4Composition(BM4Composition):
	"""BM4-based composition that projects after every direct or adjoint map."""

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate maps and re-embed both copies after every internal stage."""
		return _integrate_bm4(self, problem, request, project_each_stage=True)


def _integrate_bm4(
	method: BM4Composition,
	problem: InitialValueProblem,
	request: SimulationRequest,
	*,
	project_each_stage: bool,
) -> IntegrationData:
	"""Run BM4 with an optional formulation-owned projection after every map."""
	prepared = method.formulation.prepare(
		problem,
		track_energy=bool(method.track_energy),
	)
	stage_projection: _InternalProjection | None = None
	if project_each_stage:
		candidate = getattr(prepared, "project_internal_state", None)
		if (
			not bool(getattr(prepared, "supports_stage_projection", False))
			or not callable(candidate)
		):
			raise TypeError(
				"ProjectedBM4Composition requires a stage-projected formulation."
			)
		stage_projection = candidate

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
			stage_observer=method.stage_observer if observe else None,
			stage_projection=stage_projection,
			formulation_name=type(method.formulation).__name__,
			method_name=type(method).__name__,
		)

	internal_history, step_count = integrate_fixed_grid(
		prepared.initial_internal_state,
		request,
		advance,
		progress=bool(method.progress),
		label=type(method).__name__,
	)
	states, diagnostics = prepared.project(internal_history)
	diagnostics["step_count"] = step_count
	momentum_value = diagnostics.get("extended_momentum")
	momentum = None if momentum_value is None else np.asarray(momentum_value)
	if method.track_energy:
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


__all__ = ["BM4Composition", "ProjectedBM4Composition"]
