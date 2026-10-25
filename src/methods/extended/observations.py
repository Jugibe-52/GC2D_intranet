"""On-demand ABBA and BM4 snapshots from accepted shared composition traces."""

from __future__ import annotations

from contracts.methods.maps import DirectAdjointMaps
from contracts.methods.observation import IntegrationStage
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import numpy as np
from contracts.arrays import BackendArray

from contracts.methods.extended.abba import ABBA2ImplicitIntegrationStep, ABBA4ImplicitIntegrationStep, UnprojectedABBAIntegrationStep
from contracts.methods.observation import IntegrationStep
from methods.extended.configuration import ProjectionFormulation
from methods.extended.core.records import (
	CompositionTrace, OperationStage, ProjectedMapResult,
)
from contracts.methods.extended.core.records import ProjectedMap

StageMap: TypeAlias = Callable[[float, float, BackendArray], BackendArray]


@dataclass(frozen=True, slots=True)
class _ABBAStages:
    """Four retained shear states in one accepted ABBA pair."""

    u_initial: np.ndarray
    v_initial: np.ndarray
    u_first: np.ndarray
    v_final: np.ndarray
    u_final: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True, slots=True)
class PhysicalBaseMapTrace:
    """One ABBA pair exposed to the established diagnostic API."""

    start_time: float
    duration: float
    stages: _ABBAStages


def abba_pairs(trace: CompositionTrace) -> tuple[PhysicalBaseMapTrace, ...]:
    """Group accepted uncoupled stages for ABBA observers without numerical replay."""
    pairs: list[PhysicalBaseMapTrace] = []
    shears: tuple[OperationStage, ...] = tuple(op for op in trace.operations if op.source is not None)
    first: OperationStage
    second: OperationStage
    third: OperationStage
    last: OperationStage
    for first, second, third, last in zip(shears[::4], shears[1::4], shears[2::4], shears[3::4], strict=True):
        assert (first.operation, second.operation, third.operation, last.operation) == ('A', 'B', 'B', 'A')
        size = first.state_before.size // 2
        u0: np.ndarray
        v0: np.ndarray
        u0, v0 = np.asarray(first.state_before[:size]), np.asarray(first.state_before[size:])
        uf: np.ndarray
        vf: np.ndarray
        uf, vf = np.asarray(last.state_after[:size]), np.asarray(last.state_after[size:])
        stages: _ABBAStages = _ABBAStages(u0, v0, np.asarray(second.source), vf, uf, uf - vf)
        pairs.append(PhysicalBaseMapTrace(first.time, first.duration + last.duration, stages))
    return tuple(pairs)


EventBuilder: TypeAlias = Callable[
	[float, float, int, np.ndarray, ProjectedMapResult], IntegrationStep
]


def _solve_fields(projection: ProjectedMapResult) -> dict[str, Any]:
	"""Expose the accepted outer solve through the established observer fields."""
	return {
		"nonlinear_solver": projection.stats.solver,
		"newton_iterations": projection.stats.iterations,
		"residual_evaluations": projection.stats.residual_evaluations,
		"newton_residual_norm": projection.stats.residual_norm,
		"newton_tolerance": projection.stats.tolerance,
		"projection_multiplier_norm": float(np.linalg.norm(projection.multiplier, ord=np.inf)),
	}


def bind_event_builder(
	method_name: str,
	formulation: ProjectionFormulation,
	*,
	order: Literal[2, 4],
	coefficients: tuple[float, ...],
	project: ProjectedMap,
) -> EventBuilder:
	"""Bind the public event adapter to the accepted projection map."""
	def physical_single(
		projection: ProjectedMapResult, step_index: int
	) -> ABBA2ImplicitIntegrationStep:
		trace = projection.trace
		stages = abba_pairs(trace)[0].stages

		def map_state(candidate: np.ndarray) -> np.ndarray:
			return np.asarray(project(projection.start_time, candidate, projection.duration).state)

		return ABBA2ImplicitIntegrationStep(
			dynamics_name='', method_name=method_name,
			step_index=step_index, start_time=projection.start_time,
			time=projection.start_time + projection.duration, duration=projection.duration,
			state_before=np.asarray(projection.state_before.copy()), state_after=projection.state.copy(),
			map_state=map_state, formulation_name=formulation,
			multiplier=np.asarray(projection.multiplier.copy()),
			u_initial=stages.u_initial.copy(), v_initial=stages.v_initial.copy(),
			u_first=stages.u_first.copy(), v_final=stages.v_final.copy(),
			u_final=stages.u_final.copy(), **_solve_fields(projection),
		)

	def build(
		t: float, h: float, step_index: int, state_before: np.ndarray, result: ProjectedMapResult
	) -> IntegrationStep:
		if order == 2:
			return physical_single(result, step_index)

		def map_state(candidate: np.ndarray) -> np.ndarray:
			return np.asarray(project(t, candidate, h).state)

		fields: dict[str, Any] = dict(
			dynamics_name='', method_name=method_name,
			step_index=step_index, start_time=t, time=t + h, duration=h,
			state_before=np.asarray(state_before.copy()),
			state_after=result.state.copy(),
			map_state=map_state,  formulation_name=formulation,
			**_solve_fields(result),
		)
		trace = result.trace
		assert isinstance(trace, CompositionTrace)
		substeps: tuple[UnprojectedABBAIntegrationStep, ...] = tuple(
			UnprojectedABBAIntegrationStep(
				start_time=m.start_time, time=m.start_time + m.duration,
				duration=m.duration, u_initial=m.stages.u_initial.copy(),
				v_initial=m.stages.v_initial.copy(), u_first=m.stages.u_first.copy(),
				v_final=m.stages.v_final.copy(), u_final=m.stages.u_final.copy(),
			)
			for m in abba_pairs(trace)
		)
		return ABBA4ImplicitIntegrationStep(
			**fields, multiplier=np.asarray(result.multiplier.copy()),
			composition_coefficients=np.asarray(coefficients), substeps=substeps,
		)

	return build


__all__: list[str] = []


def stage_events(
    prepared: DirectAdjointMaps, trace: CompositionTrace, *,
    step_index: int, formulation_name: str, method_name: str,
    stage_maps: tuple[StageMap, StageMap] | None = None,
) -> tuple[IntegrationStage, ...]:
    """Copy accepted stage records without replaying the numerical composition."""
    events: list[IntegrationStage] = []
    direct: Callable[[float, float, BackendArray], BackendArray]
    adjoint: Callable[[float, float, BackendArray], BackendArray]
    direct, adjoint = stage_maps or (prepared.direct_map, prepared.adjoint_map)
    shears: tuple[OperationStage, ...] = tuple(op for op in trace.operations if op.source is not None)
    index: int
    first: OperationStage
    last: OperationStage
    for index, (first, last) in enumerate(zip(shears[::2], shears[1::2], strict=True)):
        # BM4 observations retain one direct/adjoint group per two shears.
        mapper = direct if first.operation == 'B' else adjoint
        flow: Literal['flow', 'adjoint_flow'] = 'flow' if first.operation == 'B' else 'adjoint_flow'

        def map_state(candidate: np.ndarray, _map: StageMap = mapper,
                      _duration: float = first.duration, _time: float = first.time) -> np.ndarray:
            """Freeze the signed stage for delayed diagnostic evaluation."""
            return np.asarray(_map(_duration, _time, candidate))

        events.append(IntegrationStage(
            dynamics_name='', formulation_name=formulation_name,
            method_name=method_name, flow_name=flow, step_index=step_index,
            stage_index=index, time=first.time, duration=first.duration,
            state_before=np.asarray(first.state_before.copy()), state_after=last.state_after.copy(),
            map_state=map_state,
        ))
    return tuple(events)
