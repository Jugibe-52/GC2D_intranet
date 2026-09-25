"""On-demand ABBA and BM4 snapshots from accepted shared composition traces."""

from __future__ import annotations

from formulations.base import PreparedDirectAdjointFormulation
from contracts.observation import IntegrationStage
from methods.extended.core.composition import StageMap
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import numpy as np
from dynamics import DynamicalSystem

from contracts.observation import (
	ABBA2ImplicitIntegrationStep,
	ABBA4ImplicitIntegrationStep, ABBA6ImplicitIntegrationStep,
	IntegrationStep, UnprojectedABBAIntegrationStep,
)
from methods.extended.configuration import ProjectionFormulation
from methods.extended.core.records import (
	CompositionTrace, ProjectedMapResult, StepResult,
)
from methods.extended.core.records import ProjectedMap


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
    pairs = []
    for first, last in zip(trace.stages[::2], trace.stages[1::2], strict=True):
        size = first.state_before.size // 2
        u0, v0 = first.state_before[:size], first.state_before[size:]
        uf, vf = last.state_after[:size], last.state_after[size:]
        stages = _ABBAStages(u0, v0, first.energy_points[1][2], vf, uf, uf - vf)
        pairs.append(PhysicalBaseMapTrace(first.start_time, first.duration + last.duration, stages))
    return tuple(pairs)


EventBuilder: TypeAlias = Callable[
	[float, float, int, np.ndarray, StepResult], IntegrationStep
]


def _solve_fields(projections: tuple[ProjectedMapResult, ...]) -> dict[str, Any]:
	"""Read one consistent aggregate from the accepted numerical records."""
	worst = max(projections, key=lambda p: p.stats.residual_norm / p.stats.tolerance)
	return {
		"nonlinear_solver": worst.stats.solver,
		"newton_iterations": sum(p.stats.iterations for p in projections),
		"residual_evaluations": sum(p.stats.residual_evaluations for p in projections),
		"newton_residual_norm": worst.stats.residual_norm,
		"newton_tolerance": worst.stats.tolerance,
		"projection_multiplier_norm": max(
			float(np.linalg.norm(p.multiplier, ord=np.inf)) for p in projections
		),
	}


def bind_event_builder(
	dynamics: DynamicalSystem,
	method_name: str,
	formulation: ProjectionFormulation,
	*,
	order: Literal[2, 4, 6],
	coefficients: tuple[float, ...],
	project: ProjectedMap,
) -> EventBuilder:
	"""Choose the public event adapter once, outside the integration loop."""
	def physical_single(
		projection: ProjectedMapResult, step_index: int
	) -> ABBA2ImplicitIntegrationStep:
		trace = projection.trace
		assert isinstance(trace, CompositionTrace) and len(trace.stages) // 2 == 1
		stages = abba_pairs(trace)[0].stages

		def map_state(candidate: np.ndarray) -> np.ndarray:
			return project(projection.start_time, candidate, projection.duration).state

		return ABBA2ImplicitIntegrationStep(
			dynamics_name=type(dynamics).__name__, method_name=method_name,
			step_index=step_index, start_time=projection.start_time,
			time=projection.start_time + projection.duration, duration=projection.duration,
			state_before=projection.state_before.copy(), state_after=projection.state.copy(),
			map_state=map_state, formulation_name=formulation, dynamics=dynamics,
			multiplier=projection.multiplier.copy(),
			u_initial=stages.u_initial.copy(), v_initial=stages.v_initial.copy(),
			u_first=stages.u_first.copy(), v_final=stages.v_final.copy(),
			u_final=stages.u_final.copy(), **_solve_fields((projection,)),
		)

	def build(
		t: float, h: float, step_index: int, state_before: np.ndarray, result: StepResult
	) -> IntegrationStep:
		projections = result.projections
		if order == 2:
			return physical_single(projections[0], step_index)

		def map_state(candidate: np.ndarray) -> np.ndarray:
			return project(t, candidate, h).state

		fields: dict[str, Any] = dict(
			dynamics_name=type(dynamics).__name__, method_name=method_name,
			step_index=step_index, start_time=t, time=t + h, duration=h,
			state_before=state_before.copy(),
			state_after=projections[-1].state.copy(),
			map_state=map_state, dynamics=dynamics, formulation_name=formulation,
			**_solve_fields(projections),
		)
		trace = projections[0].trace
		assert isinstance(trace, CompositionTrace)
		substeps = tuple(
			UnprojectedABBAIntegrationStep(
				start_time=m.start_time, time=m.start_time + m.duration,
				duration=m.duration, u_initial=m.stages.u_initial.copy(),
				v_initial=m.stages.v_initial.copy(), u_first=m.stages.u_first.copy(),
				v_final=m.stages.v_final.copy(), u_final=m.stages.u_final.copy(),
			)
			for m in abba_pairs(trace)
		)
		event_type = ABBA4ImplicitIntegrationStep if order == 4 else ABBA6ImplicitIntegrationStep
		return event_type(
			**fields, multiplier=projections[0].multiplier.copy(),
			composition_coefficients=np.asarray(coefficients), substeps=substeps,
		)

	return build


__all__: list[str] = []


def stage_events(
    prepared: PreparedDirectAdjointFormulation, trace: CompositionTrace, *,
    step_index: int, formulation_name: str, method_name: str,
    stage_maps: tuple[StageMap, StageMap] | None = None,
) -> tuple[IntegrationStage, ...]:
    """Copy accepted stage records without replaying the numerical composition."""
    events: list[IntegrationStage] = []
    direct, adjoint = stage_maps or (prepared.direct_map, prepared.adjoint_map)
    for index, stage in enumerate(trace.stages):
        mapper = direct if stage.direct else adjoint
        flow: Literal['flow', 'adjoint_flow'] = 'flow' if stage.direct else 'adjoint_flow'

        def map_state(candidate: np.ndarray, _map: StageMap = mapper,
                      _duration: float = stage.duration, _time: float = stage.time) -> np.ndarray:
            """Freeze the signed stage for delayed diagnostic evaluation."""
            return _map(_duration, _time, candidate)

        events.append(IntegrationStage(
            dynamics_name=prepared.dynamics_name, formulation_name=formulation_name,
            method_name=method_name, flow_name=flow, step_index=step_index,
            stage_index=index, time=stage.time, duration=stage.duration,
            state_before=stage.state_before.copy(), state_after=stage.state_after.copy(),
            map_state=map_state, dynamics=prepared.dynamics,
        ))
    return tuple(events)
