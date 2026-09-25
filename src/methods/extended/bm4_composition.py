# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Compatibility traversal and stage events for the shared BM4 recipe."""
from __future__ import annotations
from typing import Literal
import numpy as np
from formulations.base import PreparedDirectAdjointFormulation
from contracts.observation import IntegrationStage, StageObserver
from methods.extended.composition import BM4, StageMap, compose
from methods.extended.records import CompositionTrace

_BM4_STAGES = np.asarray(BM4.coefficients)
_BM4_HALF_STAGES = _BM4_STAGES[:6]
_BM4_ORDERS = np.tile(np.asarray([1, 0], dtype=int), 6)


def _advance_composition(
    prepared: PreparedDirectAdjointFormulation, t: float, state: np.ndarray, step: float, *,
    step_index: int, stage_observer: StageObserver | None,
    formulation_name: str, method_name: str,
    stage_maps: tuple[StageMap, StageMap] | None = None,
) -> np.ndarray:
    """Expose historical BM4 stage events from the common composition trace."""
    trace = compose(prepared, BM4, t, state, step, stage_maps=stage_maps)
    if stage_observer is not None:
        for event in stage_events(prepared, trace, step_index=step_index,
                                  formulation_name=formulation_name, method_name=method_name,
                                  stage_maps=stage_maps):
            stage_observer(event)
    return trace.state


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
