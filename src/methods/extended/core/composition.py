# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Palindromic direct/adjoint recipes and their accepted spatial traces.

The stage clock follows signed durations. Adjoint stages evaluate at the start
of a substep; direct stages evaluate at its end. Projection belongs outside
this traversal, so a recipe can also be used by explicit midpoint methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from formulations.base import PreparedDirectAdjointFormulation
from formulations.gc import GCDoubledMaps, _EnergyQuadraturePoint
from methods.extended.core.records import CompositionTrace, MapStage
_CUBE_ROOT_TWO = float(np.cbrt(2.0))
_GAMMA = 1.0 / (2.0 - _CUBE_ROOT_TWO)
_DELTA = -_CUBE_ROOT_TWO / (2.0 - _CUBE_ROOT_TWO)
_ABBA4_COEFFICIENTS = np.asarray((_GAMMA, _DELTA, _GAMMA), dtype=float)

# Yoshida's real symmetric order-six solution. Its negative stages are required
# for the odd composition conditions to cancel while preserving self-adjointness.
_ABBA6_COEFFICIENTS = np.asarray(
	(
		0.78451361047755726382,
		0.23557321335935813368,
		-1.17767998417887100695,
		1.31518632068391121889,
		-1.17767998417887100695,
		0.23557321335935813368,
		0.78451361047755726382,
	),
	dtype=float,
)





@dataclass(frozen=True, slots=True)
class Composition:
    """Signed weights in execution order, alternating adjoint then direct.

    Each weight multiplies the complete step. Reflection also exchanges the
    map with its adjoint; symmetry alone does not establish the claimed order.
    """

    name: str
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        weights = np.asarray(self.coefficients, dtype=float)
        if weights.ndim != 1 or weights.size == 0 or weights.size % 2 or not np.all(np.isfinite(weights)):
            raise ValueError("A composition requires an even, nonempty sequence of finite weights.")
        object.__setattr__(self, "coefficients", tuple(float(w) for w in weights))
        tolerance = float(64 * np.finfo(float).eps)
        if not np.allclose(weights, weights[::-1], rtol=0, atol=tolerance):
            raise ValueError("Composition weights must be palindromic.")
        if not np.isclose(np.sum(weights), 1, rtol=0, atol=tolerance):
            raise ValueError("Composition weights must sum to one.")


ABBA2 = Composition("ABBA2", (0.5, 0.5))
ABBA4 = Composition("ABBA4", tuple(float(c / 2) for c in _ABBA4_COEFFICIENTS for _ in range(2)))
ABBA6 = Composition("ABBA6", tuple(float(c / 2) for c in _ABBA6_COEFFICIENTS for _ in range(2)))
_BM4_HALF = (0.0792036964311957, 0.1303114101821663, 0.2228614958676077,
             -0.3667132690474257, 0.3246481886897062, 0.1096884778767498)
BM4 = Composition("BM4", _BM4_HALF + _BM4_HALF[::-1])
StageMap = Callable[[float, float, np.ndarray], np.ndarray]


def compose(
    maps: PreparedDirectAdjointFormulation, recipe: Composition,
    t: float, state: np.ndarray, h: float, *,
    stage_maps: tuple[StageMap, StageMap] | None = None,
) -> CompositionTrace:
    """Apply a recipe once, retaining spatial inputs without evaluating energy.

    GC shear points also provide the inputs for exact stage Jacobians. These
    private numerical records are distinct from optional public observer events.
    Arrays are not mutated by the prepared maps, so internal traces share them.
    """
    initial = np.asarray(state, dtype=float)
    if initial.ndim != 1 or not initial.size or not np.all(np.isfinite(initial)):
        raise ValueError("The doubled composition state must be a finite vector.")
    if not np.isfinite(t) or not np.isfinite(h):
        raise ValueError("Composition time and duration must be finite.")
    current = initial
    clock = float(t)
    stages = []
    direct, adjoint = stage_maps or (maps.direct_map, maps.adjoint_map)
    for index, coefficient in enumerate(recipe.coefficients):
        duration = float(coefficient * h)
        is_direct = index % 2 == 1
        time = clock + duration if is_direct else clock
        mapper = direct if is_direct else adjoint
        points: list[_EnergyQuadraturePoint] = []
        if stage_maps is None and isinstance(maps, GCDoubledMaps):
            gc_map = maps.direct_map if is_direct else maps.adjoint_map
            after = np.asarray(gc_map(duration, time, current, energy_points=points))
        else:
            after = np.asarray(mapper(duration, time, current))
        if after.shape != initial.shape or not np.all(np.isfinite(after)):
            raise ValueError("A composition map changed shape or became non-finite.")
        stages.append(MapStage(clock, time, duration, is_direct, current, after, tuple(points)))
        current = after
        clock += duration
    return CompositionTrace(t, h, initial, current, tuple(stages))
