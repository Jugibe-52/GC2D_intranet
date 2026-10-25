"""Arithmetic diagonal projection shared by the explicit ABBA and BM4 methods."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from contracts.arrays import BackendArray
from types import ModuleType
from contracts.arrays import array_namespace, checked_finite
from methods.extended.maps import GCDoubledMaps
from methods.extended.core.composition import Composition, compose
from methods.extended.core.records import CompositionTrace


@dataclass(frozen=True, slots=True)
class MidpointResult:
    """Physical mean, copy separation and accepted composition trace."""

    state: BackendArray
    copy_separation_norm: float
    trace: CompositionTrace


def midpoint_step(maps: GCDoubledMaps, recipe: Composition,
                  t: float, state: BackendArray, h: float) -> MidpointResult:
    """Advance diagonal copies and average once after the complete recipe."""
    xp: ModuleType = array_namespace(state)
    candidate: BackendArray = maps.state_formulation.lift(state)
    trace: CompositionTrace = compose(maps, recipe, t, candidate, h)
    first: BackendArray
    second: BackendArray
    first, second = maps.state_formulation.split_copies(trace.state)
    return MidpointResult((first + second) / 2,
                          xp.linalg.norm(first - second, ord=np.inf), trace)
