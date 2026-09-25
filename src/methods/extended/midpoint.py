"""Arithmetic diagonal projection shared by the explicit ABBA and BM4 methods."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from formulations.gc import GCDoubledMaps
from methods.extended.composition import Composition, compose
from methods.extended.records import CompositionTrace, _ABBAStages


@dataclass(frozen=True, slots=True)
class MidpointResult:
    """Physical mean, copy separation and accepted composition trace."""

    state: np.ndarray
    copy_separation_norm: float
    trace: CompositionTrace

    @property
    def stages(self) -> _ABBAStages:
        """Expose the established ABBA2 diagnostic view when requested."""
        return self.trace.maps[0].stages


def midpoint_step(maps: GCDoubledMaps, recipe: Composition,
                  t: float, state: np.ndarray, h: float) -> MidpointResult:
    """Advance diagonal copies and average once after the complete recipe."""
    trace = compose(maps, recipe, t, np.concatenate((state, state)), h)
    size = state.size
    first, second = trace.state[:size], trace.state[size:]
    return MidpointResult((first + second) / 2,
                          float(np.linalg.norm(first - second, ord=np.inf)), trace)
