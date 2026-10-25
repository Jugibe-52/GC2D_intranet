"""Shared accepted-stage records and nonlinear-work statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from contracts.arrays import BackendArray
from types import ModuleType
from contracts.arrays import array_namespace, checked_finite

from solvers.nonlinear import SolveStats
from contracts.methods.quadrature import StageQuadrature

QuadraturePoint = tuple[float, float, BackendArray]
from methods.extended.core.operations import OperationName


@dataclass(frozen=True, slots=True)
class OperationStage:
    """One elementary flow; a spatial source is retained only for A or B."""

    operation: OperationName
    time: float
    duration: Any
    state_before: BackendArray
    state_after: BackendArray
    source: BackendArray | None

    @property
    def end_time(self) -> float:
        return self.time + self.duration if self.operation == 'T' else self.time


@dataclass(frozen=True, slots=True)
class MapStage:
    """One signed direct/adjoint stage and the two physical shear inputs."""

    start_time: float
    time: float
    duration: Any
    direct: bool
    state_before: BackendArray
    state_after: BackendArray
    quadrature_points: tuple[QuadraturePoint, ...]


@dataclass(frozen=True, slots=True)
class CompositionTrace:
    """Numerical trace reused by projection, differentiation and passive energy."""

    start_time: float
    duration: Any
    state_before: BackendArray
    state: BackendArray
    stages: tuple[MapStage, ...]
    operations: tuple[OperationStage, ...]
    time: float

    @property
    def quadrature_points(self) -> tuple[QuadraturePoint, ...]:
        """Expose signed shear inputs in their original execution order."""
        return tuple((stage.time, stage.duration, stage.source)
                     for stage in self.operations if stage.source is not None)

    @property
    def quadrature(self) -> StageQuadrature:
        """Normalize signed doubled shear weights to one physical copy."""
        points = self.quadrature_points
        return StageQuadrature(tuple(p[0] for p in points), tuple(p[2] for p in points),
                               tuple(p[1] / 2 for p in points), reduction_scale=2.)


@dataclass(frozen=True, slots=True)
class ProjectedMapResult:
    """One accepted projection, shared by ABBA and BM4."""

    start_time: float
    duration: Any
    state_before: BackendArray
    state: BackendArray
    multiplier: BackendArray
    stats: SolveStats
    trace: CompositionTrace

    @property
    def internal_input(self) -> BackendArray:
        return self.trace.state_before

    @property
    def mapped(self) -> BackendArray:
        return self.trace.state

    @property
    def iterations(self) -> int:
        return self.stats.iterations

    @property
    def residual_evaluations(self) -> int:
        return self.stats.residual_evaluations

    @property
    def residual_norm(self) -> float:
        return self.stats.residual_norm


def step_statistics(result: ProjectedMapResult, *, include_substeps: bool) -> dict[str, BackendArray | float | int]:
	"""Extract the single outer solve's metrics without building observer events."""
	xp: ModuleType = array_namespace(result.state)
	stats = result.stats
	multiplier_norm: float = xp.linalg.norm(result.multiplier, ord=np.inf)
	metrics: dict[str, BackendArray | float | int] = {
		"nonlinear_iterations": stats.iterations,
		"residual_evaluations": stats.residual_evaluations,
		"nonlinear_residual_norms": stats.residual_norm,
		"nonlinear_tolerances": stats.tolerance,
		"projection_multiplier_norms": multiplier_norm,
	}
	if include_substeps:
		# Preserve the published one-column histories for ABBA4 and ABBA6.
		# Their unprojected stage pairs do not represent additional solves.
		metrics.update({
			"substep_nonlinear_iterations": xp.asarray([stats.iterations], dtype=int),
			"substep_residual_evaluations": xp.asarray([stats.residual_evaluations], dtype=int),
			"substep_nonlinear_residual_norms": xp.asarray([stats.residual_norm], dtype=float),
			"substep_nonlinear_tolerances": xp.asarray([stats.tolerance], dtype=float),
			"substep_projection_multiplier_norms": xp.asarray([multiplier_norm]),
		})
	return metrics


__all__: list[str] = []
