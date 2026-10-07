"""Shared accepted-stage records and nonlinear-work statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from contracts.nonlinear import NewtonObserver
from methods._nonlinear import SolveStats
from formulations.gc import _EnergyQuadraturePoint


@dataclass(frozen=True, slots=True)
class MapStage:
    """One signed direct/adjoint stage and the two physical shear inputs."""

    start_time: float
    time: float
    duration: float
    direct: bool
    state_before: np.ndarray
    state_after: np.ndarray
    energy_points: tuple[_EnergyQuadraturePoint, ...]


@dataclass(frozen=True, slots=True)
class CompositionTrace:
    """Numerical trace reused by projection, differentiation and passive energy."""

    start_time: float
    duration: float
    state_before: np.ndarray
    state: np.ndarray
    stages: tuple[MapStage, ...]

    @property
    def energy_points(self) -> tuple[_EnergyQuadraturePoint, ...]:
        """Expose signed shear inputs in their original execution order."""
        return tuple(point for stage in self.stages for point in stage.energy_points)


@dataclass(frozen=True, slots=True)
class ProjectedMapResult:
    """One accepted projection, shared by ABBA and BM4."""

    start_time: float
    duration: float
    state_before: np.ndarray
    state: np.ndarray
    multiplier: np.ndarray
    stats: SolveStats
    trace: CompositionTrace
    retain_energy_points: bool = False

    @property
    def internal_input(self) -> np.ndarray:
        return self.trace.state_before

    @property
    def mapped(self) -> np.ndarray:
        return self.trace.state

    @property
    def energy_points(self) -> tuple[_EnergyQuadraturePoint, ...]:
        return self.trace.energy_points if self.retain_energy_points else ()

    @property
    def iterations(self) -> int:
        return self.stats.iterations

    @property
    def residual_evaluations(self) -> int:
        return self.stats.residual_evaluations

    @property
    def residual_norm(self) -> float:
        return self.stats.residual_norm


def step_statistics(result: ProjectedMapResult, *, include_substeps: bool) -> dict[str, np.ndarray | float | int]:
	"""Extract the single outer solve's metrics without building observer events."""
	stats = result.stats
	multiplier_norm = float(np.linalg.norm(result.multiplier, ord=np.inf))
	metrics: dict[str, np.ndarray | float | int] = {
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
			"substep_nonlinear_iterations": np.asarray([stats.iterations], dtype=int),
			"substep_residual_evaluations": np.asarray([stats.residual_evaluations], dtype=int),
			"substep_nonlinear_residual_norms": np.asarray([stats.residual_norm], dtype=float),
			"substep_nonlinear_tolerances": np.asarray([stats.tolerance], dtype=float),
			"substep_projection_multiplier_norms": np.asarray([multiplier_norm]),
		})
	return metrics


__all__: list[str] = []


class ProjectedMap(Protocol):
    """One physical projection with an explicitly scoped Newton callback."""

    def __call__(
        self, t: float, state: np.ndarray, h: float, *,
        newton_observer: NewtonObserver | None = None,
    ) -> ProjectedMapResult:
        """Evaluate a map; diagnostic callers omit the optional callback."""
