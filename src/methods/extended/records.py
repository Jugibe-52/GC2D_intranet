"""Shared accepted-step records and observer-independent ABBA diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import TypeAlias

import numpy as np

from methods._nonlinear import SolveStats
from formulations.gc import _EnergyQuadraturePoint


@dataclass(frozen=True, slots=True)
class _ABBAStages:
    """Compatibility view of the four shears in an accepted ABBA pair."""

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

    @property
    def maps(self) -> tuple[PhysicalBaseMapTrace, ...]:
        """Adapt uncoupled ABBA pairs for the existing observer record types."""
        result = []
        for first, last in zip(self.stages[::2], self.stages[1::2], strict=True):
            size = first.state_before.size // 2
            u0, v0 = first.state_before[:size], first.state_before[size:]
            uf, vf = last.state_after[:size], last.state_after[size:]
            stages = _ABBAStages(u0, v0, first.energy_points[1][2], vf, uf, uf - vf)
            result.append(PhysicalBaseMapTrace(first.start_time, first.duration + last.duration, stages))
        return tuple(result)


# Existing private ABBA consumers can keep their trace type checks.
PhysicalProjectionTrace = CompositionTrace


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


@dataclass(frozen=True, slots=True)
class StepResult:
	"""Complete workspace advance and its ordered accepted projections."""

	next_workspace: np.ndarray
	projections: tuple[ProjectedMapResult, ...]



def step_statistics(result: StepResult, *, include_substeps: bool) -> dict[str, np.ndarray | float | int]:
	"""Extract one metric row without accumulating history or building events."""
	projections = result.projections
	iterations = np.asarray([p.stats.iterations for p in projections], dtype=int)
	evaluations = np.asarray([p.stats.residual_evaluations for p in projections], dtype=int)
	residuals = np.asarray([p.stats.residual_norm for p in projections], dtype=float)
	tolerances = np.asarray([p.stats.tolerance for p in projections], dtype=float)
	multipliers = np.asarray([float(np.linalg.norm(p.multiplier, ord=np.inf)) for p in projections])
	worst = int(np.argmax(residuals / tolerances))
	metrics: dict[str, np.ndarray | float | int] = {
		"nonlinear_iterations": int(np.sum(iterations)),
		"residual_evaluations": int(np.sum(evaluations)),
		"nonlinear_residual_norms": float(residuals[worst]),
		"nonlinear_tolerances": float(tolerances[worst]),
		"projection_multiplier_norms": float(np.max(multipliers)),
	}
	if include_substeps:
		metrics.update({
			"substep_nonlinear_iterations": iterations,
			"substep_residual_evaluations": evaluations,
			"substep_nonlinear_residual_norms": residuals,
			"substep_nonlinear_tolerances": tolerances,
			"substep_projection_multiplier_norms": multipliers,
		})
	return metrics


__all__: list[str] = []


ProjectedMap: TypeAlias = Callable[[float, np.ndarray, float], ProjectedMapResult]
StepSolver: TypeAlias = Callable[[float, np.ndarray, float], tuple[ProjectedMapResult, ...]]
