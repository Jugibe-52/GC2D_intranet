"""Shared accepted-step records and observer-independent ABBA diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .._nonlinear import SolveStats
from .maps.physical import _ABBAStages
from .projection_extended import _FullProjectedStep


@dataclass(frozen=True, slots=True)
class PhysicalBaseMapTrace:
	"""Accepted physical stages with their signed duration and start time."""

	start_time: float
	duration: float
	stages: _ABBAStages


@dataclass(frozen=True, slots=True)
class PhysicalProjectionTrace:
	"""One base map, or the continuous maps enclosed by an outer projection."""

	maps: tuple[PhysicalBaseMapTrace, ...]


@dataclass(frozen=True, slots=True)
class ExtendedProjectionTrace:
	"""Accepted R8 base-map data for one full-diagonal projection.

	The kernel retains the complete selected map and its derivative callback.
	An outer composition still owns only one projection and one base-map event.
	"""

	result: _FullProjectedStep
	coefficients: tuple[float, ...]


ProjectionTrace: TypeAlias = PhysicalProjectionTrace | ExtendedProjectionTrace


@dataclass(frozen=True, slots=True)
class ProjectedMapResult:
	"""One accepted projection, independent of the enclosing method order."""

	start_time: float
	duration: float
	state_before: np.ndarray
	state: np.ndarray
	multiplier: np.ndarray
	stats: SolveStats
	trace: ProjectionTrace


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
