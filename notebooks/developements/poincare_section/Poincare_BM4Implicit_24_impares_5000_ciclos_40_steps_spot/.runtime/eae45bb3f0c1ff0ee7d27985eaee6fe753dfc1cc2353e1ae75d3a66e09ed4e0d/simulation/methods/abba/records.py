"""Shared accepted-step records and observer-independent ABBA diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from ..._result import DiagnosticValue
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


@dataclass(slots=True)
class StepMetrics:
	"""Small numerical rows, collected without allocating observer events."""

	iterations: list[list[int]] = field(default_factory=list)
	evaluations: list[list[int]] = field(default_factory=list)
	residuals: list[list[float]] = field(default_factory=list)
	tolerances: list[list[float]] = field(default_factory=list)
	multipliers: list[list[float]] = field(default_factory=list)

	def finalize(self, *, include_substeps: bool) -> dict[str, DiagnosticValue]:
		"""Aggregate work and retain the residual/tolerance from one solve."""
		iterations = np.asarray(self.iterations, dtype=int)
		evaluations = np.asarray(self.evaluations, dtype=int)
		residuals = np.asarray(self.residuals, dtype=float)
		tolerances = np.asarray(self.tolerances, dtype=float)
		multipliers = np.asarray(self.multipliers, dtype=float)
		worst = np.argmax(residuals / tolerances, axis=1)
		rows = np.arange(residuals.shape[0])
		result: dict[str, DiagnosticValue] = {
			"nonlinear_iterations": np.sum(iterations, axis=1),
			"residual_evaluations": np.sum(evaluations, axis=1),
			"nonlinear_residual_norms": residuals[rows, worst],
			"nonlinear_tolerances": tolerances[rows, worst],
			"projection_multiplier_norms": np.max(multipliers, axis=1),
		}
		if include_substeps:
			result.update({
				"substep_nonlinear_iterations": iterations,
				"substep_residual_evaluations": evaluations,
				"substep_nonlinear_residual_norms": residuals,
				"substep_nonlinear_tolerances": tolerances,
				"substep_projection_multiplier_norms": multipliers,
			})
		return result


def record_completed_step(metrics: StepMetrics, result: StepResult) -> None:
	"""Append the numerical records of one main-grid advance."""
	projections = result.projections
	metrics.iterations.append([p.stats.iterations for p in projections])
	metrics.evaluations.append([p.stats.residual_evaluations for p in projections])
	metrics.residuals.append([p.stats.residual_norm for p in projections])
	metrics.tolerances.append([p.stats.tolerance for p in projections])
	metrics.multipliers.append([
		float(np.linalg.norm(p.multiplier, ord=np.inf)) for p in projections
	])


__all__: list[str] = []
