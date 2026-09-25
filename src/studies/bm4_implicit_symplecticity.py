"""Symplecticity study for the physical Hairer-projected BM4 method."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, cast

import numpy as np

from initial_conditions import Area
from potential import Potential
from methods.extended.bm4 import BM4Implicit

from ._gc_symplecticity import (
	GCSymplecticityResult,
	_run_gc_symplecticity_study,
)
from ._gc_symplecticity_models import (
	GCSymplecticityConfig,
	GCSymplecticitySummary,
)
from ._validation import nonnegative_finite, positive_finite, positive_integer


@dataclass(frozen=True, slots=True)
class BM4ImplicitSymplecticityConfig(GCSymplecticityConfig):
	"""Reproducible grids and nonlinear controls for projected BM4 studies."""

	block_prefix: str = "bm4_implicit_symplecticity"
	coupling_frequency: float = 0.0
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))

	def __post_init__(self) -> None:
		"""Validate common study grids and projected-BM4 solver controls."""
		GCSymplecticityConfig.__post_init__(self)
		object.__setattr__(
			self,
			"coupling_frequency",
			nonnegative_finite(self.coupling_frequency, "coupling_frequency"),
		)
		for name in (
			"newton_absolute_tolerance",
			"newton_relative_tolerance",
			"newton_jacobian_relative_step",
		):
			object.__setattr__(self, name, positive_finite(getattr(self, name), name))
		object.__setattr__(
			self,
			"newton_max_iterations",
			positive_integer(self.newton_max_iterations, "newton_max_iterations"),
		)


@dataclass(frozen=True, slots=True)
class BM4ImplicitSymplecticitySummary(GCSymplecticitySummary):
	"""Maximum physical and nonlinear diagnostics for one projected BM4 step."""


@dataclass(frozen=True, slots=True)
class BM4ImplicitSymplecticityResult(GCSymplecticityResult):
	"""Symplecticity data from the reduced physical BM4 projection solve."""

	method_name: ClassVar[str] = "BM4Implicit"
	summary_type: ClassVar[type[GCSymplecticitySummary]] = (
		BM4ImplicitSymplecticitySummary
	)

	def summaries(self) -> tuple[BM4ImplicitSymplecticitySummary, ...]:
		"""Return typed summaries in configured step order."""
		return cast(
			tuple[BM4ImplicitSymplecticitySummary, ...],
			GCSymplecticityResult.summaries(self),
		)


def _solver_metadata(config: BM4ImplicitSymplecticityConfig) -> dict[str, Any]:
	"""Return the complete projected-BM4 solver configuration for persistence."""
	return {
		"coupling_frequency": config.coupling_frequency,
		"newton_absolute_tolerance": config.newton_absolute_tolerance,
		"newton_relative_tolerance": config.newton_relative_tolerance,
		"newton_max_iterations": config.newton_max_iterations,
		"newton_jacobian_relative_step": config.newton_jacobian_relative_step,
		"bm4_projection_scope": "one_complete_twelve_stage_cycle",
		"step_jacobian": "centered_difference_of_emitted_solver_map",
	}


def run_bm4_implicit_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: BM4ImplicitSymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> BM4ImplicitSymplecticityResult:
	"""Run physical reduced-projection BM4 and persist flow diagnostics."""
	return _run_gc_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=config,
		method_factory=lambda observer: BM4Implicit(
			coupling_frequency=config.coupling_frequency,
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			newton_jacobian_relative_step=config.newton_jacobian_relative_step,
			progress=config.progress,
			step_observer=observer,
		),
		result_type=BM4ImplicitSymplecticityResult,
		project_root=project_root,
		metadata={**dict(metadata or {}), **_solver_metadata(config)},
	)


__all__ = [
	"BM4ImplicitSymplecticityConfig",
	"BM4ImplicitSymplecticityResult",
	"BM4ImplicitSymplecticitySummary",
	"run_bm4_implicit_symplecticity_study",
]
