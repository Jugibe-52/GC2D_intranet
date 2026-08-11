"""Comparative symplecticity studies for the two implicit BM4 formulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, cast

import numpy as np

from initial_conditions import Area
from potential import Potential
from simulation import BM4Implicit1, BM4Implicit2

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
	coupling_frequency: float = float(np.pi / 8)
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
class BM4Implicit1SymplecticityResult(GCSymplecticityResult):
	"""Symplecticity data from the reduced BM4 projection solve."""

	method_name: ClassVar[str] = "BM4Implicit1"
	summary_type: ClassVar[type[GCSymplecticitySummary]] = (
		BM4ImplicitSymplecticitySummary
	)

	def summaries(self) -> tuple[BM4ImplicitSymplecticitySummary, ...]:
		"""Return typed summaries in configured step order."""
		return cast(
			tuple[BM4ImplicitSymplecticitySummary, ...],
			GCSymplecticityResult.summaries(self),
		)


@dataclass(frozen=True, slots=True)
class BM4Implicit2SymplecticityResult(BM4Implicit1SymplecticityResult):
	"""Symplecticity data from the simultaneous BM4 projection solve."""

	method_name: ClassVar[str] = "BM4Implicit2"


BM4ImplicitStudyResult = (
	BM4Implicit1SymplecticityResult | BM4Implicit2SymplecticityResult
)


@dataclass(frozen=True, slots=True)
class BM4ImplicitSymplecticityComparison:
	"""Aligned results for reduced and simultaneous projected BM4 runs."""

	results: Mapping[str, BM4ImplicitStudyResult]

	def __post_init__(self) -> None:
		"""Require one result from each projected BM4 formulation."""
		values = dict(self.results)
		if set(values) != {"implicit_1", "implicit_2"}:
			raise ValueError(
				"BM4 implicit comparison requires 'implicit_1' and 'implicit_2'."
			)
		object.__setattr__(self, "results", MappingProxyType(values))

	def maximum_state_differences(self) -> Mapping[str, float]:
		"""Return maximum aligned state differences for every integration step."""
		first = self.results["implicit_1"]
		second = self.results["implicit_2"]
		differences: dict[str, float] = {}
		for step in first.steps:
			first_solution = first.solutions[step.label]
			second_solution = second.solutions[step.label]
			if not np.array_equal(first_solution.t, second_solution.t):
				raise ValueError("Projected BM4 comparison times are not aligned.")
			differences[step.label] = float(
				np.max(np.abs(first_solution.states - second_solution.states))
			)
		return MappingProxyType(differences)


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


def run_bm4_implicit_1_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: BM4ImplicitSymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> BM4Implicit1SymplecticityResult:
	"""Run reduced projected BM4 and persist physical-flow diagnostics."""
	return _run_gc_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=config,
		method_factory=lambda observer: BM4Implicit1(
			coupling_frequency=config.coupling_frequency,
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			newton_jacobian_relative_step=config.newton_jacobian_relative_step,
			progress=config.progress,
			step_observer=observer,
		),
		result_type=BM4Implicit1SymplecticityResult,
		project_root=project_root,
		metadata={**dict(metadata or {}), **_solver_metadata(config)},
	)


def run_bm4_implicit_2_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: BM4ImplicitSymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> BM4Implicit2SymplecticityResult:
	"""Run simultaneous projected BM4 and persist physical-flow diagnostics."""
	return _run_gc_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=config,
		method_factory=lambda observer: BM4Implicit2(
			coupling_frequency=config.coupling_frequency,
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			newton_jacobian_relative_step=config.newton_jacobian_relative_step,
			progress=config.progress,
			step_observer=observer,
		),
		result_type=BM4Implicit2SymplecticityResult,
		project_root=project_root,
		metadata={**dict(metadata or {}), **_solver_metadata(config)},
	)


def run_bm4_implicit_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: BM4ImplicitSymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> BM4ImplicitSymplecticityComparison:
	"""Run both projected BM4 formulations on identical physical grids."""
	first = run_bm4_implicit_1_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=config,
		project_root=project_root,
		metadata=metadata,
	)
	second = run_bm4_implicit_2_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=config,
		project_root=project_root,
		metadata=metadata,
	)
	return BM4ImplicitSymplecticityComparison(
		results=MappingProxyType({"implicit_1": first, "implicit_2": second})
	)


__all__ = [
	"BM4Implicit1SymplecticityResult",
	"BM4Implicit2SymplecticityResult",
	"BM4ImplicitSymplecticityComparison",
	"BM4ImplicitSymplecticityConfig",
	"BM4ImplicitSymplecticitySummary",
	"run_bm4_implicit_1_symplecticity_study",
	"run_bm4_implicit_2_symplecticity_study",
	"run_bm4_implicit_symplecticity_study",
]
