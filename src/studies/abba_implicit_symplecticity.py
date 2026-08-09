"""Comparative symplecticity study for both implicit ABBA formulations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np

from initial_conditions import Area
from potential import Potential
from simulation import ImplicitABBA1, ImplicitABBA2

from ._gc_symplecticity import _run_gc_symplecticity_study
from .abba_symplecticity import (
	ABBASymplecticityConfig,
	ABBASymplecticityResult,
	ABBASymplecticitySummary,
)


IMPLICIT_ABBA_FORMULATIONS = ("implicit_1", "implicit_2")


@dataclass(frozen=True, slots=True)
class ImplicitABBASymplecticityConfig(ABBASymplecticityConfig):
	"""Shared reproducible controls for comparing both implicit formulations."""

	block_prefix: str = "implicit_abba_symplecticity"


@dataclass(frozen=True, slots=True)
class ImplicitABBA1SymplecticityResult(ABBASymplecticityResult):
	"""Symplecticity diagnostics produced by the reduced formulation."""

	method_name: ClassVar[str] = "ImplicitABBA1"
	summary_type: ClassVar[type[ABBASymplecticitySummary]] = (
		ABBASymplecticitySummary
	)


@dataclass(frozen=True, slots=True)
class ImplicitABBA2SymplecticityResult(ABBASymplecticityResult):
	"""Symplecticity diagnostics produced by simultaneous equation (21)."""

	method_name: ClassVar[str] = "ImplicitABBA2"
	summary_type: ClassVar[type[ABBASymplecticitySummary]] = (
		ABBASymplecticitySummary
	)


@dataclass(frozen=True, slots=True)
class ImplicitABBASymplecticityComparison:
	"""Paired runs of the mathematically equivalent implicit formulations."""

	results: Mapping[
		str,
		ImplicitABBA1SymplecticityResult | ImplicitABBA2SymplecticityResult,
	]

	def maximum_state_differences(self) -> Mapping[str, float]:
		"""Return the maximum componentwise trajectory difference for each step."""
		first = self.results["implicit_1"]
		second = self.results["implicit_2"]
		differences: dict[str, float] = {}
		for step in first.steps:
			first_solution = first.solutions[step.label]
			second_solution = second.solutions[step.label]
			if not np.array_equal(first_solution.t, second_solution.t):
				raise ValueError(
					"Implicit ABBA comparison trajectories must share output times."
				)
			if first_solution.states.shape != second_solution.states.shape:
				raise ValueError(
					"Implicit ABBA comparison trajectories must have equal shapes."
				)
			differences[step.label] = float(
				np.max(np.abs(first_solution.states - second_solution.states))
			)
		return MappingProxyType(differences)

	def print_summary(self) -> None:
		"""Print each formulation summary and their trajectory agreement."""
		for name in IMPLICIT_ABBA_FORMULATIONS:
			print(f"\n{name}")
			self.results[name].print_summary()
		print("\nMaximum |implicit_1 - implicit_2| by integration step:")
		for label, difference in self.maximum_state_differences().items():
			print(f"  {label}: {difference:.8e}")


def _implicit_study_metadata(
	config: ImplicitABBASymplecticityConfig,
	metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
	"""Build shared persisted metadata for either implicit formulation."""
	return {
		**dict(metadata or {}),
		"newton_absolute_tolerance": config.newton_absolute_tolerance,
		"newton_relative_tolerance": config.newton_relative_tolerance,
		"newton_max_iterations": config.newton_max_iterations,
		"newton_initial_multiplier": "zero",
		"newton_residual_norm": "infinity",
		"abba_stage_times": "t_n,t_n,t_n_plus_h,t_n_plus_h",
		"step_jacobian": "centered_difference_of_emitted_solver_map",
	}


def run_implicit_abba_1_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: ImplicitABBASymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> ImplicitABBA1SymplecticityResult:
	"""Run a finite-tolerance physical symplecticity audit for implicit ABBA 1."""
	if not isinstance(config, ImplicitABBASymplecticityConfig):
		raise TypeError(
			"`config` must be an ImplicitABBASymplecticityConfig instance."
		)
	method_config = replace(
		config,
		block_prefix=f"{config.block_prefix}_implicit_1",
	)
	return _run_gc_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=method_config,
		method_factory=lambda observer: ImplicitABBA1(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			progress=config.progress,
			step_observer=observer,
		),
		result_type=ImplicitABBA1SymplecticityResult,
		project_root=project_root,
		metadata={
			**_implicit_study_metadata(config, metadata),
			"implicit_formulation": "implicit_1_reduced_equation_11",
		},
	)


def run_implicit_abba_2_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: ImplicitABBASymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> ImplicitABBA2SymplecticityResult:
	"""Run a finite-tolerance physical symplecticity audit for equation (21)."""
	if not isinstance(config, ImplicitABBASymplecticityConfig):
		raise TypeError(
			"`config` must be an ImplicitABBASymplecticityConfig instance."
		)
	method_config = replace(
		config,
		block_prefix=f"{config.block_prefix}_implicit_2",
	)
	return _run_gc_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=method_config,
		method_factory=lambda observer: ImplicitABBA2(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			progress=config.progress,
			step_observer=observer,
		),
		result_type=ImplicitABBA2SymplecticityResult,
		project_root=project_root,
		metadata={
			**_implicit_study_metadata(config, metadata),
			"implicit_formulation": "implicit_2_simultaneous_equation_21",
		},
	)


def run_implicit_abba_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: ImplicitABBASymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> ImplicitABBASymplecticityComparison:
	"""Run synchronized finite-tolerance audits for implicit 1 and implicit 2."""
	if not isinstance(config, ImplicitABBASymplecticityConfig):
		raise TypeError(
			"`config` must be an ImplicitABBASymplecticityConfig instance."
		)
	first = run_implicit_abba_1_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=config,
		project_root=project_root,
		metadata=metadata,
	)
	second = run_implicit_abba_2_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=config,
		project_root=project_root,
		metadata=metadata,
	)
	return ImplicitABBASymplecticityComparison(
		results=MappingProxyType(
			{
				"implicit_1": first,
				"implicit_2": second,
			}
		)
	)


__all__ = [
	"IMPLICIT_ABBA_FORMULATIONS",
	"ImplicitABBA1SymplecticityResult",
	"ImplicitABBA2SymplecticityResult",
	"ImplicitABBASymplecticityComparison",
	"ImplicitABBASymplecticityConfig",
	"run_implicit_abba_1_symplecticity_study",
	"run_implicit_abba_2_symplecticity_study",
	"run_implicit_abba_symplecticity_study",
]
