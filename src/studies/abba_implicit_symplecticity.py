"""Configurable symplecticity diagnostics for both implicit ABBA formulations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, TypeAlias

import numpy as np

from diagnostics.symplecticity import StepJacobianMethod
from initial_conditions import Area
from potential import Potential
from simulation import (
	ABBA_PROJECTION_FORMULATIONS,
	ABBA2Implicit,
	ProjectionFormulation,
)

from ._gc_symplecticity import (
	_run_gc_symplecticity_observers,
	_run_gc_symplecticity_study,
)
from .abba_symplecticity import (
	ABBASymplecticityConfig,
	ABBASymplecticityResult,
	ABBASymplecticitySummary,
)


ImplicitABBAFormulation: TypeAlias = ProjectionFormulation
IMPLICIT_ABBA_FORMULATIONS = ABBA_PROJECTION_FORMULATIONS
IMPLICIT_ABBA_JACOBIAN_METHODS: tuple[StepJacobianMethod, ...] = (
	"finite_difference",
	"implicit_function",
	"stage_increment",
)
_LABEL = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class ImplicitABBAObserverConfig:
	"""Select one implicit formulation and one step-Jacobian observer."""

	label: str
	formulation: ImplicitABBAFormulation
	jacobian_method: StepJacobianMethod

	def __post_init__(self) -> None:
		"""Validate a persistence-safe unique-case label and supported choices."""
		if not isinstance(self.label, str) or not _LABEL.fullmatch(self.label):
			raise ValueError(
				"Observer `label` may contain only letters, numbers, '_' and '-'."
			)
		if self.formulation not in IMPLICIT_ABBA_FORMULATIONS:
			raise ValueError("Unknown implicit ABBA formulation.")
		if self.jacobian_method not in IMPLICIT_ABBA_JACOBIAN_METHODS:
			raise ValueError("Unknown implicit ABBA Jacobian method.")


DEFAULT_IMPLICIT_ABBA_OBSERVERS = tuple(
	ImplicitABBAObserverConfig(
		label=f"{formulation}_{jacobian_method}",
		formulation=formulation,
		jacobian_method=jacobian_method,
	)
	for formulation in IMPLICIT_ABBA_FORMULATIONS
	for jacobian_method in IMPLICIT_ABBA_JACOBIAN_METHODS
)


@dataclass(frozen=True, slots=True)
class ImplicitABBASymplecticityConfig(ABBASymplecticityConfig):
	"""Reproducible controls and arbitrary implicit-method observer combinations."""

	block_prefix: str = "implicit_abba_symplecticity"
	observers: tuple[ImplicitABBAObserverConfig, ...] = (
		DEFAULT_IMPLICIT_ABBA_OBSERVERS
	)

	def __post_init__(self) -> None:
		"""Validate common controls and require unique configured observers."""
		ABBASymplecticityConfig.__post_init__(self)
		observers = tuple(self.observers)
		if not observers or any(
			not isinstance(observer, ImplicitABBAObserverConfig)
			for observer in observers
		):
			raise ValueError(
				"`observers` must contain at least one ImplicitABBAObserverConfig."
			)
		labels = [observer.label for observer in observers]
		if len(set(labels)) != len(labels):
			raise ValueError("Implicit ABBA observer labels must be unique.")
		object.__setattr__(self, "observers", observers)


@dataclass(frozen=True, slots=True)
class ABBA2ReducedMultiplierSymplecticityResult(ABBASymplecticityResult):
	"""Symplecticity diagnostics produced by the reduced formulation."""

	method_name: ClassVar[str] = "ABBA2Implicit[reduced_multiplier]"
	projection_formulation: ClassVar[ProjectionFormulation] = "reduced_multiplier"
	summary_type: ClassVar[type[ABBASymplecticitySummary]] = (
		ABBASymplecticitySummary
	)


@dataclass(frozen=True, slots=True)
class ABBA2SimultaneousStateMultiplierSymplecticityResult(
	ABBASymplecticityResult
):
	"""Symplecticity diagnostics produced by simultaneous equation (21)."""

	method_name: ClassVar[str] = "ABBA2Implicit[simultaneous_state_multiplier]"
	projection_formulation: ClassVar[ProjectionFormulation] = (
		"simultaneous_state_multiplier"
	)
	summary_type: ClassVar[type[ABBASymplecticitySummary]] = (
		ABBASymplecticitySummary
	)


ImplicitABBAStudyResult: TypeAlias = (
	ABBA2ReducedMultiplierSymplecticityResult
	| ABBA2SimultaneousStateMultiplierSymplecticityResult
)


@dataclass(frozen=True, slots=True)
class ImplicitABBASymplecticityComparison:
	"""Observer-labeled results from one integration per configured formulation."""

	observers: tuple[ImplicitABBAObserverConfig, ...]
	results: Mapping[str, ImplicitABBAStudyResult]

	def __post_init__(self) -> None:
		"""Require results that match every configured formulation and diagnostic."""
		labels = tuple(observer.label for observer in self.observers)
		if tuple(self.results) != labels:
			raise ValueError("`results` must follow the configured observer order.")
		for observer in self.observers:
			result = self.results[observer.label]
			expected_type = (
				ABBA2ReducedMultiplierSymplecticityResult
				if observer.formulation == "reduced_multiplier"
				else ABBA2SimultaneousStateMultiplierSymplecticityResult
			)
			if not isinstance(result, expected_type):
				raise TypeError(
					f"Result `{observer.label}` does not match "
					f"formulation '{observer.formulation}'."
				)
			if result.jacobian_method != observer.jacobian_method:
				raise ValueError(
					f"Result `{observer.label}` does not use its configured "
					"Jacobian method."
				)

	def maximum_state_differences(self) -> Mapping[str, float]:
		"""Compare the two solver formulations independently of observer choice."""
		by_formulation: dict[ImplicitABBAFormulation, ImplicitABBAStudyResult] = {}
		for observer in self.observers:
			by_formulation.setdefault(
				observer.formulation,
				self.results[observer.label],
			)
		if any(
			formulation not in by_formulation
			for formulation in IMPLICIT_ABBA_FORMULATIONS
		):
			return MappingProxyType({})
		first = by_formulation["reduced_multiplier"]
		second = by_formulation["simultaneous_state_multiplier"]
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
		"""Print every method-observer combination and trajectory agreement."""
		for observer in self.observers:
			print(
				f"\n{observer.label}: {observer.formulation} / "
				f"{observer.jacobian_method}"
			)
			self.results[observer.label].print_summary()
		if self.maximum_state_differences():
			print("\nMaximum reduced-versus-simultaneous difference by step:")
			for label, difference in self.maximum_state_differences().items():
				print(f"  {label}: {difference:.8e}")


def _implicit_study_metadata(
	config: ImplicitABBASymplecticityConfig,
	metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
	"""Build persisted metadata shared by both implicit formulations."""
	return {
		**dict(metadata or {}),
		"newton_absolute_tolerance": config.newton_absolute_tolerance,
		"newton_relative_tolerance": config.newton_relative_tolerance,
		"newton_max_iterations": config.newton_max_iterations,
		"newton_initial_multiplier": "zero",
		"newton_residual_norm": "infinity",
		"abba_stage_times": "t_n,t_n,t_n_plus_h,t_n_plus_h",
	}


def run_abba2_reduced_multiplier_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: ImplicitABBASymplecticityConfig,
	jacobian_method: StepJacobianMethod = "finite_difference",
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> ABBA2ReducedMultiplierSymplecticityResult:
	"""Run one Jacobian diagnostic for the reduced-multiplier ABBA2 solve."""
	if not isinstance(config, ImplicitABBASymplecticityConfig):
		raise TypeError(
			"`config` must be an ImplicitABBASymplecticityConfig instance."
		)
	method_config = replace(
		config,
		block_prefix=(
			f"{config.block_prefix}_reduced_multiplier_{jacobian_method}"
		),
	)
	return _run_gc_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=method_config,
		method_factory=lambda observer: ABBA2Implicit(
			projection_formulation="reduced_multiplier",
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			progress=config.progress,
			step_observer=observer,
		),
		result_type=ABBA2ReducedMultiplierSymplecticityResult,
		project_root=project_root,
		metadata={
			**_implicit_study_metadata(config, metadata),
			"projection_formulation": "reduced_multiplier",
			"step_jacobian_method": jacobian_method,
			"step_jacobian_scope": (
				"emitted_finite_tolerance_solver_map"
				if jacobian_method == "finite_difference"
				else "ideal_converged_projection_root"
			),
		},
		jacobian_method=jacobian_method,
	)


def run_abba2_simultaneous_state_multiplier_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: ImplicitABBASymplecticityConfig,
	jacobian_method: StepJacobianMethod = "finite_difference",
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> ABBA2SimultaneousStateMultiplierSymplecticityResult:
	"""Run one Jacobian diagnostic for the simultaneous ABBA2 solve."""
	if not isinstance(config, ImplicitABBASymplecticityConfig):
		raise TypeError(
			"`config` must be an ImplicitABBASymplecticityConfig instance."
		)
	method_config = replace(
		config,
		block_prefix=(
			f"{config.block_prefix}_simultaneous_state_multiplier_{jacobian_method}"
		),
	)
	return _run_gc_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=method_config,
		method_factory=lambda observer: ABBA2Implicit(
			projection_formulation="simultaneous_state_multiplier",
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			progress=config.progress,
			step_observer=observer,
		),
		result_type=ABBA2SimultaneousStateMultiplierSymplecticityResult,
		project_root=project_root,
		metadata={
			**_implicit_study_metadata(config, metadata),
			"projection_formulation": "simultaneous_state_multiplier",
			"step_jacobian_method": jacobian_method,
			"step_jacobian_scope": (
				"emitted_finite_tolerance_solver_map"
				if jacobian_method == "finite_difference"
				else "ideal_converged_projection_root"
			),
		},
		jacobian_method=jacobian_method,
	)


def _run_formulation_observers(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: ImplicitABBASymplecticityConfig,
	formulation: ImplicitABBAFormulation,
	observer_methods: Mapping[str, StepJacobianMethod],
	project_root: str | Path | None,
	metadata: Mapping[str, Any] | None,
) -> Mapping[str, ImplicitABBAStudyResult]:
	"""Run all observers for one formulation against the same trajectories."""
	method_config = replace(
		config,
		block_prefix=f"{config.block_prefix}_{formulation}",
	)
	common_metadata = {
		**_implicit_study_metadata(config, metadata),
		"projection_formulation": formulation,
	}
	if formulation == "reduced_multiplier":
		return _run_gc_symplecticity_observers(
			potential,
			area,
			notebook_path=notebook_path,
			config=method_config,
			method_factory=lambda observer: ABBA2Implicit(
				projection_formulation=formulation,
				newton_absolute_tolerance=config.newton_absolute_tolerance,
				newton_relative_tolerance=config.newton_relative_tolerance,
				newton_max_iterations=config.newton_max_iterations,
				progress=config.progress,
				step_observer=observer,
			),
			result_type=ABBA2ReducedMultiplierSymplecticityResult,
			project_root=project_root,
			metadata=common_metadata,
			jacobian_methods=observer_methods,
		)
	return _run_gc_symplecticity_observers(
		potential,
		area,
		notebook_path=notebook_path,
		config=method_config,
		method_factory=lambda observer: ABBA2Implicit(
			projection_formulation=formulation,
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			progress=config.progress,
			step_observer=observer,
		),
		result_type=ABBA2SimultaneousStateMultiplierSymplecticityResult,
		project_root=project_root,
		metadata=common_metadata,
		jacobian_methods=observer_methods,
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
	"""Run configured observers with one integration per implicit formulation."""
	if not isinstance(config, ImplicitABBASymplecticityConfig):
		raise TypeError(
			"`config` must be an ImplicitABBASymplecticityConfig instance."
		)
	results: dict[str, ImplicitABBAStudyResult] = {}
	for formulation in IMPLICIT_ABBA_FORMULATIONS:
		methods = {
			observer.label: observer.jacobian_method
			for observer in config.observers
			if observer.formulation == formulation
		}
		if not methods:
			continue
		results.update(
			_run_formulation_observers(
				potential,
				area,
				notebook_path=notebook_path,
				config=config,
				formulation=formulation,
				observer_methods=methods,
				project_root=project_root,
				metadata=metadata,
			)
		)
	ordered_results = {
		observer.label: results[observer.label] for observer in config.observers
	}
	return ImplicitABBASymplecticityComparison(
		observers=config.observers,
		results=MappingProxyType(ordered_results),
	)


__all__ = [
	"DEFAULT_IMPLICIT_ABBA_OBSERVERS",
	"IMPLICIT_ABBA_FORMULATIONS",
	"IMPLICIT_ABBA_JACOBIAN_METHODS",
	"ABBA2ReducedMultiplierSymplecticityResult",
	"ABBA2SimultaneousStateMultiplierSymplecticityResult",
	"ImplicitABBAFormulation",
	"ImplicitABBAObserverConfig",
	"ImplicitABBASymplecticityComparison",
	"ImplicitABBASymplecticityConfig",
	"run_abba2_reduced_multiplier_symplecticity_study",
	"run_abba2_simultaneous_state_multiplier_symplecticity_study",
	"run_implicit_abba_symplecticity_study",
]
