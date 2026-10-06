"""One-pass runtime and trajectory comparison for the three ABBA methods."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from collections.abc import Iterable

from ._validation import (
	unpacked_time_span,
	validate_block_prefix,
	integer_ratio,
	nonnegative_finite,
	positive_finite,
	positive_integer,
)

from studies._trajectory_distances import minimum_image_displacement

from initial_conditions import Area
from potential import Potential
from solution import Solution

from .abba_midpoint_symplecticity import (
	ABBA2MidpointSymplecticityConfig,
	ABBA2MidpointSymplecticityResult,
	run_abba2_midpoint_symplecticity_study,
)
from .abba_implicit_symplecticity import (
	ABBA2ReducedMultiplierSymplecticityResult,
	ABBA2SimultaneousStateMultiplierSymplecticityResult,
	ImplicitABBASymplecticityConfig,
	run_abba2_reduced_multiplier_symplecticity_study,
	run_abba2_simultaneous_state_multiplier_symplecticity_study,
)
from .area_comparison import AreaStep


ABBA_METHOD_NAMES = (
	"ABBA2Midpoint",
	"ABBA2Implicit[reduced_multiplier]",
	"ABBA2Implicit[simultaneous_state_multiplier]",
)
ABBAComparisonStudy = (
	ABBA2MidpointSymplecticityResult
	| ABBA2ReducedMultiplierSymplecticityResult
	| ABBA2SimultaneousStateMultiplierSymplecticityResult
)


@dataclass(frozen=True, slots=True)
class ABBAComparisonConfig:
	"""Shared grid, diagnostics, and solver parameters for one run per method."""

	integration_step: float
	step_label: str
	t_span: tuple[float, float]
	save_interval: float
	rho: float | None = None
	chunk_size: int = 16
	progress: bool = False
	block_prefix: str = "abba_method_comparison"
	finite_difference_relative_step: float | None = None
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12

	def __post_init__(self) -> None:
		"""Validate the common integration, observation, and Newton grids."""
		object.__setattr__(
			self,
			"integration_step",
			positive_finite(self.integration_step, "integration_step"),
		)
		if not isinstance(self.step_label, str) or not self.step_label.strip():
			raise ValueError("`step_label` must be a non-empty string.")

		start, stop = unpacked_time_span(self.t_span)
		object.__setattr__(self, "t_span", (start, stop))

		save_interval = positive_finite(self.save_interval, "save_interval")
		object.__setattr__(self, "save_interval", save_interval)
		if self.rho is not None:
			object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		integer_ratio(stop - start, save_interval, "duration / save_interval")
		integer_ratio(
			save_interval,
			self.integration_step,
			"save_interval / integration_step",
		)
		object.__setattr__(
			self,
			"chunk_size",
			positive_integer(self.chunk_size, "chunk_size"),
		)
		validate_block_prefix(self.block_prefix)
		if self.finite_difference_relative_step is not None:
			object.__setattr__(
				self,
				"finite_difference_relative_step",
				positive_finite(
					self.finite_difference_relative_step,
					"finite_difference_relative_step",
				),
			)
		for name in ("newton_absolute_tolerance", "newton_relative_tolerance"):
			object.__setattr__(
				self,
				name,
				positive_finite(getattr(self, name), name),
			)
		object.__setattr__(
			self,
			"newton_max_iterations",
			positive_integer(self.newton_max_iterations, "newton_max_iterations"),
		)

	@property
	def output_sample_count(self) -> int:
		"""Number of synchronized saved states, including both endpoints."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		) + 1


@dataclass(frozen=True, slots=True)
class ABBARuntimeSummary:
	"""Wall-clock simulation cost outside the symplecticity observer."""

	method_name: str
	seconds: float
	relative_to_fastest: float


@dataclass(frozen=True, slots=True)
class ABBATrajectoryDifferenceSeries:
	"""Periodic pointwise distance between two transported boundaries."""

	first_method: str
	second_method: str
	times: np.ndarray
	rms_particle_distance: np.ndarray
	max_particle_distance: np.ndarray

	@property
	def label(self) -> str:
		"""Compact pair label used by tables and plots."""
		return f"{self.first_method} vs {self.second_method}"


@dataclass(frozen=True, slots=True)
class ABBATrajectoryDifferenceSummary:
	"""Aggregate periodic trajectory distances for one method pair."""

	first_method: str
	second_method: str
	rms_distance: float
	max_distance: float
	final_rms_distance: float
	final_max_distance: float


@dataclass(frozen=True, slots=True)
class ABBAComparisonResult:
	"""Three ABBA studies plus runtime and physical-trajectory comparisons."""

	potential: Potential
	area: Area
	config: ABBAComparisonConfig
	studies: Mapping[str, ABBAComparisonStudy]
	runtimes: Mapping[str, float]

	def __post_init__(self) -> None:
		"""Require one aligned result and one positive runtime per ABBA method."""
		if not isinstance(self.potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		if not isinstance(self.area, Area):
			raise TypeError("`area` must be an Area instance.")
		if not isinstance(self.config, ABBAComparisonConfig):
			raise TypeError("`config` must be an ABBAComparisonConfig instance.")
		if tuple(self.studies) != ABBA_METHOD_NAMES:
			raise ValueError("`studies` must contain the three ABBA methods in order.")
		if tuple(self.runtimes) != ABBA_METHOD_NAMES:
			raise ValueError("`runtimes` must contain the three ABBA methods in order.")
		for method_name, seconds in self.runtimes.items():
			if not np.isfinite(float(seconds)) or float(seconds) <= 0:
				raise ValueError(
					f"The runtime for {method_name} must be positive and finite."
				)
		_validate_abba_output_grids((self._solution(name) for name in ABBA_METHOD_NAMES))

	@property
	def solutions(self) -> Mapping[str, Solution]:
		"""Physical solutions indexed by ABBA method name."""
		return MappingProxyType(
			{method_name: self._solution(method_name) for method_name in ABBA_METHOD_NAMES}
		)

	def _solution(self, method_name: str) -> Solution:
		"""Return the sole physical solution from one method study."""
		if method_name not in self.studies:
			raise KeyError(f"Unknown ABBA method: {method_name}")
		study = self.studies[method_name]
		if tuple(study.solutions) != (self.config.step_label,):
			raise ValueError("Each ABBA study must contain exactly the configured step.")
		return study.solutions[self.config.step_label]

	def runtime_summaries(self) -> tuple[ABBARuntimeSummary, ...]:
		"""Return measured wall times relative to the fastest complete run."""
		fastest = min(float(value) for value in self.runtimes.values())
		return tuple(
			ABBARuntimeSummary(
				method_name=method_name,
				seconds=float(self.runtimes[method_name]),
				relative_to_fastest=float(self.runtimes[method_name]) / fastest,
			)
			for method_name in ABBA_METHOD_NAMES
		)

	def trajectory_difference_series(
		self,
	) -> tuple[ABBATrajectoryDifferenceSeries, ...]:
		"""Return pairwise RMS and maximum particle distances on the periodic cell."""
		series: list[ABBATrajectoryDifferenceSeries] = []
		period = self.potential.grid.period
		for first_method, second_method in combinations(ABBA_METHOD_NAMES, 2):
			first = self._solution(first_method)
			second = self._solution(second_method)
			first_x, first_y = first.positions()
			second_x, second_y = second.positions()
			delta_x = minimum_image_displacement(first_x - second_x, period)
			delta_y = minimum_image_displacement(first_y - second_y, period)
			distances = np.hypot(delta_x, delta_y)
			series.append(
				ABBATrajectoryDifferenceSeries(
					first_method=first_method,
					second_method=second_method,
					times=np.asarray(first.t, dtype=float),
					rms_particle_distance=np.sqrt(np.mean(distances**2, axis=0)),
					max_particle_distance=np.max(distances, axis=0),
				)
			)
		return tuple(series)

	def trajectory_difference_summaries(
		self,
	) -> tuple[ABBATrajectoryDifferenceSummary, ...]:
		"""Aggregate pairwise periodic distances over particles and saved times."""
		return tuple(
			ABBATrajectoryDifferenceSummary(
				first_method=series.first_method,
				second_method=series.second_method,
				rms_distance=float(
					np.sqrt(np.mean(series.rms_particle_distance**2))
				),
				max_distance=float(np.max(series.max_particle_distance)),
				final_rms_distance=float(series.rms_particle_distance[-1]),
				final_max_distance=float(series.max_particle_distance[-1]),
			)
			for series in self.trajectory_difference_series()
		)

	def print_summary(self) -> None:
		"""Print timing, geometric diagnostics, and trajectory differences."""
		print(
			f"{'method':>24} {'runtime [s]':>14} {'vs fastest':>12} "
			f"{'max |area error|':>20} {'max flow defect':>20}"
		)
		for runtime in self.runtime_summaries():
			method_summary = self.studies[runtime.method_name].summaries()[0]
			print(
				f"{runtime.method_name:>24} {runtime.seconds:14.6f} "
				f"{runtime.relative_to_fastest:12.3f} "
				f"{method_summary.max_area_error:20.8e} "
				f"{method_summary.max_flow_defect:20.8e}"
			)

		print("\nPeriodic physical-trajectory distances:")
		print(
			f"{'method pair':>49} {'global RMS':>16} {'maximum':>16} "
			f"{'final RMS':>16} {'final maximum':>16}"
		)
		for difference_summary in self.trajectory_difference_summaries():
			label = (
				f"{difference_summary.first_method} vs "
				f"{difference_summary.second_method}"
			)
			print(
				f"{label:>49} {difference_summary.rms_distance:16.8e} "
				f"{difference_summary.max_distance:16.8e} "
				f"{difference_summary.final_rms_distance:16.8e} "
				f"{difference_summary.final_max_distance:16.8e}"
			)
		print(
			"\nRuntime excludes time spent evaluating and persisting symplecticity "
			"diagnostics. Both implicit formulations use the same configured "
			"finite-difference observer in this comparison."
		)


def run_abba_comparison(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: ABBAComparisonConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> ABBAComparisonResult:
	"""Run each ABBA method once and measure complete diagnostic wall time."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(area, Area):
		raise TypeError("`area` must be an Area instance.")
	if not isinstance(config, ABBAComparisonConfig):
		raise TypeError("`config` must be an ABBAComparisonConfig instance.")

	step = (AreaStep(label=config.step_label, value=config.integration_step),)
	midpoint_config = ABBA2MidpointSymplecticityConfig(
		steps=step,
		t_span=config.t_span,
		save_interval=config.save_interval,
		rho=config.rho,
		chunk_size=config.chunk_size,
		progress=config.progress,
		block_prefix=f"{config.block_prefix}_midpoint",
		finite_difference_relative_step=config.finite_difference_relative_step,
	)
	implicit_config = ImplicitABBASymplecticityConfig(
		steps=step,
		t_span=config.t_span,
		save_interval=config.save_interval,
		rho=config.rho,
		chunk_size=config.chunk_size,
		progress=config.progress,
		block_prefix=f"{config.block_prefix}_implicit",
		finite_difference_relative_step=config.finite_difference_relative_step,
		newton_absolute_tolerance=config.newton_absolute_tolerance,
		newton_relative_tolerance=config.newton_relative_tolerance,
		newton_max_iterations=config.newton_max_iterations,
	)
	common_metadata = {
		**dict(metadata or {}),
		"study_kind": "single_step_three_abba_comparison",
		"timing_scope": "simulation_excluding_symplecticity_observer",
	}

	midpoint_result = run_abba2_midpoint_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=midpoint_config,
		project_root=project_root,
		metadata=common_metadata,
	)

	reduced_result = run_abba2_reduced_multiplier_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=implicit_config,
		project_root=project_root,
		metadata=common_metadata,
	)

	simultaneous_result = run_abba2_simultaneous_state_multiplier_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=implicit_config,
		project_root=project_root,
		metadata=common_metadata,
	)

	studies: dict[str, ABBAComparisonStudy] = {
		ABBA_METHOD_NAMES[0]: midpoint_result,
		ABBA_METHOD_NAMES[1]: reduced_result,
		ABBA_METHOD_NAMES[2]: simultaneous_result,
	}
	runtimes = {
		method_name: studies[method_name].simulation_runtime_seconds[
			config.step_label
		]
		for method_name in ABBA_METHOD_NAMES
	}
	return ABBAComparisonResult(
		potential=potential,
		area=area,
		config=config,
		studies=MappingProxyType(studies),
		runtimes=MappingProxyType(runtimes),
	)


def _validate_abba_output_grids(solutions: Iterable[Solution]) -> None:
	"""Keep the historical epsilon-scaled tolerance for common ABBA output grids."""
	# Validate the common output grid eagerly so all later comparisons are direct.
	reference_times: np.ndarray | None = None
	for solution in solutions:
		if reference_times is None:
			reference_times = np.asarray(solution.t, dtype=float)
		else:
			candidate_times = np.asarray(solution.t, dtype=float)
			time_scale = max(1.0, float(np.max(np.abs(reference_times))))
			tolerance = float(32 * np.finfo(float).eps * time_scale)
			if candidate_times.shape != reference_times.shape or not np.allclose(
				candidate_times,
				reference_times,
				rtol=0.0,
				atol=tolerance,
			):
				raise ValueError(
					"All ABBA solutions must share the same saved-time grid."
				)


__all__ = [
	"ABBAComparisonConfig",
	"ABBAComparisonResult",
	"ABBARuntimeSummary",
	"ABBATrajectoryDifferenceSeries",
	"ABBATrajectoryDifferenceSummary",
	"ABBA_METHOD_NAMES",
	"run_abba_comparison",
]
