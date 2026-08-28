"""Numerical accuracy of ten fixed-step variants against a stored reference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from diagnostics import StoredReferenceTrajectory
from initial_conditions import GCInitialConfiguration
from potential import Potential

from ._trajectory_accuracy import (
	TrajectoryAccuracySeries,
	accuracy_series as _accuracy_series,
	periodic_particle_distances,
	reference_distance_convention as _reference_distance_convention,
	reference_indices_for_times as _reference_indices_for_times,
	validate_reference_identity as _validate_reference_identity,
	validated_refinement_steps as _validated_refinement_steps,
)
from .ten_method_trajectory_comparison import (
	TEN_METHOD_LABELS,
	TEN_METHOD_VARIANTS,
	TenMethodTrajectoryComparisonConfig,
	TenMethodTrajectoryComparisonResult,
	run_ten_method_trajectory_comparison,
)


@dataclass(frozen=True, slots=True)
class TenMethodAccuracySummary:
	"""Scalar trajectory error and cost metrics for one numerical variant."""

	method_name: str
	family: str
	nonlinear_solver: str
	global_rms_distance: float
	time_integrated_rms_distance: float
	maximum_distance: float
	final_rms_distance: float
	final_maximum_distance: float
	normalized_global_rms_distance: float
	reference_floor_ratio: float
	within_ten_reference_floors: bool
	runtime_seconds: float


@dataclass(frozen=True, slots=True)
class TenMethodStepAccuracySummary:
	"""Comparable accuracy and runtime metrics at one complete step size."""

	integration_step: float
	method_name: str
	family: str
	nonlinear_solver: str
	global_rms_distance: float
	time_integrated_rms_distance: float
	maximum_distance: float
	final_rms_distance: float
	final_maximum_distance: float
	reference_floor_ratio: float
	runtime_seconds: float


@dataclass(frozen=True, slots=True)
class TenMethodAccuracyOrder:
	"""Observed error order between two adjacent step refinements."""

	method_name: str
	coarse_step: float
	fine_step: float
	time_integrated_rms_gain: float
	final_rms_gain: float
	time_integrated_rms_order: float
	final_rms_order: float
	resolved_above_reference_floor: bool


@dataclass(frozen=True, slots=True)
class TenMethodAccuracyResult:
	"""Reference errors for ten variants on one fixed-step main grid."""

	reference: StoredReferenceTrajectory
	comparison: TenMethodTrajectoryComparisonResult
	reference_sample_indices: np.ndarray
	series: Mapping[str, TrajectoryAccuracySeries]

	def __post_init__(self) -> None:
		"""Require complete label coverage and exact sample alignment."""
		if tuple(self.series) != TEN_METHOD_LABELS:
			raise ValueError("Accuracy results must contain all ten variants.")
		indices = np.array(self.reference_sample_indices, dtype=np.int64, copy=True)
		comparison_times = next(iter(self.comparison.solutions.values())).t
		if (
			indices.ndim != 1
			or indices.shape != comparison_times.shape
			or np.any(indices < 0)
			or np.any(indices >= self.reference.times.size)
			or not np.array_equal(self.reference.times[indices], comparison_times)
		):
			raise ValueError("Reference sample indices do not align with the main-step grid.")
		for label in TEN_METHOD_LABELS:
			candidate = self.series[label]
			if candidate.method_name != label:
				raise ValueError("Accuracy series labels are inconsistent.")
			if candidate.distances.shape[1] != comparison_times.size:
				raise ValueError("Accuracy series do not match the reference grid.")
		indices.setflags(write=False)
		object.__setattr__(self, "reference_sample_indices", indices)
		object.__setattr__(self, "series", MappingProxyType(dict(self.series)))

	@property
	def times(self) -> np.ndarray:
		"""Return the saved main-step times used by this comparison."""
		return next(iter(self.comparison.solutions.values())).t

	@property
	def reference_floor(self) -> float:
		"""Return the DOP853/Radau global RMS on this result's saved grid."""
		distances = self.reference.audit_distances[
			:, self.reference_sample_indices
		]
		floor = float(np.sqrt(np.mean(distances**2)))
		if not np.isfinite(floor) or floor < 0.0:
			raise ValueError("Reference audit floor is invalid.")
		return floor

	def summaries(self) -> tuple[TenMethodAccuracySummary, ...]:
		"""Return accuracy, resolution margin, and runtime in stable order."""
		rows: list[TenMethodAccuracySummary] = []
		period = float(self.comparison.potential.grid.period)
		times = self.times
		duration = float(times[-1] - times[0])
		floor = max(
			self.reference_floor,
			float(np.finfo(float).eps) * max(1.0, period),
		)
		for variant in TEN_METHOD_VARIANTS:
			series = self.series[variant.label]
			global_rms = float(np.sqrt(np.mean(series.distances**2)))
			time_rms = float(
				np.sqrt(np.trapz(series.rms_distance**2, times) / duration)
			)
			ratio = global_rms / floor
			rows.append(
				TenMethodAccuracySummary(
					method_name=variant.label,
					family=variant.family,
					nonlinear_solver=variant.nonlinear_solver or "explicit midpoint",
					global_rms_distance=global_rms,
					time_integrated_rms_distance=time_rms,
					maximum_distance=float(np.max(series.distances)),
					final_rms_distance=float(series.rms_distance[-1]),
					final_maximum_distance=float(series.maximum_distance[-1]),
					normalized_global_rms_distance=global_rms / period,
					reference_floor_ratio=ratio,
					within_ten_reference_floors=ratio <= 10.0,
					runtime_seconds=float(self.comparison.runtimes[variant.label]),
				)
			)
		return tuple(rows)


@dataclass(frozen=True, slots=True)
class TenMethodAccuracyRefinementResult:
	"""Ten-method accuracy results on nested steps and one common output grid."""

	reference: StoredReferenceTrajectory
	integration_steps: tuple[float, ...]
	results: Mapping[float, TenMethodAccuracyResult]

	def __post_init__(self) -> None:
		"""Require coarse-to-fine steps, stable keys, and identical saved times."""
		steps = _validated_refinement_steps(self.integration_steps)
		if tuple(self.results) != steps:
			raise ValueError("Refinement results must follow the configured step order.")
		common_times: np.ndarray | None = None
		for step, result in self.results.items():
			if result.reference is not self.reference:
				raise ValueError("Every refinement must use the same reference artifact.")
			if result.comparison.config.integration_step != step:
				raise ValueError("A refinement result used the wrong integration step.")
			if common_times is None:
				common_times = result.times
			elif not np.array_equal(result.times, common_times):
				raise ValueError("Every refinement must use one common saved-time grid.")
		object.__setattr__(self, "integration_steps", steps)
		object.__setattr__(self, "results", MappingProxyType(dict(self.results)))

	@property
	def finest_result(self) -> TenMethodAccuracyResult:
		"""Return the result produced with the smallest complete step."""
		return self.results[self.integration_steps[-1]]

	@property
	def reference_floor(self) -> float:
		"""Return the common-grid DOP853/Radau time-integrated RMS floor."""
		indices = self.finest_result.reference_sample_indices
		times = self.finest_result.times
		distances = self.reference.audit_distances[:, indices]
		particle_rms_squared = np.mean(distances**2, axis=0)
		return float(
			np.sqrt(
				np.trapz(particle_rms_squared, times)
				/ float(times[-1] - times[0])
			)
		)

	@property
	def final_reference_floor(self) -> float:
		"""Return the particle-RMS DOP853/Radau discrepancy at final time."""
		indices = self.finest_result.reference_sample_indices
		final_distances = self.reference.audit_distances[:, indices[-1]]
		return float(np.sqrt(np.mean(final_distances**2)))

	def summaries(self) -> tuple[TenMethodStepAccuracySummary, ...]:
		"""Flatten step and method metrics for notebook tables and plots."""
		rows: list[TenMethodStepAccuracySummary] = []
		floor = max(self.reference_floor, float(np.finfo(float).eps))
		for step in self.integration_steps:
			for summary in self.results[step].summaries():
				rows.append(
					TenMethodStepAccuracySummary(
						integration_step=step,
						method_name=summary.method_name,
						family=summary.family,
						nonlinear_solver=summary.nonlinear_solver,
						global_rms_distance=summary.global_rms_distance,
						time_integrated_rms_distance=(
							summary.time_integrated_rms_distance
						),
						maximum_distance=summary.maximum_distance,
						final_rms_distance=summary.final_rms_distance,
						final_maximum_distance=summary.final_maximum_distance,
						reference_floor_ratio=(
							summary.time_integrated_rms_distance
							/ floor
						),
						runtime_seconds=summary.runtime_seconds,
					)
				)
		return tuple(rows)

	def convergence_orders(self) -> tuple[TenMethodAccuracyOrder, ...]:
		"""Estimate adjacent observed orders from comparable error norms."""
		summaries = {
			(step, row.method_name): row
			for step in self.integration_steps
			for row in self.results[step].summaries()
		}
		time_floor = max(self.reference_floor, float(np.finfo(float).eps))
		final_floor = max(self.final_reference_floor, float(np.finfo(float).eps))
		rows: list[TenMethodAccuracyOrder] = []
		for coarse_step, fine_step in zip(
			self.integration_steps,
			self.integration_steps[1:],
		):
			step_ratio = coarse_step / fine_step
			for method_name in TEN_METHOD_LABELS:
				coarse = summaries[(coarse_step, method_name)]
				fine = summaries[(fine_step, method_name)]
				time_resolved = (
					fine.time_integrated_rms_distance > 10.0 * time_floor
					and coarse.time_integrated_rms_distance > 0.0
				)
				final_resolved = (
					fine.final_rms_distance > 10.0 * final_floor
					and coarse.final_rms_distance > 0.0
				)
				if fine.time_integrated_rms_distance > 0.0:
					time_gain = (
						coarse.time_integrated_rms_distance
						/ fine.time_integrated_rms_distance
					)
				else:
					time_gain = float("nan")
				if fine.final_rms_distance > 0.0:
					final_gain = coarse.final_rms_distance / fine.final_rms_distance
				else:
					final_gain = float("nan")
				if time_resolved:
					time_order = np.log(time_gain) / np.log(step_ratio)
				else:
					time_order = float("nan")
				if final_resolved:
					final_order = np.log(final_gain) / np.log(step_ratio)
				else:
					final_order = float("nan")
				rows.append(
					TenMethodAccuracyOrder(
						method_name=method_name,
						coarse_step=coarse_step,
						fine_step=fine_step,
						time_integrated_rms_gain=float(time_gain),
						final_rms_gain=float(final_gain),
						time_integrated_rms_order=float(time_order),
						final_rms_order=float(final_order),
						resolved_above_reference_floor=(
							time_resolved and final_resolved
						),
					)
				)
		return tuple(rows)


def run_ten_method_accuracy_study(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	reference: StoredReferenceTrajectory,
	*,
	config: TenMethodTrajectoryComparisonConfig,
	potential_metadata: Mapping[str, Any],
	initial_condition_metadata: Mapping[str, Any],
) -> TenMethodAccuracyResult:
	"""Run all ten variants and measure their periodic distance to the reference."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(reference, StoredReferenceTrajectory):
		raise TypeError("`reference` must be a StoredReferenceTrajectory.")
	if not isinstance(config, TenMethodTrajectoryComparisonConfig):
		raise TypeError("`config` must be a TenMethodTrajectoryComparisonConfig.")
	_validate_reference_identity(
		potential,
		initial_configuration,
		reference,
		config,
		potential_metadata=potential_metadata,
		initial_condition_metadata=initial_condition_metadata,
	)
	comparison = run_ten_method_trajectory_comparison(
		potential,
		initial_configuration,
		config=config,
	)
	comparison_times = next(iter(comparison.solutions.values())).t
	reference_indices = _reference_indices_for_times(reference, comparison_times)
	reference_states = reference.states[:, reference_indices]
	period = float(potential.grid.period)
	distance_convention = _reference_distance_convention(reference)
	series = {
		label: _accuracy_series(
			label,
			comparison.solutions[label].states,
			reference_states,
			period=period,
			distance_convention=distance_convention,
		)
		for label in TEN_METHOD_LABELS
	}
	return TenMethodAccuracyResult(
		reference=reference,
		comparison=comparison,
		reference_sample_indices=reference_indices,
		series=series,
	)


def run_ten_method_accuracy_refinement_study(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	reference: StoredReferenceTrajectory,
	*,
	base_config: TenMethodTrajectoryComparisonConfig,
	integration_steps: Sequence[float],
	potential_metadata: Mapping[str, Any],
	initial_condition_metadata: Mapping[str, Any],
) -> TenMethodAccuracyRefinementResult:
	"""Run all ten variants on nested steps and one main-grid-aligned cadence."""
	if not isinstance(base_config, TenMethodTrajectoryComparisonConfig):
		raise TypeError("`base_config` must be a TenMethodTrajectoryComparisonConfig.")
	steps = _validated_refinement_steps(integration_steps)
	results: dict[float, TenMethodAccuracyResult] = {}
	for step in steps:
		config = replace(base_config, integration_step=step)
		results[step] = run_ten_method_accuracy_study(
			potential,
			initial_configuration,
			reference,
			config=config,
			potential_metadata=potential_metadata,
			initial_condition_metadata=initial_condition_metadata,
		)
	return TenMethodAccuracyRefinementResult(
		reference=reference,
		integration_steps=steps,
		results=results,
	)


__all__ = [
	"TenMethodAccuracyResult",
	"TenMethodAccuracyRefinementResult",
	"TenMethodAccuracyOrder",
	"TenMethodAccuracySummary",
	"TenMethodStepAccuracySummary",
	"TrajectoryAccuracySeries",
	"periodic_particle_distances",
	"run_ten_method_accuracy_refinement_study",
	"run_ten_method_accuracy_study",
]
