"""Numerical accuracy of ten fixed-step variants against a stored reference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from diagnostics import StoredReferenceTrajectory
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential

from ._validation import integer_ratio, positive_finite
from .reference_trajectory import potential_fingerprint
from .ten_method_trajectory_comparison import (
	TEN_METHOD_LABELS,
	TEN_METHOD_VARIANTS,
	TenMethodTrajectoryComparisonConfig,
	TenMethodTrajectoryComparisonResult,
	run_ten_method_trajectory_comparison,
)


@dataclass(frozen=True, slots=True)
class TrajectoryAccuracySeries:
	"""Per-particle periodic distances and time-dependent reductions."""

	method_name: str
	distances: np.ndarray
	rms_distance: np.ndarray
	mean_distance: np.ndarray
	maximum_distance: np.ndarray

	def __post_init__(self) -> None:
		"""Own immutable finite non-negative accuracy arrays."""
		distances = np.array(self.distances, dtype=float, copy=True)
		rms = np.array(self.rms_distance, dtype=float, copy=True)
		mean = np.array(self.mean_distance, dtype=float, copy=True)
		maximum = np.array(self.maximum_distance, dtype=float, copy=True)
		if distances.ndim != 2 or distances.size == 0:
			raise ValueError("Accuracy distances must have shape (particles, samples).")
		for value in (distances, rms, mean, maximum):
			if not np.all(np.isfinite(value)) or np.any(value < 0.0):
				raise ValueError("Accuracy distances must be finite and non-negative.")
		if any(value.shape != (distances.shape[1],) for value in (rms, mean, maximum)):
			raise ValueError("Reduced accuracy series must have one value per sample.")
		for value in (distances, rms, mean, maximum):
			value.setflags(write=False)
		object.__setattr__(self, "distances", distances)
		object.__setattr__(self, "rms_distance", rms)
		object.__setattr__(self, "mean_distance", mean)
		object.__setattr__(self, "maximum_distance", maximum)


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


def _json_canonical(value: object) -> str:
	"""Normalize tuples and NumPy scalar values before metadata comparison."""
	def default(candidate: object) -> object:
		if isinstance(candidate, np.generic):
			return candidate.item()
		if isinstance(candidate, np.ndarray):
			return candidate.tolist()
		raise TypeError(f"Cannot canonicalize {type(candidate).__name__}.")

	return json.dumps(value, sort_keys=True, separators=(",", ":"), default=default)


def periodic_particle_distances(
	states: np.ndarray,
	reference_states: np.ndarray,
	*,
	period: float,
) -> np.ndarray:
	"""Return minimum-image planar distances with shape (particles, samples)."""
	first = np.asarray(states, dtype=float)
	second = np.asarray(reference_states, dtype=float)
	if first.shape != second.shape or first.ndim != 2 or first.shape[0] % 2:
		raise ValueError("Compared GC histories must have one matching packed shape.")
	if not np.isfinite(period) or period <= 0.0:
		raise ValueError("`period` must be positive and finite.")
	particle_count = first.shape[0] // 2
	difference = (first - second + period / 2.0) % period - period / 2.0
	return np.asarray(
		np.hypot(difference[:particle_count], difference[particle_count:]),
		dtype=float,
	)


def _accuracy_series(
	method_name: str,
	states: np.ndarray,
	reference_states: np.ndarray,
	*,
	period: float,
) -> TrajectoryAccuracySeries:
	"""Build per-time reductions without averaging coordinate components first."""
	distances = periodic_particle_distances(
		states,
		reference_states,
		period=period,
	)
	return TrajectoryAccuracySeries(
		method_name=method_name,
		distances=distances,
		rms_distance=np.sqrt(np.mean(distances**2, axis=0)),
		mean_distance=np.mean(distances, axis=0),
		maximum_distance=np.max(distances, axis=0),
	)


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
		distances = self.reference.audit_periodic_distances[
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
		distances = self.reference.audit_periodic_distances[:, indices]
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
		final_distances = self.reference.audit_periodic_distances[:, indices[-1]]
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


def _validate_reference_identity(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	reference: StoredReferenceTrajectory,
	config: TenMethodTrajectoryComparisonConfig,
	*,
	potential_metadata: Mapping[str, Any],
	initial_condition_metadata: Mapping[str, Any],
) -> None:
	"""Reject mismatched ODEs, initial states, parameters, and time grids."""
	initial_state = initial_configuration.initial_state
	if initial_state is None or not np.array_equal(initial_state, reference.initial_state):
		raise ValueError("The comparison initial state differs from the reference.")
	metadata = reference.metadata
	if _json_canonical(metadata.get("potential")) != _json_canonical(
		dict(potential_metadata)
	):
		raise ValueError("Potential metadata differs from the reference artifact.")
	if _json_canonical(metadata.get("initial_conditions")) != _json_canonical(
		dict(initial_condition_metadata)
	):
		raise ValueError("Initial-condition metadata differs from the reference artifact.")
	reference_config = metadata.get("config")
	if not isinstance(reference_config, Mapping):
		raise ValueError("Reference numerical configuration is missing.")
	if float(reference_config["rho"]) != config.rho:
		raise ValueError("The comparison rho differs from the reference.")
	if tuple(float(value) for value in reference_config["t_span"]) != config.t_span:
		raise ValueError("The comparison time span differs from the reference.")
	reference_interval = float(reference_config["save_interval"])
	reference_sample_count = integer_ratio(
		config.t_span[1] - config.t_span[0],
		reference_interval,
		"reference duration / saved interval",
	) + 1
	if reference.times.size != reference_sample_count or not np.array_equal(
		reference.times,
		np.linspace(*config.t_span, reference_sample_count),
	):
		raise ValueError("The reference saved-time grid is inconsistent with its manifest.")
	assert config.save_interval is not None
	integer_ratio(
		config.save_interval,
		reference_interval,
		"comparison saved interval / reference saved interval",
	)
	grid_metadata = metadata.get("potential_grid")
	grid = potential.grid
	actual_grid_metadata = {
		"xmin": grid.xmin,
		"ymin": grid.ymin,
		"dx": grid.dx,
		"dy": grid.dy,
		"nx": grid.nx,
		"ny": grid.ny,
		"period": grid.period,
		"shape": grid.shape,
		"interpolation_order": potential.interpolation_order,
	}
	if _json_canonical(grid_metadata) != _json_canonical(actual_grid_metadata):
		raise ValueError("The comparison periodic grid differs from the reference.")
	if metadata.get("dynamics_fingerprint_algorithm") != (
		"gc2d-sampled-potential-v1-sha256"
	):
		raise ValueError("The reference dynamics fingerprint algorithm is unsupported.")
	stored_fingerprint = metadata.get("dynamics_fingerprint_sha256")
	actual_fingerprint = potential_fingerprint(
		GuidingCenterDynamics(potential, rho=config.rho).effective_potential
	)
	if stored_fingerprint != actual_fingerprint:
		raise ValueError("The comparison interpolated ODE differs from the reference.")


def _reference_indices_for_times(
	reference: StoredReferenceTrajectory,
	times: np.ndarray,
) -> np.ndarray:
	"""Locate fixed-step main-grid nodes exactly in the stored reference grid."""
	values = np.asarray(times, dtype=float)
	indices = np.searchsorted(reference.times, values)
	if (
		indices.shape != values.shape
		or np.any(indices >= reference.times.size)
		or not np.array_equal(reference.times[indices], values)
	):
		raise ValueError("Every comparison main-grid time must exist in the reference.")
	return np.asarray(indices, dtype=np.int64)


def _validated_refinement_steps(
	integration_steps: Sequence[float],
) -> tuple[float, ...]:
	"""Normalize a coarse-to-fine sequence whose main grids are nested."""
	steps = tuple(
		positive_finite(step, "integration_steps") for step in integration_steps
	)
	if len(steps) < 2 or any(
		coarse <= fine for coarse, fine in zip(steps, steps[1:])
	):
		raise ValueError("Integration steps must be strictly decreasing coarse to fine.")
	for coarse, fine in zip(steps, steps[1:]):
		integer_ratio(coarse, fine, "coarse step / fine step")
	return steps


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
	series = {
		label: _accuracy_series(
			label,
			comparison.solutions[label].states,
			reference_states,
			period=period,
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
