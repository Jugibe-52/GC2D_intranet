"""Numerical accuracy of ten fixed-step variants against a stored reference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from diagnostics import StoredReferenceTrajectory
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential

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
	reference: StoredReferenceTrajectory,
	*,
	period: float,
) -> TrajectoryAccuracySeries:
	"""Build per-time reductions without averaging coordinate components first."""
	distances = periodic_particle_distances(
		states,
		reference.states,
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
	"""Aligned reference errors for all ten numerical variants."""

	reference: StoredReferenceTrajectory
	comparison: TenMethodTrajectoryComparisonResult
	series: Mapping[str, TrajectoryAccuracySeries]

	def __post_init__(self) -> None:
		"""Require complete label coverage and exact sample alignment."""
		if tuple(self.series) != TEN_METHOD_LABELS:
			raise ValueError("Accuracy results must contain all ten variants.")
		if not np.array_equal(
			self.reference.times,
			next(iter(self.comparison.solutions.values())).t,
		):
			raise ValueError("The reference and compared solutions must share times.")
		for label in TEN_METHOD_LABELS:
			candidate = self.series[label]
			if candidate.method_name != label:
				raise ValueError("Accuracy series labels are inconsistent.")
			if candidate.distances.shape[1] != self.reference.times.size:
				raise ValueError("Accuracy series do not match the reference grid.")
		object.__setattr__(self, "series", MappingProxyType(dict(self.series)))

	@property
	def reference_floor(self) -> float:
		"""Global RMS discrepancy between DOP853 and the Radau audit."""
		value = self.reference.metadata.get("audit")
		if not isinstance(value, Mapping):
			raise ValueError("Reference audit metadata is missing.")
		floor = float(value["global_rms_distance"])
		if not np.isfinite(floor) or floor < 0.0:
			raise ValueError("Reference audit floor is invalid.")
		return floor

	def summaries(self) -> tuple[TenMethodAccuracySummary, ...]:
		"""Return accuracy, resolution margin, and runtime in stable order."""
		rows: list[TenMethodAccuracySummary] = []
		period = float(self.comparison.potential.grid.period)
		times = self.reference.times
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
	if float(reference_config["save_interval"]) != config.integration_step:
		raise ValueError("The comparison step must match the saved reference interval.")
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
	expected_count = config.output_sample_count
	if reference.times.size != expected_count:
		raise ValueError("The reference does not contain every comparison grid node.")


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
	if not np.array_equal(reference.times, next(iter(comparison.solutions.values())).t):
		raise ValueError("Fixed-step solutions do not share the reference time grid.")
	period = float(potential.grid.period)
	series = {
		label: _accuracy_series(
			label,
			comparison.solutions[label].states,
			reference,
			period=period,
		)
		for label in TEN_METHOD_LABELS
	}
	return TenMethodAccuracyResult(
		reference=reference,
		comparison=comparison,
		series=series,
	)


__all__ = [
	"TenMethodAccuracyResult",
	"TenMethodAccuracySummary",
	"TrajectoryAccuracySeries",
	"periodic_particle_distances",
	"run_ten_method_accuracy_study",
]
