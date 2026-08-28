"""Shared reference validation and trajectory-accuracy primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Protocol

import numpy as np

from diagnostics import StoredReferenceTrajectory
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential

from ._trajectory_distances import (
	DistanceConvention,
	normalized_distance_convention,
	particle_distances,
	periodic_particle_distances,
)
from ._validation import integer_ratio, positive_finite
from .reference_trajectory import potential_fingerprint


class ReferenceAccuracyConfig(Protocol):
	"""Configuration fields required to validate a stored GC reference."""

	@property
	def rho(self) -> float:
		"""Return the normalized gyroradius."""
		...

	@property
	def t_span(self) -> tuple[float, float]:
		"""Return the integration interval."""
		...

	@property
	def save_interval(self) -> float | None:
		"""Return the common saved-time interval after configuration validation."""
		...


@dataclass(frozen=True, slots=True)
class TrajectoryAccuracySeries:
	"""Per-particle planar distances and time-dependent reductions."""

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


def _json_canonical(value: object) -> str:
	"""Normalize tuples and NumPy scalar values before metadata comparison."""
	def default(candidate: object) -> object:
		if isinstance(candidate, np.generic):
			return candidate.item()
		if isinstance(candidate, np.ndarray):
			return candidate.tolist()
		raise TypeError(f"Cannot canonicalize {type(candidate).__name__}.")

	return json.dumps(value, sort_keys=True, separators=(",", ":"), default=default)


def accuracy_series(
	method_name: str,
	states: np.ndarray,
	reference_states: np.ndarray,
	*,
	period: float | None,
	distance_convention: DistanceConvention = "periodic",
) -> TrajectoryAccuracySeries:
	"""Build per-time reductions without averaging coordinate components first."""
	distances = particle_distances(
		states,
		reference_states,
		distance_convention=distance_convention,
		period=period,
	)
	return TrajectoryAccuracySeries(
		method_name=method_name,
		distances=distances,
		rms_distance=np.sqrt(np.mean(distances**2, axis=0)),
		mean_distance=np.mean(distances, axis=0),
		maximum_distance=np.max(distances, axis=0),
	)


def reference_distance_convention(
	reference: StoredReferenceTrajectory,
) -> DistanceConvention:
	"""Read the reference metric, defaulting schema-v1 artifacts to periodic."""
	config = reference.metadata.get("config")
	if not isinstance(config, Mapping):
		raise ValueError("Reference numerical configuration is missing.")
	value = config.get("distance_convention", "periodic")
	if not isinstance(value, str):
		raise ValueError("Reference distance convention must be textual.")
	return normalized_distance_convention(value)


def validate_reference_identity(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	reference: StoredReferenceTrajectory,
	config: ReferenceAccuracyConfig,
	*,
	potential_metadata: Mapping[str, object],
	initial_condition_metadata: Mapping[str, object],
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
	reference_distance_convention(reference)
	if float(reference_config["rho"]) != config.rho:
		raise ValueError("The comparison rho differs from the reference.")
	if tuple(float(value) for value in reference_config["t_span"]) != config.t_span:
		raise ValueError("The comparison time span differs from the reference.")
	reference_interval = float(reference_config["save_interval"])
	comparison_interval = config.save_interval
	if comparison_interval is None:
		raise ValueError("The comparison saved interval must be configured.")
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
	integer_ratio(
		comparison_interval,
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
		raise ValueError("The comparison potential grid differs from the reference.")
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


def reference_indices_for_times(
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


def validated_refinement_steps(
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


__all__: list[str] = []
