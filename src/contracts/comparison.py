"""Immutable numerical comparison records, independent of study execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np
from solution import Solution


@dataclass(frozen=True, slots=True)
class AdaptiveReference:
	"""DOP853 trajectory with an independent Radau resolution audit."""

	times: np.ndarray
	states: np.ndarray
	audit_states: np.ndarray
	audit_distances: np.ndarray
	dop853_runtime_seconds: float
	radau_runtime_seconds: float
	dop853_function_evaluations: int
	radau_function_evaluations: int

	def __post_init__(self) -> None:
		"""Own immutable, aligned reference and audit arrays."""
		times = np.array(self.times, dtype=float, copy=True)
		states = np.array(self.states, dtype=float, copy=True)
		audit_states = np.array(self.audit_states, dtype=float, copy=True)
		distances = np.array(self.audit_distances, dtype=float, copy=True)
		_freeze_reference_arrays(times, states, audit_states, distances)
		object.__setattr__(self, "times", times)
		object.__setattr__(self, "states", states)
		object.__setattr__(self, "audit_states", audit_states)
		object.__setattr__(self, "audit_distances", distances)

	@property
	def time_integrated_rms_floor(self) -> float:
		"""Return the time-integrated particle-RMS DOP853/Radau discrepancy."""
		particle_rms_squared = np.mean(self.audit_distances**2, axis=0)
		return float(
			np.sqrt(
				np.trapz(particle_rms_squared, self.times)
				/ float(self.times[-1] - self.times[0])
			)
		)

	@property
	def final_rms_floor(self) -> float:
		"""Return the final-time particle-RMS DOP853/Radau discrepancy."""
		return float(np.sqrt(np.mean(self.audit_distances[:, -1] ** 2)))



def _freeze_reference_arrays(
	times: np.ndarray, states: np.ndarray, audit_states: np.ndarray, distances: np.ndarray,
) -> None:
	"""Validate aligned reference and audit arrays before making the owned copies immutable."""
	if (
		times.ndim != 1
		or times.size < 2
		or states.ndim != 2
		or states.shape != audit_states.shape
		or states.shape[1] != times.size
		or distances.ndim != 2
		or distances.shape[1] != times.size
		or not all(
			np.all(np.isfinite(value))
			for value in (times, states, audit_states, distances)
		)
		or np.any(distances < 0.0)
	):
		raise ValueError("Adaptive reference arrays are invalid or misaligned.")
	for value in (times, states, audit_states, distances):
		value.setflags(write=False)



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



@dataclass(frozen=True, slots=True)
class EnergyAccuracySeries:
	"""Signed physical-Hamiltonian errors against DOP853 for every particle."""

	method_name: str
	errors: np.ndarray

	def __post_init__(self) -> None:
		"""Own one immutable finite ``(particles, samples)`` error history."""
		if not isinstance(self.method_name, str) or not self.method_name:
			raise ValueError("`method_name` must be a non-empty string.")
		errors = np.array(self.errors, dtype=float, copy=True)
		if errors.ndim != 2 or errors.size == 0 or not np.all(np.isfinite(errors)):
			raise ValueError("Energy errors must be one finite particle-time array.")
		errors.setflags(write=False)
		object.__setattr__(self, "errors", errors)

	@property
	def rms_error(self) -> np.ndarray:
		"""Return the particle-RMS physical-energy error at every saved time."""
		return np.asarray(np.sqrt(np.mean(self.errors**2, axis=0)), dtype=float)

	@property
	def maximum_absolute_error(self) -> np.ndarray:
		"""Return the worst absolute particle-energy error at every saved time."""
		return np.asarray(np.max(np.abs(self.errors), axis=0), dtype=float)

	@property
	def running_maximum_absolute_error(self) -> np.ndarray:
		"""Return the running worst particle-energy error through time."""
		return np.maximum.accumulate(self.maximum_absolute_error)



@dataclass(frozen=True, slots=True)
class ThreeMethodNewtonSummary:
	"""Accuracy, runtime, and nonlinear work for one complete method run."""

	method_name: str
	method_label: str
	nonlinear_solver: str
	trajectory_count: int
	step_count: int
	global_rms_distance: float
	time_integrated_rms_distance: float
	final_rms_distance: float
	maximum_distance: float
	reference_floor_ratio: float
	time_integrated_rms_energy_error: float
	relative_time_integrated_rms_energy_error: float
	final_rms_energy_error: float
	maximum_absolute_energy_error: float
	energy_reference_floor_ratio: float
	runtime_seconds: float
	runtime_first_quartile_seconds: float
	runtime_third_quartile_seconds: float
	runtime_minimum_seconds: float
	runtime_maximum_seconds: float
	minimum_newton_iterations: int
	mean_newton_iterations: float
	maximum_newton_iterations: int
	total_newton_iterations: int
	mean_residual_evaluations: float
	total_residual_evaluations: int
	maximum_residual_to_tolerance: float



@dataclass(frozen=True, slots=True)
class ExecutionLogEntry:
	"""One completed reference, warm-up, or timed execution."""

	phase: str
	method_name: str
	method_label: str
	repeat: int
	trajectory_count: int
	step_count: int
	runtime_seconds: float

	def __post_init__(self) -> None:
		"""Require one finite, self-describing completion record."""
		if self.phase not in {"reference", "warmup", "timing"}:
			raise ValueError("Unknown execution-log phase.")
		if not self.method_name or not self.method_label:
			raise ValueError("Execution-log method names must not be empty.")
		if self.repeat < 0 or self.trajectory_count < 1 or self.step_count < 0:
			raise ValueError("Execution-log counts must be non-negative and valid.")
		if not np.isfinite(self.runtime_seconds) or self.runtime_seconds <= 0.0:
			raise ValueError("Execution-log runtime must be positive and finite.")



@dataclass(frozen=True, slots=True)
class FiveMethodComparisonSummary:
	"""Accuracy, energy, and timing values shared by all five methods."""

	method_name: str
	method_label: str
	solver: str
	trajectory_count: int
	step_count: int
	global_rms_distance: float
	time_integrated_rms_distance: float
	final_rms_distance: float
	maximum_distance: float
	reference_floor_ratio: float
	time_integrated_rms_energy_error: float
	relative_time_integrated_rms_energy_error: float
	final_rms_energy_error: float
	maximum_absolute_energy_error: float
	energy_reference_floor_ratio: float
	runtime_seconds: float
	runtime_first_quartile_seconds: float
	runtime_third_quartile_seconds: float
	runtime_minimum_seconds: float
	runtime_maximum_seconds: float



@dataclass(frozen=True, slots=True)
class FiveMethodNonlinearWorkSummary:
	"""Newton work for one of the four implicit methods."""

	method_name: str
	method_label: str
	step_count: int
	nonlinear_solves_per_step: int
	minimum_newton_iterations: int
	mean_newton_iterations: float
	maximum_newton_iterations: int
	total_newton_iterations: int
	mean_residual_evaluations: float
	total_residual_evaluations: int
	maximum_residual_to_tolerance: float



class ComparisonReadView(Protocol):
	"""Read contract shared by live comparisons and their persisted CSV views."""

	@property
	def reference(self) -> AdaptiveReference: ...
	@property
	def solutions(self) -> Mapping[str, Solution]: ...
	@property
	def accuracy(self) -> Mapping[str, TrajectoryAccuracySeries]: ...
	@property
	def energy_accuracy(self) -> Mapping[str, EnergyAccuracySeries]: ...
	@property
	def reference_energy_errors(self) -> np.ndarray: ...
	@property
	def runtime_samples(self) -> Mapping[str, np.ndarray]: ...
	@property
	def wall_runtime_seconds(self) -> float: ...
	@property
	def execution_log(self) -> tuple[ExecutionLogEntry, ...]: ...
	def summaries(self) -> tuple[FiveMethodComparisonSummary, ...]: ...
	def nonlinear_work_summaries(self) -> tuple[FiveMethodNonlinearWorkSummary, ...]: ...


__all__ = [
	"AdaptiveReference", "TrajectoryAccuracySeries", "EnergyAccuracySeries",
	"ThreeMethodNewtonSummary", "ExecutionLogEntry", "FiveMethodComparisonSummary",
	"FiveMethodNonlinearWorkSummary", "ComparisonReadView",
]
