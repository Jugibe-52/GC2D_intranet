"""Energy and independent-particle recurrence records usable without study runners."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from contracts._study_validation import (
	positive_finite, nonnegative_finite, positive_integer, integer_ratio, finite_time_span,
)


GC_ENERGY_BOUND_METHODS = ("BM4Implicit", "RK4")


@dataclass(frozen=True, slots=True)
class ParallelBM4RecurrenceConfig:
	"""Physical, temporal, nonlinear, and process controls for the campaign."""

	particle_count: int = 40
	t_span: tuple[float, float] = (0.0, 100.0)
	steps_per_cycle: int = 40
	saved_samples_per_cycle: int = 1
	rho: float = 0.3
	coupling_frequency: float = 0.0
	absolute_tolerance: float = 1e-12
	relative_tolerance: float = 1e-11
	max_iterations: int = 40
	jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	worker_count: int = 4
	progress: bool = True

	def __post_init__(self) -> None:
		"""Validate every value that affects reproducibility or scheduling."""
		object.__setattr__(
			self, "t_span",
			finite_time_span(self.t_span, message="`t_span` must contain two finite increasing times."),
		)
		for name in (
			"particle_count",
			"steps_per_cycle",
			"saved_samples_per_cycle",
			"max_iterations",
			"worker_count",
		):
			object.__setattr__(self, name, positive_integer(getattr(self, name), name))
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		object.__setattr__(
			self,
			"coupling_frequency",
			nonnegative_finite(self.coupling_frequency, "coupling_frequency"),
		)
		for name in (
			"absolute_tolerance",
			"relative_tolerance",
			"jacobian_relative_step",
		):
			object.__setattr__(self, name, positive_finite(getattr(self, name), name))
		integer_ratio(self.duration, 1.0, "duration / cycle_duration")
		object.__setattr__(self, "progress", bool(self.progress))

	@property
	def duration(self) -> float:
		"""Return the normalized integration duration."""
		return self.t_span[1] - self.t_span[0]

	@property
	def cycle_count(self) -> int:
		"""Return the number of complete unit-duration cycles."""
		return integer_ratio(self.duration, 1.0, "duration / cycle_duration")

	@property
	def integration_step(self) -> float:
		"""Return the complete BM4 step."""
		return 1.0 / self.steps_per_cycle

	@property
	def step_count(self) -> int:
		"""Return the complete BM4 steps in each independent trajectory."""
		return self.cycle_count * self.steps_per_cycle

	@property
	def output_sample_count(self) -> int:
		"""Return saved states including both endpoints."""
		return self.cycle_count * self.saved_samples_per_cycle + 1



def _readonly(values: np.ndarray, *, dtype: Any = float) -> np.ndarray:
	"""Own and freeze one validated result array."""
	result = np.array(values, dtype=dtype, copy=True)
	if not np.all(np.isfinite(result)):
		raise ValueError("Parallel BM4 output arrays must be finite.")
	result.setflags(write=False)
	return result



@dataclass(frozen=True, slots=True)
class ParallelBM4RecurrenceResult:
	"""Aligned independent BM4 trajectories and worker diagnostics."""

	config: ParallelBM4RecurrenceConfig
	times: np.ndarray
	initial_positions: np.ndarray
	positions: np.ndarray
	runtime_seconds: np.ndarray
	total_newton_iterations: np.ndarray
	mean_newton_iterations: np.ndarray
	maximum_newton_iterations: np.ndarray
	maximum_residual_to_tolerance: np.ndarray
	wall_runtime_seconds: float

	def __post_init__(self) -> None:
		"""Require one aligned planar history and work record per particle."""
		if not isinstance(self.config, ParallelBM4RecurrenceConfig):
			raise TypeError("`config` must be ParallelBM4RecurrenceConfig.")
		count = self.config.particle_count
		samples = self.config.output_sample_count
		times = _readonly(self.times)
		initial_positions = _readonly(self.initial_positions)
		positions = _readonly(self.positions)
		runtimes = _readonly(self.runtime_seconds)
		total_iterations = _readonly(self.total_newton_iterations, dtype=int)
		mean_iterations = _readonly(self.mean_newton_iterations)
		maximum_iterations = _readonly(self.maximum_newton_iterations, dtype=int)
		residual_ratios = _readonly(self.maximum_residual_to_tolerance)
		_validate_parallel_trajectories(times, initial_positions, positions, count=count, samples=samples)
		_validate_parallel_work(
			runtimes, total_iterations, mean_iterations, maximum_iterations, residual_ratios, count=count,
		)
		wall_runtime = positive_finite(
			self.wall_runtime_seconds,
			"wall_runtime_seconds",
		)
		object.__setattr__(self, "times", times)
		object.__setattr__(self, "initial_positions", initial_positions)
		object.__setattr__(self, "positions", positions)
		object.__setattr__(self, "runtime_seconds", runtimes)
		object.__setattr__(self, "total_newton_iterations", total_iterations)
		object.__setattr__(self, "mean_newton_iterations", mean_iterations)
		object.__setattr__(self, "maximum_newton_iterations", maximum_iterations)
		object.__setattr__(self, "maximum_residual_to_tolerance", residual_ratios)
		object.__setattr__(self, "wall_runtime_seconds", wall_runtime)

	def recurrence_distances(self, *, period: float) -> np.ndarray:
		"""Return minimum-image distance to each trajectory's initial point."""
		cell_period = positive_finite(period, "period")
		displacement = self.positions - self.initial_positions[:, :, None]
		periodic = (
			(displacement + 0.5 * cell_period) % cell_period
			- 0.5 * cell_period
		)
		distances = np.asarray(np.linalg.norm(periodic, axis=1), dtype=float)
		distances[:, 0] = 0.0
		distances.setflags(write=False)
		return distances



def _validate_parallel_trajectories(
	times: np.ndarray, initial_positions: np.ndarray, positions: np.ndarray,
	*, count: int, samples: int,
) -> None:
	"""Require the packed planar histories and exact initial positions for every particle."""
	if times.shape != (samples,) or np.any(np.diff(times) <= 0.0):
		raise ValueError("Saved times must be aligned, finite, and increasing.")
	if initial_positions.shape != (count, 2):
		raise ValueError("Initial positions must have shape (particle_count, 2).")
	if positions.shape != (count, 2, samples):
		raise ValueError(
			"Positions must have shape (particle_count, 2, saved_times)."
		)
	if not np.array_equal(positions[:, :, 0], initial_positions):
		raise ValueError("Every returned trajectory must preserve its initial state.")



def _validate_parallel_work(
	runtimes: np.ndarray, total_iterations: np.ndarray, mean_iterations: np.ndarray,
	maximum_iterations: np.ndarray, residual_ratios: np.ndarray, *, count: int,
) -> None:
	"""Require one aligned runtime and nonlinear-work summary per trajectory."""
	for values in (
		runtimes,
		total_iterations,
		mean_iterations,
		maximum_iterations,
		residual_ratios,
	):
		if values.shape != (count,):
			raise ValueError("Every particle must have one work summary.")
	if np.any(runtimes <= 0.0) or np.any(total_iterations < 0):
		raise ValueError("Runtime and iteration summaries must be non-negative.")



@dataclass
class GCEnergyBoundResult:
	"""Every accepted node and independently auditable summary records."""

	arrays: dict[str, np.ndarray]
	metadata: dict[str, Any]
	summary: list[dict[str, Any]]
	envelopes: list[dict[str, Any]]
	blocks: list[dict[str, Any]]
	orders: list[dict[str, Any]]


__all__ = ["ParallelBM4RecurrenceConfig", "ParallelBM4RecurrenceResult", "GCEnergyBoundResult", "GC_ENERGY_BOUND_METHODS"]
