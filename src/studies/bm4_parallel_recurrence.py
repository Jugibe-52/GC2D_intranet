"""Parallel independent-particle BM4 recurrence campaign."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
import sys
from time import perf_counter
from typing import Any, TypeAlias

import numpy as np
from threadpoolctl import ThreadpoolController

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import GC2DH5Metadata, Grid, Potential
from methods.extended.bm4 import BM4Implicit
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from simulation.runner import simulate

from ._validation import integer_ratio, nonnegative_finite, positive_finite, positive_integer


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
		span = np.asarray(self.t_span, dtype=float)
		if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
			raise ValueError("`t_span` must contain two finite increasing times.")
		object.__setattr__(self, "t_span", (float(span[0]), float(span[1])))
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


@dataclass(frozen=True, slots=True)
class _H5PotentialSnapshot:
	"""Pickle-safe processed HDF5 potential for spawned workers."""

	grid: Grid
	mean: np.ndarray
	modes: np.ndarray
	frequencies: np.ndarray
	metadata: GC2DH5Metadata
	interpolation_order: int

	@classmethod
	def from_potential(cls, potential: Potential) -> _H5PotentialSnapshot:
		"""Capture processed fields without reopening the source HDF5 file."""
		if not isinstance(potential.metadata, GC2DH5Metadata):
			raise TypeError("The potential must contain GC2D HDF5 metadata.")
		return cls(
			grid=potential.grid,
			mean=potential.mean,
			modes=potential.modes,
			frequencies=potential.frequencies,
			metadata=potential.metadata,
			interpolation_order=potential.interpolation_order,
		)

	def restore(self) -> Potential:
		"""Build worker-local interpolation objects from the processed fields."""
		return Potential(
			self.grid,
			mean=self.mean,
			modes=self.modes,
			frequencies=self.frequencies,
			metadata=self.metadata,
			interpolation_order=self.interpolation_order,
		)


_PotentialPayload: TypeAlias = Potential | _H5PotentialSnapshot


@dataclass(frozen=True, slots=True)
class _WorkerContext:
	"""Objects constructed once and reused by one worker process."""

	dynamics: GuidingCenterDynamics
	request: SimulationRequest
	coordinates: np.ndarray
	config: ParallelBM4RecurrenceConfig


@dataclass(frozen=True, slots=True)
class _ParticlePayload:
	"""Compact trajectory and nonlinear-work record returned by one worker."""

	particle: int
	times: np.ndarray
	positions: np.ndarray
	runtime_seconds: float
	total_newton_iterations: int
	mean_newton_iterations: float
	maximum_newton_iterations: int
	maximum_residual_to_tolerance: float


_WORKER_CONTEXT: _WorkerContext | None = None
_WORKER_THREADPOOL_LIMITER: Any = None


def _potential_payload(potential: Potential) -> _PotentialPayload:
	"""Return a spawn-safe potential representation."""
	if isinstance(potential.metadata, GC2DH5Metadata):
		return _H5PotentialSnapshot.from_potential(potential)
	return potential


def _initialize_worker(
	potential_payload: _PotentialPayload,
	coordinates: np.ndarray,
	config: ParallelBM4RecurrenceConfig,
) -> None:
	"""Initialize one single-threaded process-local BM4 environment."""
	global _WORKER_CONTEXT, _WORKER_THREADPOOL_LIMITER
	_WORKER_THREADPOOL_LIMITER = ThreadpoolController().limit(limits=1)
	potential = (
		potential_payload.restore()
		if isinstance(potential_payload, _H5PotentialSnapshot)
		else potential_payload
	)
	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.integration_step,
		sample_count=config.output_sample_count,
	)
	_WORKER_CONTEXT = _WorkerContext(
		dynamics=dynamics,
		request=request,
		coordinates=np.asarray(coordinates, dtype=float),
		config=config,
	)


def _run_particle(particle: int) -> _ParticlePayload:
	"""Integrate one complete independent BM4 trajectory."""
	context = _WORKER_CONTEXT
	if context is None:
		raise RuntimeError("The parallel BM4 worker was not initialized.")
	x, y = context.coordinates[particle]
	initial_configuration = GCInitialConfiguration.from_components(
		x=np.asarray([x], dtype=float),
		y=np.asarray([y], dtype=float),
	)
	problem = InitialValueProblem(context.dynamics, initial_configuration)
	method = BM4Implicit(
		coupling_frequency=context.config.coupling_frequency,
		newton_absolute_tolerance=context.config.absolute_tolerance,
		newton_relative_tolerance=context.config.relative_tolerance,
		newton_max_iterations=context.config.max_iterations,
		newton_jacobian_method="analytic",
		newton_jacobian_relative_step=context.config.jacobian_relative_step,
		progress=False,
	)
	started = perf_counter()
	solution = simulate(problem, method, context.request)
	runtime = perf_counter() - started
	position_x, position_y = solution.positions()
	positions = np.stack((position_x[0], position_y[0]), axis=0)
	diagnostics = solution.diagnostics
	iterations = np.asarray(diagnostics["nonlinear_iterations"], dtype=int)
	residuals = np.asarray(diagnostics["nonlinear_residual_norms"], dtype=float)
	tolerances = np.asarray(diagnostics["nonlinear_tolerances"], dtype=float)
	if iterations.shape != (context.config.step_count,):
		raise ValueError("BM4 iteration history does not cover every complete step.")
	return _ParticlePayload(
		particle=particle,
		times=solution.t,
		positions=positions,
		runtime_seconds=runtime,
		total_newton_iterations=int(np.sum(iterations)),
		mean_newton_iterations=float(np.mean(iterations)),
		maximum_newton_iterations=int(np.max(iterations)),
		maximum_residual_to_tolerance=float(np.max(residuals / tolerances)),
	)


def _progress_message(
	*,
	completed: int,
	total: int,
	worker_count: int,
	started: float,
) -> None:
	"""Print one parent-owned progress heartbeat with a simple ETA."""
	elapsed = perf_counter() - started
	eta = "ETA after first completion"
	if completed:
		eta = f"ETA {elapsed * (total - completed) / completed:.1f} s"
	print(
		f"Parallel BM4 recurrence: {completed}/{total} trajectories; "
		f"workers {worker_count}; elapsed {elapsed:.1f} s; {eta}.",
		file=sys.stderr,
		flush=True,
	)


def run_parallel_bm4_recurrence(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	config: ParallelBM4RecurrenceConfig,
) -> ParallelBM4RecurrenceResult:
	"""Run one independent single-particle BM4 integration per process task."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(config, ParallelBM4RecurrenceConfig):
		raise TypeError("`config` must be ParallelBM4RecurrenceConfig.")
	initial_state = initial_configuration.initial_state
	if initial_state is None:
		raise ValueError("The initial configuration must contain an initial state.")
	initial_x, initial_y = initial_configuration.positions(initial_state)
	if initial_x.size != config.particle_count:
		raise ValueError(
			"The initial configuration particle count must match the campaign."
		)
	coordinates = np.column_stack((initial_x, initial_y))
	worker_count = min(config.worker_count, config.particle_count)
	started = perf_counter()
	payloads: dict[int, _ParticlePayload] = {}
	if config.progress:
		_progress_message(
			completed=0,
			total=config.particle_count,
			worker_count=worker_count,
			started=started,
		)
	with ProcessPoolExecutor(
		max_workers=worker_count,
		mp_context=get_context("spawn"),
		initializer=_initialize_worker,
		initargs=(_potential_payload(potential), coordinates, config),
	) as executor:
		future_particles: dict[Future[_ParticlePayload], int] = {
			executor.submit(_run_particle, particle): particle
			for particle in range(config.particle_count)
		}
		pending = set(future_particles)
		try:
			while pending:
				completed_futures, pending = wait(
					pending,
					timeout=30.0,
					return_when=FIRST_COMPLETED,
				)
				if not completed_futures:
					if config.progress:
						_progress_message(
							completed=len(payloads),
							total=config.particle_count,
							worker_count=worker_count,
							started=started,
						)
					continue
				for future in completed_futures:
					particle = future_particles[future]
					try:
						payload = future.result()
					except Exception as exc:
						raise RuntimeError(
							f"Parallel BM4 failed for trajectory {particle + 1}."
						) from exc
					if payload.particle != particle:
						raise RuntimeError("A worker returned the wrong trajectory.")
					payloads[particle] = payload
					if config.progress:
						_progress_message(
							completed=len(payloads),
							total=config.particle_count,
							worker_count=worker_count,
							started=started,
						)
		except BaseException:
			for future in future_particles:
				future.cancel()
			raise
	ordered = tuple(payloads[particle] for particle in range(config.particle_count))
	times = ordered[0].times
	if any(not np.array_equal(payload.times, times) for payload in ordered[1:]):
		raise ValueError("Parallel trajectories returned different saved-time grids.")
	return ParallelBM4RecurrenceResult(
		config=config,
		times=times,
		initial_positions=coordinates,
		positions=np.stack(tuple(payload.positions for payload in ordered), axis=0),
		runtime_seconds=np.asarray(
			tuple(payload.runtime_seconds for payload in ordered),
			dtype=float,
		),
		total_newton_iterations=np.asarray(
			tuple(payload.total_newton_iterations for payload in ordered),
			dtype=int,
		),
		mean_newton_iterations=np.asarray(
			tuple(payload.mean_newton_iterations for payload in ordered),
			dtype=float,
		),
		maximum_newton_iterations=np.asarray(
			tuple(payload.maximum_newton_iterations for payload in ordered),
			dtype=int,
		),
		maximum_residual_to_tolerance=np.asarray(
			tuple(payload.maximum_residual_to_tolerance for payload in ordered),
			dtype=float,
		),
		wall_runtime_seconds=perf_counter() - started,
	)


__all__ = [
	"ParallelBM4RecurrenceConfig",
	"ParallelBM4RecurrenceResult",
	"run_parallel_bm4_recurrence",
]
