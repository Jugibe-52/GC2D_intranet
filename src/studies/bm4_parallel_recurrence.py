"""Parallel independent-particle BM4 recurrence campaign."""

from __future__ import annotations

from studies._validation import require_uncoupled_bm4

from formulations.layout import GCStateLayout

from contracts.studies.bm4_parallel_recurrence import ParallelBM4RecurrenceConfig, ParallelBM4RecurrenceResult

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
import sys
from time import perf_counter
from typing import Any, TYPE_CHECKING, TypeAlias

import numpy as np

from threadpoolctl import ThreadpoolController

from dynamics import GuidingCenterDynamics

from potential import GC2DH5Metadata, Potential
from methods.extended.bm4 import BM4Implicit
from problem import InitialValueProblem
from request import SimulationRequest
from simulation.runner import simulate

from ._potential_snapshot import _H5PotentialSnapshot


if TYPE_CHECKING:
	from states.history import States


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
	require_uncoupled_bm4(config.coupling_frequency)
	global _WORKER_CONTEXT, _WORKER_THREADPOOL_LIMITER
	_WORKER_THREADPOOL_LIMITER = ThreadpoolController().limit(limits=1)
	potential: Potential = (
		potential_payload.restore()
		if isinstance(potential_payload, _H5PotentialSnapshot)
		else potential_payload
	)
	dynamics: GuidingCenterDynamics = GuidingCenterDynamics(potential, rho=config.rho)
	request: SimulationRequest = SimulationRequest.uniform(
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
	x: Any
	y: Any
	x, y = context.coordinates[particle]
	initial_configuration: np.ndarray = GCStateLayout().pack_components(np.asarray([x], dtype=float), np.asarray([y], dtype=float))
	problem: InitialValueProblem = InitialValueProblem(context.dynamics, initial_configuration, layout=GCStateLayout())
	method: BM4Implicit = BM4Implicit(
		newton_absolute_tolerance=context.config.absolute_tolerance,
		newton_relative_tolerance=context.config.relative_tolerance,
		newton_max_iterations=context.config.max_iterations,
		newton_jacobian_method="analytic",
		newton_jacobian_relative_step=context.config.jacobian_relative_step,
		progress=False,
	)
	started: float = perf_counter()
	solution: States = simulate(problem, method, context.request)
	runtime = perf_counter() - started
	position_x: np.ndarray
	position_y: np.ndarray
	position_x, position_y = solution.positions()
	positions: np.ndarray = np.stack((position_x[0], position_y[0]), axis=0)
	diagnostics = solution.diagnostics
	iterations: np.ndarray = np.asarray(diagnostics["nonlinear_iterations"], dtype=int)
	residuals: np.ndarray = np.asarray(diagnostics["nonlinear_residual_norms"], dtype=float)
	tolerances: np.ndarray = np.asarray(diagnostics["nonlinear_tolerances"], dtype=float)
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
	initial_configuration: np.ndarray,
	*,
	config: ParallelBM4RecurrenceConfig,
) -> ParallelBM4RecurrenceResult:
	"""Run one independent single-particle BM4 integration per process task."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not (isinstance(initial_configuration, np.ndarray) and GCStateLayout().state_dimension == 2):
		raise TypeError("`initial_configuration` must be np.ndarray.")
	if not isinstance(config, ParallelBM4RecurrenceConfig):
		raise TypeError("`config` must be ParallelBM4RecurrenceConfig.")
	require_uncoupled_bm4(config.coupling_frequency)
	coordinates: np.ndarray = _validated_campaign_coordinates(initial_configuration, config.particle_count)
	worker_count: int = min(config.worker_count, config.particle_count)
	started: float = perf_counter()
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
		pending: set[Future[_ParticlePayload]] = set(future_particles)
		try:
			while pending:
				completed_futures: set[Future[_ParticlePayload]]
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
						payload: _ParticlePayload = future.result()
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
	ordered: tuple[_ParticlePayload, ...] = tuple(payloads[particle] for particle in range(config.particle_count))
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


def _validated_campaign_coordinates(
	initial_configuration: np.ndarray, particle_count: int,
) -> np.ndarray:
	"""Resolve concrete initial positions with exactly the campaign particle count."""
	initial_state = initial_configuration
	if initial_state is None:
		raise ValueError("The initial configuration must contain an initial state.")
	initial_x: np.ndarray
	initial_y: np.ndarray
	initial_x, initial_y = GCStateLayout().positions(initial_state)
	if initial_x.size != particle_count:
		raise ValueError(
			"The initial configuration particle count must match the campaign."
		)
	coordinates: np.ndarray = np.column_stack((initial_x, initial_y))
	return coordinates


__all__ = [
	"ParallelBM4RecurrenceConfig",
	"ParallelBM4RecurrenceResult",
	"run_parallel_bm4_recurrence",
]
