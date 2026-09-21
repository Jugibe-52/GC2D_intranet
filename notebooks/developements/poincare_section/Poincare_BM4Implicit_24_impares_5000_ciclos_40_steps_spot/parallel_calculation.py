"""Spawn independent BM4 particle groups and merge them by original particle ID.

Scientific parameters are supplied by calculo.ipynb. The frozen numerical method
is unchanged; each group uses its own state norm for Newton's stopping rule.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from contextlib import nullcontext
import multiprocessing as mp

from newton_diagnostics import observe_newton
import os
import time

import numpy as np

from study_io import load_snapshot, utc_now


_START_BARRIER = None


def _initialize_worker(barrier):
    global _START_BARRIER
    _START_BARRIER = barrier


def calculate_group(indices, initial_xy, settings):
    """Integrate a particle group; reusable serially for partition validation."""
    from threadpoolctl import threadpool_info, threadpool_limits

    potential, _, snapshot_hash = load_snapshot()
    from dynamics import GuidingCenterDynamics
    from initial_conditions import GCInitialConfiguration
    from simulation import BM4Implicit, InitialValueProblem, SimulationRequest, simulate

    indices = np.asarray(indices, dtype=int)
    xy0 = np.asarray(initial_xy, dtype=np.float64)[indices]
    initial = GCInitialConfiguration.from_components(x=xy0[:, 0], y=xy0[:, 1])
    problem = InitialValueProblem(GuidingCenterDynamics(potential, rho=settings['rho']), initial)
    request = SimulationRequest.uniform(
        t_span=tuple(settings['t_span']), max_step=settings['step'],
        sample_count=settings['n_steps'] + 1,
    )
    method = BM4Implicit(
        coupling_frequency=settings['coupling_frequency'],
        newton_absolute_tolerance=settings['newton_atol'],
        newton_relative_tolerance=settings['newton_rtol'],
        newton_max_iterations=settings['newton_max_iterations'],
        newton_jacobian_relative_step=settings['jacobian_relative_step'],
        newton_jacobian_method='analytic', nonlinear_solver='newton', progress=False,
    )
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        assert all(item['num_threads'] == 1 for item in pools)
        # Exactly one task per process; the barrier proves all workers are ready
        # before any integration starts, even for very short validation runs.
        if _START_BARRIER is not None:
            _START_BARRIER.wait(timeout=180)
        started_utc = utc_now()
        started = time.perf_counter()
        cpu_started = time.process_time()
        observer = (observe_newton(settings['n_steps'], settings.get('progress_every', 1000))
                    if settings.get('record_newton_history', True) else nullcontext(None))
        with observer as history:
            solution = simulate(problem, method, request)
        history_arrays = history.arrays() if history is not None else None
        cpu_seconds = time.process_time() - cpu_started
        finished = time.perf_counter()
        finished_utc = utc_now()
    diagnostics = solution.diagnostics
    assert diagnostics['step_count'] == settings['n_steps']
    assert diagnostics['projection_solver_formulation'] == 'bm4_implicit_reduced'
    assert diagnostics['newton_jacobian_method'] == 'analytic'
    xy = np.stack(solution.positions(), axis=-1).transpose(1, 0, 2)
    np.testing.assert_array_equal(xy[0], xy0)
    residuals = np.asarray(diagnostics['nonlinear_residual_norms'])
    tolerances = np.asarray(diagnostics['nonlinear_tolerances'])
    assert residuals.shape == tolerances.shape == (settings['n_steps'],)
    assert np.isfinite(xy).all() and np.isfinite(residuals).all()
    assert np.all(tolerances > 0)
    assert np.all(residuals <= tolerances * (1 + 32*np.finfo(float).eps))
    return {
        'indices': indices, 'times': solution.t, 'xy': xy,
        'newton_history': history_arrays,
        'diagnostics': {key: np.asarray(diagnostics[key]) for key in (
            'nonlinear_iterations', 'nonlinear_residual_norms',
            'nonlinear_tolerances', 'projection_multiplier_norms')},
        'record': {
            'pid': os.getpid(), 'particle_ids': (indices + 1).tolist(),
            'particle_count': len(indices), 'started_utc': started_utc,
            'finished_utc': finished_utc, 'started_monotonic_s': started,
            'finished_monotonic_s': finished, 'integration_seconds': finished - started,
            'cpu_seconds': cpu_seconds, 'snapshot_sha256': snapshot_hash,
            'maximum_residual_to_tolerance': float(np.max(residuals / tolerances)),
            'threadpools': pools,
        },
    }


@dataclass
class ParallelSolution:
    """Merged trajectory with worker-by-step diagnostics (not pooled residuals)."""

    t: np.ndarray
    states: np.ndarray
    diagnostics: dict
    workers: list
    parallel_wall_seconds: float
    simultaneous_integration_seconds: float

    def positions(self):
        n = self.states.shape[0] // 2
        return self.states[:n], self.states[n:]


def simulate_parallel(initial_xy, settings, processes=16):
    """Partition by round-robin ID, spawn all workers, and restore global order."""
    xy0 = np.asarray(initial_xy, dtype=np.float64)
    if isinstance(processes, bool) or not isinstance(processes, int) or not 1 <= processes <= len(xy0):
        raise ValueError('Process count must be an integer between 1 and the particle count.')
    load_snapshot()  # Materialize and verify the immutable source before spawning.
    from checkpoint_calculation import calculate_checkpointed_group
    groups = [np.arange(i, len(xy0), processes) for i in range(processes)]
    context = mp.get_context('spawn')
    barrier = context.Barrier(processes)
    started = time.perf_counter()
    outputs = {}
    with ProcessPoolExecutor(max_workers=processes, mp_context=context,
                             initializer=_initialize_worker, initargs=(barrier,)) as pool:
        futures = {pool.submit(calculate_checkpointed_group, group, xy0, settings): i
                   for i, group in enumerate(groups)}
        for future in as_completed(futures):
            i = futures[future]
            outputs[i] = future.result()  # Any failure prevents publishing a complete run.
            item = outputs[i]['record']
            print(f"Worker {i+1}/{processes}: PID {item['pid']}, "
                  f"{item['particle_count']} particles, {item['integration_seconds']:.2f} s", flush=True)
    ordered = [outputs[i] for i in range(processes)]
    times = ordered[0]['times']
    xy = np.empty((len(times), len(xy0), 2), dtype=np.float64)
    for item in ordered:
        np.testing.assert_array_equal(item['times'], times)
        xy[:, item['indices'], :] = item['xy']
    workers = [dict(worker=i+1, **item['record']) for i, item in enumerate(ordered)]
    assert len({record['pid'] for record in workers}) == processes
    overlap = min(w['finished_monotonic_s'] for w in workers) - max(w['started_monotonic_s'] for w in workers)
    assert overlap > 0, 'Workers did not integrate concurrently.'
    diagnostics = {
        key: np.stack([item['diagnostics'][key] for item in ordered])
        for key in ordered[0]['diagnostics']
    }
    if settings.get('record_newton_history', True):
        histories = [item['newton_history'] for item in ordered]
        bases = np.cumsum([0] + [len(h['residuals']) for h in histories[:-1]])
        diagnostics['newton_history_offsets'] = np.stack([
            h['offsets'] + base for h, base in zip(histories, bases)])
        diagnostics['newton_history_residuals'] = np.concatenate([h['residuals'] for h in histories])
        diagnostics['newton_history_mu_norms'] = np.concatenate([h['mu_norms'] for h in histories])
    diagnostics.update(step_count=settings['n_steps'],
                       projection_solver_formulation='bm4_implicit_reduced' if settings['method'] == 'BM4Implicit' else 'bm4_midpoint',
                       newton_jacobian_method='analytic' if settings['method'] == 'BM4Implicit' else None)
    states = np.concatenate((xy[..., 0].T, xy[..., 1].T), axis=0)
    wall_seconds = time.perf_counter() - started
    return ParallelSolution(times, states, diagnostics, workers, wall_seconds, overlap)
