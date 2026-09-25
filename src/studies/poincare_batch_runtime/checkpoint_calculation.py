"""Restartable fixed-grid orchestration of frozen implicit and midpoint BM4."""
from pathlib import Path
from contextlib import nullcontext
import hashlib
import json
import os
import time

import numpy as np

from newton_diagnostics import observe_newton
from study_io import ROOT, atomic_json, digest, load_snapshot, utc_now


class CheckpointTestInterruption(RuntimeError):
    """Deliberate interruption used only by the disposable validation run."""


def calculate_checkpointed_group(indices, initial_xy, settings):
    """Save immutable step chunks and resume from their last verified endpoint.

    Each call uses the original BM4 reduced step and its exact global schedule
    t0 + k*h. No interpolated endpoint or restarted local clock is introduced.
    Every multiplier starts at zero, as in the original integrator. Checkpoints
    contain every accepted state, per-step diagnostics and all Newton iterates.
    """
    from threadpoolctl import threadpool_info, threadpool_limits
    potential, _, snapshot = load_snapshot()
    from dynamics import GuidingCenterDynamics
    from initial_conditions import GCInitialConfiguration
    from contracts.problem import InitialValueProblem
    from formulations.gc import GCDoubledMaps
    from methods.extended.core.composition import BM4
    from methods.extended.core.projection import solve_projection
    from methods.extended.core.midpoint import midpoint_step
    from methods._nonlinear import SolverOptions
    import parallel_calculation

    indices = np.asarray(indices, dtype=int)
    xy0 = np.asarray(initial_xy, dtype=float)[indices]
    initial = GCInitialConfiguration.from_components(x=xy0[:, 0], y=xy0[:, 1])
    problem = InitialValueProblem(GuidingCenterDynamics(potential, rho=settings['rho']), initial)
    prepared = GCDoubledMaps(problem, coupling_frequency=settings['coupling_frequency'])
    implicit = settings['method'] == 'BM4Implicit'
    solver = SolverOptions('newton', settings['newton_atol'], settings['newton_rtol'],
                           settings['newton_max_iterations'])
    particle_ids = np.asarray(settings['particle_ids'])[indices]
    n = settings['n_steps']
    t0, tf = settings['t_span']
    h = (tf - t0) / n
    assert h == settings['step']
    chunk_size = settings['checkpoint_steps']
    folder = Path(settings['checkpoint_directory']) / f'worker_{int(indices[0]) + 1:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    contract = {
        'schema': 1, 'settings': {k: settings[k] for k in (
            'method', 'particle_ids', 'rho', 'coupling_frequency', 'newton_atol', 'newton_rtol',
            'newton_max_iterations', 'jacobian_relative_step', 't_span', 'step',
            'n_steps', 'checkpoint_steps')},
        'indices': indices.tolist(), 'initial_xy': xy0.tolist(), 'snapshot': snapshot,
        'driver_sha256': digest(Path(__file__)),
        'observer_sha256': digest(ROOT / 'newton_diagnostics.py'),
        'numpy_version': np.__version__,
    }
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    parts = []
    completed = 0
    value = problem.initial_state.copy()
    loaded = 0
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        assert all(p['num_threads'] == 1 for p in pools)
        if parallel_calculation._START_BARRIER is not None:
            parallel_calculation._START_BARRIER.wait(timeout=180)
        started_utc = utc_now()
        started = time.perf_counter()
        cpu_started = time.process_time()
        for marker in sorted(folder.glob('chunk_*.json')):
            record = json.loads(marker.read_text())
            assert record['fingerprint'] == fingerprint, 'Checkpoint parameters/source mismatch.'
            assert record['start_step'] == completed and completed < record['end_step'] <= n
            file = folder / (marker.stem + '.npz')
            assert file.name == record['file'] and digest(file) == record['sha256'], 'Checkpoint checksum mismatch.'
            with np.load(file, allow_pickle=False) as saved:
                part = {k: saved[k] for k in saved.files}
            np.testing.assert_array_equal(part['states'][:, 0], value)
            assert part['states'].shape == (len(value), record['end_step'] - completed + 1)
            parts.append(part)
            value = part['states'][:, -1].copy()
            completed = record['end_step']
            loaded += 1
        resumed_steps = completed
        if completed:
            print(f'Worker {indices[0] + 1}: resumed verified step {completed}/{n}', flush=True)
        new_chunks = 0
        while completed < n:
            end = min(completed + chunk_size, n)
            length = end - completed
            states = np.empty((len(value), length + 1))
            states[:, 0] = value
            counters = np.empty(length, dtype=np.int64)
            residuals = np.empty(length)
            tolerances = np.empty(length)
            mu = np.empty(length)
            with (observe_newton(length, progress_every=0) if implicit else nullcontext(None)) as history:
                for local, k in enumerate(range(completed, end)):
                    if implicit:
                        threshold = settings['newton_atol'] + settings['newton_rtol'] * max(
                            1.0, float(np.linalg.norm(value, ord=np.inf)))
                        result = solve_projection(
                            prepared, BM4, solver, 'reduced_multiplier', t0 + k * h, value, h,
                            jacobian_relative_step=settings['jacobian_relative_step'],
                            jacobian_method='analytic')
                        value = result.state
                        assert np.isfinite(value).all() and result.residual_norm <= threshold
                        states[:, local + 1] = value
                        counters[local] = result.iterations
                        residuals[local] = result.residual_norm
                        tolerances[local] = threshold
                        mu[local] = np.linalg.norm(result.multiplier, ord=np.inf)
                    else:
                        result = midpoint_step(prepared, BM4, t0 + k * h, value, h)
                        value = result.state
                        assert np.isfinite(value).all()
                        states[:, local + 1] = value
                        mu[local] = result.copy_separation_norm
            if implicit:
                hist = history.arrays()
                part = {'states': states, 'nonlinear_iterations': counters,
                        'nonlinear_residual_norms': residuals, 'nonlinear_tolerances': tolerances,
                        'projection_multiplier_norms': mu,
                        'history_offsets': hist['offsets'], 'history_residuals': hist['residuals'],
                        'history_mu_norms': hist['mu_norms']}
            else:
                part = {'states': states, 'copy_separation_norms': mu}
            file = folder / f'chunk_{end:08d}.npz'
            temporary = file.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **part)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(file)
            atomic_json(file.with_suffix('.json'), {
                'fingerprint': fingerprint, 'file': file.name, 'sha256': digest(file),
                'start_step': completed, 'end_step': end, 'committed_utc': utc_now(),
                'particle_ids': particle_ids.tolist(), 'pid': os.getpid()})
            parts.append(part)
            completed = end
            new_chunks += 1
            atomic_json(folder / 'progress.json', {'completed_steps': completed, 'total_steps': n,
                        'updated_utc': utc_now(), 'pid': os.getpid(), 'resumed_steps': resumed_steps})
            print(f'Worker {indices[0] + 1}: checkpoint {completed}/{n} '
                  f'({100 * completed / n:.1f}%), t={t0 + completed*h:g}', flush=True)
            if settings.get('test_stop_after_chunks') == new_chunks:
                raise CheckpointTestInterruption('Simulated interruption after a durable checkpoint.')
        cpu_seconds = time.process_time() - cpu_started
        finished = time.perf_counter()
        finished_utc = utc_now()
    states = np.concatenate([parts[0]['states']] + [p['states'][:, 1:] for p in parts[1:]], axis=1)
    history_result = None
    if implicit:
        diagnostics = {k: np.concatenate([p[k] for p in parts]) for k in (
            'nonlinear_iterations', 'nonlinear_residual_norms', 'nonlinear_tolerances', 'projection_multiplier_norms')}
        bases = np.cumsum([0] + [len(p['history_residuals']) for p in parts[:-1]])
        offsets = np.concatenate([p['history_offsets'][:-1] + b for p, b in zip(parts, bases)] +
                                 [np.array([sum(len(p['history_residuals']) for p in parts)])])
        np.testing.assert_array_equal(np.diff(offsets), diagnostics['nonlinear_iterations'] + 1)
        history_result = {'offsets': offsets,
                          'residuals': np.concatenate([p['history_residuals'] for p in parts]),
                          'mu_norms': np.concatenate([p['history_mu_norms'] for p in parts])}
    else:
        diagnostics = {'copy_separation_norms': np.concatenate([p['copy_separation_norms'] for p in parts])}
    xy = np.stack([states[:len(indices)].T, states[len(indices):].T], axis=-1)
    return {
        'indices': indices, 'times': t0 + np.arange(n + 1) * h, 'xy': xy,
        'newton_history': history_result,
        'diagnostics': diagnostics,
        'record': {'pid': os.getpid(), 'particle_ids': particle_ids.tolist(),
                   'particle_count': len(indices), 'started_utc': started_utc, 'finished_utc': finished_utc,
                   'started_monotonic_s': started, 'finished_monotonic_s': finished,
                   'integration_seconds': finished - started, 'cpu_seconds': cpu_seconds,
                   'snapshot_sha256': snapshot, 'threadpools': pools,
                   'resumed_steps': resumed_steps, 'new_steps': n - resumed_steps,
                   'checkpoint_chunks': len(parts), 'checkpoint_fingerprint': fingerprint,
                   'maximum_residual_to_tolerance': float(np.max(
                       diagnostics['nonlinear_residual_norms'] / diagnostics['nonlinear_tolerances'])) if implicit else None},
    }
