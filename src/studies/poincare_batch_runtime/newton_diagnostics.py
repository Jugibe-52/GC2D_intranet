"""Observe Newton iterates without repeating or changing BM4 map evaluations."""
from array import array
from contextlib import contextmanager
import gzip
import inspect
import os
from collections.abc import Iterator, Callable
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


class NewtonHistory:
    """Ragged history: iteration zero is the zero-multiplier initial guess."""

    def __init__(self, expected_steps, progress_every):
        self.expected_steps = expected_steps
        self.progress_every = progress_every
        self.offsets = []
        self.residuals = array('d')
        self.mu_norms = array('d')

    def record(self, iteration, residual_norm, threshold, multiplier, time, step):
        if iteration == 0:
            self.offsets.append(len(self.residuals))
        self.residuals.append(residual_norm)
        self.mu_norms.append(float(np.linalg.norm(multiplier, ord=np.inf)))
        number = len(self.offsets)
        if residual_norm <= threshold and self.progress_every and (
            number % self.progress_every == 0 or number == self.expected_steps
        ):
            print(f'PID {os.getpid()}: {number}/{self.expected_steps} steps, '
                  f't={time + step:g}, Newton corrections={iteration}, '
                  f'residual/tolerance={residual_norm / threshold:.3g}', flush=True)

    def arrays(self):
        assert len(self.offsets) == self.expected_steps, 'Unexpected main/shadow step count.'
        return {
            'offsets': np.asarray(self.offsets + [len(self.residuals)], dtype=np.int64),
            'residuals': np.asarray(self.residuals, dtype=np.float64),
            'mu_norms': np.asarray(self.mu_norms, dtype=np.float64),
        }


@contextmanager
def _observe_shared_newton(module: ModuleType, history: NewtonHistory) -> Iterator[None]:
    """Observe the common residual callback without copying solver source code."""
    original = getattr(module, '_solve_newton')

    def observed_solve(
        evaluate: Callable[[np.ndarray], tuple[np.ndarray, Any]],
        initial: np.ndarray, update: Callable[..., np.ndarray], **options: Any,
    ) -> Any:
        # The spatial projection callback owns the step clock and the reduced
        # or simultaneous layout. Reading its closure does not evaluate a map.
        values = inspect.getclosurevars(evaluate).nonlocals
        time, step = values['t'], values['h']
        size = values['size']
        simultaneous = values['simultaneous']
        iteration = 0

        def record(unknown: np.ndarray, residual: np.ndarray) -> None:
            nonlocal iteration
            multiplier = unknown[2 * size:] if simultaneous else unknown
            history.record(iteration, float(np.linalg.norm(residual, ord=np.inf)),
                           options['tolerance'], multiplier, time, step)
            iteration += 1

        def observed_residual(unknown: np.ndarray) -> tuple[np.ndarray, Any]:
            residual, payload = evaluate(unknown)
            record(unknown, residual)
            return residual, payload

        cached = options.get('initial_evaluation')
        if cached is not None:
            record(initial, cached[0])
        return original(observed_residual, initial, update, **options)

    setattr(module, '_solve_newton', observed_solve)
    try:
        yield
    finally:
        setattr(module, '_solve_newton', original)


@contextmanager
def observe_newton(expected_steps, progress_every=1000):
    """Temporarily observe the canonical projection engine in this process.

    The callback reads already-computed residuals and multipliers. It neither
    evaluates the field nor modifies solver state or the source snapshot.
    Each spawned process owns its callback; restoration also occurs on failure.
    """
    from methods.extended.core import projection

    history = NewtonHistory(expected_steps, progress_every)
    with _observe_shared_newton(projection, history):
        yield history


def write_diagnostic_tables(directory, metadata, arrays):
    """Write lossless compressed CSV tables, grouped by worker to bound memory."""
    n = metadata['complete_steps']
    times = arrays['times'][1:]
    iterations = arrays['nonlinear_iterations']
    names = ('newton_steps.csv.gz', 'newton_iterations.csv.gz')
    with gzip.open(directory / names[0], 'wt', encoding='utf-8', newline='') as steps_file, \
         gzip.open(directory / names[1], 'wt', encoding='utf-8', newline='') as history_file:
        for w in range(metadata['process_count']):
            residual = arrays['nonlinear_residuals'][w]
            tolerance = arrays['nonlinear_tolerances'][w]
            offsets = arrays['newton_history_offsets'][w]
            counts = np.diff(offsets)
            np.testing.assert_array_equal(counts, iterations[w] + 1)
            selected = slice(offsets[0], offsets[-1])
            history_residual = arrays['newton_history_residuals'][selected]
            history_mu = arrays['newton_history_mu_norms'][selected]
            local_offsets = offsets - offsets[0]
            final = local_offsets[1:] - 1
            np.testing.assert_array_equal(history_residual[final], residual)
            np.testing.assert_array_equal(history_mu[final], arrays['projection_multiplier_norms'][w])
            frame = pd.DataFrame({
                'worker': w + 1, 'step': np.arange(1, n + 1),
                'cycle': (np.arange(n) // metadata['steps_per_cycle']) + 1,
                'time_normalized': times, 'time_s': times * metadata['time_scale_s'],
                'newton_corrections': iterations[w], 'residual_evaluations': counts,
                'initial_residual_inf': history_residual[local_offsets[:-1]],
                'final_residual_inf': residual, 'tolerance': tolerance,
                'residual_over_tolerance': residual / tolerance,
                'mu_inf': arrays['projection_multiplier_norms'][w],
                'converged': residual <= tolerance,
            })
            frame.to_csv(steps_file, index=False, header=w == 0, float_format='%.17g')
            step_indices = np.repeat(np.arange(n), counts)
            iterate = np.arange(len(history_residual)) - np.repeat(local_offsets[:-1], counts)
            pd.DataFrame({
                'worker': w + 1, 'step': step_indices + 1,
                'time_normalized': times[step_indices], 'newton_iteration': iterate,
                'residual_inf': history_residual, 'tolerance': tolerance[step_indices],
                'residual_over_tolerance': history_residual / tolerance[step_indices],
                'mu_inf': history_mu,
                'accepted': iterate == iterations[w, step_indices],
            }).to_csv(history_file, index=False, header=w == 0, float_format='%.17g')
    return names
