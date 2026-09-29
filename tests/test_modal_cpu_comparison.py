"""Scientific geometry, synchronized JAX timing and archive roundtrip controls."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from contracts.execution_options import ExecutionOptions
from contracts.result import IntegrationData
from execution._modal_benchmark import benchmark_payload
from execution._modal_worker import encode_job, execute_payload
from execution.execution import Execution
from initial_conditions.star import radial_star
from methods.extended.bm4 import BM4Midpoint
from potential.potential import Potential
from studies.modal_cpu_comparison import (
    ModalCPUStarConfig, build_star_problem, run_cpu_star_comparison,
    save_cpu_star_comparison, load_cpu_star_comparison,
)


def settings() -> ModalCPUStarConfig:
    """Match the requested geometry with explicit physical and numerical controls."""
    return ModalCPUStarConfig(1.5, .06, (0, 1), 3, .3, np.pi / 8,
                             8, 6, .5, .05, .35, (.5, .5), 0.,
                             10, 50, (1, 2, 4), 3, 1e-10, 1e-11)


def field() -> Potential:
    """Use a small harmonic field with the same normalized forcing period."""
    base = Potential.random(A=.01, M=3, nx=16, ny=16, seed=27)
    return Potential(base.grid, mean=base.mean, modes=base.modes, frequencies=np.ones(1))


class ModalCPUComparisonTests(unittest.TestCase):
    def test_star_spans_cell_radius_fractions_and_preserves_arm_order(self):
        potential, config = field(), settings()
        problem, request = build_star_problem(potential, config)
        x, y = problem.initial_configuration.layout.positions(problem.initial_state)
        relative = np.column_stack((x, y)) / potential.grid.period - .5
        radii = np.linalg.norm(relative, axis=1).reshape(8, 6) / .5
        np.testing.assert_allclose(radii, np.tile([.05, .11, .17, .23, .29, .35], (8, 1)))
        directions = relative.reshape(8, 6, 2)[:, -1] / (.5 * .35)
        angles = np.arange(8) * np.pi / 4
        np.testing.assert_allclose(directions, np.column_stack((np.cos(angles), np.sin(angles))), atol=1e-15)
        self.assertEqual(problem.particle_count, 48)
        self.assertEqual(request.output_times.size, 501)
        np.testing.assert_allclose(np.diff(request.output_times), .02)
        self.assertEqual(request.t_span, (0., 10.))

    def test_default_star_radii_remain_unchanged_and_invalid_inner_radius_fails(self):
        initial = radial_star(center=(0, 0), arm_length=3, arms=2, particles_per_arm=3)
        x, y = initial.positions(initial.initial_state)
        np.testing.assert_allclose(np.hypot(x, y), [1, 2, 3, 1, 2, 3])
        for radius in (0., -1., 4., np.nan):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                radial_star(center=(0, 0), arm_length=3, arms=2, particles_per_arm=3, inner_radius=radius)

    def test_invalid_sampling_and_geometry_fail_before_remote_work(self):
        for change in (dict(cycles=0), dict(steps_per_cycle=True), dict(inner_radius_fraction=.4),
                       dict(center_fraction=(.01, .5)), dict(repetitions=2), dict(cpu_counts=(1, 2))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_star_problem(field(), replace(settings(), **change))

    def test_study_collects_and_persists_all_cpu_results_without_cloud(self):
        config = replace(settings(), cycles=1, steps_per_cycle=2)
        potential = field()

        class Worker(Execution):
            """Model a completed remote benchmark with known timing controls."""
            def __init__(self, app_name, function_name, *, record_directory):
                self.cores = int(function_name.rsplit('_', 1)[1])
                self.last_call_id = f'fc-{self.cores}'
                self.last_record_path = Path(record_directory) / f'{self.cores}.json'

            def run(self, problem, method, request, *, options):
                # Numerical work uses the real local NumPy method; only the
                # transport and timer are substituted in this orchestration test.
                data = method.integrate(problem, request)
                diagnostics = dict(data.diagnostics)
                diagnostics.update(benchmark_cpu_cores=self.cores, benchmark_repetitions=3,
                                   benchmark_wall_seconds=np.array([3., 2., 4.]) / self.cores,
                                   benchmark_first_seconds=8., benchmark_environment='{}')
                return IntegrationData(data.t, data.states, diagnostics)

        with tempfile.TemporaryDirectory() as temporary, patch('studies.modal_cpu_comparison.Execution_Modal', Worker):
            comparison = run_cpu_star_comparison(potential, config, {}, app_name='offline', record_directory=temporary)
            self.assertEqual(comparison.metadata['runs']['4']['speedup'], 4.)
            self.assertEqual(comparison.metadata['runs']['2']['maximum_periodic_discrepancy'], 0.)
            self.assertFalse(comparison.metadata['accuracy_certified'])
            destination = str(Path(temporary) / 'archive')
            save_cpu_star_comparison(comparison, destination)
            loaded = load_cpu_star_comparison(destination)
            for cores in (1, 2, 4):
                np.testing.assert_array_equal(loaded.solutions[cores].states, comparison.solutions[cores].states)
            self.assertEqual(loaded.metadata['config']['radius_over_period'], .5)

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX is optional')
    def test_remote_worker_accepts_jax_and_benchmark_returns_synchronized_samples(self):
        import jax
        config = replace(settings(), cycles=1, steps_per_cycle=2)
        problem, request = build_star_problem(field(), config)
        payload = encode_job(problem, BM4Midpoint(), request, ExecutionOptions(backend='jax'))
        with jax.experimental.enable_x64():
            _, digest, single = execute_payload(payload)
            _, measured_digest, measured = benchmark_payload(payload, cpu_cores=2, repetitions=3)
        self.assertEqual(digest, measured_digest)
        self.assertIsInstance(measured.states, np.ndarray)
        self.assertEqual(measured.states.dtype, np.float64)
        np.testing.assert_allclose(single.states, measured.states, rtol=1e-13, atol=1e-13)
        self.assertEqual(measured.diagnostics['benchmark_wall_seconds'].shape, (3,))
        self.assertTrue(np.all(measured.diagnostics['benchmark_wall_seconds'] > 0))
        self.assertTrue(json.loads(measured.diagnostics['benchmark_environment'])['jax_enable_x64'])
        self.assertEqual(measured.diagnostics['output_interpolation_count'], 0)
        self.assertEqual(measured.diagnostics['nonlinear_unknown_dimension'], 0)


if __name__ == '__main__':
    unittest.main()
