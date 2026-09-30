"""Physical scaling, radial identity, cycle sampling and saved rho selection."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from contracts.execution_options import ExecutionOptions
from execution.execution import Execution
from potential import Grid, Potential
from studies.dimensional_h5_midpoint import DimensionalH5Field
from studies.poincare_rho_sweep import (
    RhoStarConfig, build_rho_star, run_rho_star, run_and_save_rho_star,
    modal_rho_executor,
)
from visualization.poincare_rho_sweep import export_rho_sweep, load_available_rho_runs


def field():
    """Small smooth dimensional field with one time-periodic mode."""
    length, n = 0.18, 16
    grid = Grid(-length/2, -length/2, length/n, length/n, n, n, length)
    xx, yy = np.meshgrid(np.arange(n) * 2*np.pi/n, np.arange(n) * 2*np.pi/n, indexing='ij')
    raw = Potential(grid, mean=1e-5*np.cos(xx)*np.cos(yy),
                    modes=np.asarray([1e-6*np.exp(1j*xx)]), frequencies=np.array([1.]),
                    interpolation_order=3)
    return DimensionalH5Field(raw, raw, 1., 1.5, (0, 1))


class RhoSweepTests(unittest.TestCase):
    def test_distinct_radial_ranks_geometry_and_physical_rho(self):
        config = RhoStarConfig(rho_hat=.3)
        source = field()
        prepared = build_rho_star(source, config, source_sha256='synthetic')
        m = prepared.metadata
        xy = np.asarray(m['initial_positions_m'])
        radii = np.linalg.norm(xy - m['star_center_m'], axis=1)
        np.testing.assert_allclose(radii, np.linspace(0, .85*.09, 40), atol=1e-16)
        self.assertEqual(np.unique(radii).size, 40)
        np.testing.assert_array_equal(np.bincount(m['arm_id'])[1:], np.full(8, 5))
        angles = np.arctan2(xy[1:, 1], xy[1:, 0])
        expected = 2*np.pi*(np.arange(1, 40) % 8)/8
        np.testing.assert_allclose(np.exp(1j*angles), np.exp(1j*expected), atol=1e-15)
        self.assertAlmostEqual(prepared.problem.dynamics.rho, .3*.06/(2*np.pi))
        np.testing.assert_array_equal(prepared.request.output_times, np.arange(5001))
        self.assertEqual(prepared.request.max_step, .02)
        self.assertEqual(m['expected_step_count'], 250000)
        # Spectral averaging must preserve the dimensionless product k*rho.
        scale = 2*np.pi/.06
        g = source.raw.grid
        normalized = Potential(Grid(g.x0*scale, g.y0*scale, g.dx*scale, g.dy*scale,
                                    g.nx, g.ny, g.period*scale), mean=source.raw.mean,
                               modes=source.raw.modes, frequencies=source.raw.frequencies)
        np.testing.assert_allclose(prepared.problem.dynamics.effective_potential.mean,
                                   normalized.gyroaverage(.3).mean, rtol=1e-13, atol=1e-20)

    def test_cycle_only_roundtrip_and_no_resubmission(self):
        config = RhoStarConfig(rho_hat=.3, cycles=2)
        prepared = build_rho_star(field(), config, source_sha256='synthetic')
        executor = Execution()
        with tempfile.TemporaryDirectory() as tmp:
            dest = str(Path(tmp)/'rho_030')
            saved = run_and_save_rho_star(prepared, dest, executor=executor, options=ExecutionOptions())
            self.assertEqual(saved.solution.states.shape, (80, 3))
            self.assertEqual(saved.solution.diagnostics['step_count'], 100)
            self.assertFalse(any(isinstance(v, np.ndarray) for v in saved.solution.diagnostics.values()))
            with patch.object(executor, 'run', side_effect=AssertionError('must not resubmit')):
                again = run_and_save_rho_star(prepared, dest, executor=executor, options=ExecutionOptions())
            np.testing.assert_array_equal(again.solution.states, saved.solution.states)
            runs = load_available_rho_runs({0.: str(Path(tmp)/'missing'), .3: dest})
            path = export_rho_sweep(runs, Path(tmp)/'viewer.html', rho_values=(0., .3, .5))
            cfg = json.loads(re.search(r'const cfg=(.*);', path.read_text()).group(1))
            self.assertEqual(cfg['axisLabels'], ['R (m)', 'Z (m)'])
            self.assertEqual([d['key'] for d in cfg['datasets']], ['0.00', '0.30', '0.50'])
            self.assertEqual([d['key'] for d in cfg['datasets'] if 'panels' in d], ['0.30'])
            self.assertIn('85.00%', cfg['particleLabels']['40'])
            self.assertEqual(cfg['panels'][1]['shape'], [2, 40, 2])
            other = build_rho_star(field(), replace(config, rho_hat=0.), source_sha256='synthetic')
            runs[0.] = run_rho_star(other, executor=Execution(), options=ExecutionOptions())
            export_rho_sweep(runs, path, rho_values=(0., .3, .5))
            cfg = json.loads(re.search(r'const cfg=(.*);', path.read_text()).group(1))
            self.assertEqual([d['key'] for d in cfg['datasets'] if 'panels' in d], ['0.00', '0.30'])
            self.assertEqual(cfg['selectedDataset'], '0.30')

    def test_normalized_space_preserves_physical_trajectory_and_time(self):
        source = field()
        config = RhoStarConfig(rho_hat=.3, cycles=2)
        physical = build_rho_star(source, config, source_sha256='synthetic')
        normalized = build_rho_star(source, replace(config, spatial_normalization='characteristic_length'),
                                    source_sha256='synthetic')
        scale = 2*np.pi/config.characteristic_length
        origin = np.array([source.raw.grid.x0, source.raw.grid.y0])
        self.assertEqual(normalized.problem.dynamics.rho, .3)
        self.assertAlmostEqual(normalized.metadata['rho_m'], physical.metadata['rho_m'])
        np.testing.assert_array_equal(normalized.request.output_times, physical.request.output_times)
        np.testing.assert_allclose(normalized.problem.dynamics.effective_potential.mean / scale**2,
                                   physical.problem.dynamics.effective_potential.mean, rtol=1e-12, atol=1e-19)
        a = run_rho_star(physical, executor=Execution(), options=ExecutionOptions())
        b = run_rho_star(normalized, executor=Execution(), options=ExecutionOptions())
        for axis, dimensional in enumerate(a.solution.positions()):
            np.testing.assert_allclose(b.solution.positions()[axis]/scale + origin[axis],
                                       dimensional, rtol=1e-11, atol=1e-13)
        with tempfile.TemporaryDirectory() as tmp:
            path = export_rho_sweep({.3: b}, Path(tmp)/'normalized.html', rho_values=(0., .3, .5))
            cfg = json.loads(re.search(r'const cfg=(.*);', path.read_text()).group(1))
            self.assertEqual(cfg['axisLabels'], ['R_hat', 'Z_hat'])
            self.assertEqual(cfg['panels'][1]['shape'], [2, 40, 2])
            # Different unit systems must never appear under one rho selector.
            a.metadata['config']['rho_hat'] = 0.
            with self.assertRaisesRegex(ValueError, 'same spatial units'):
                export_rho_sweep({0.: a, .3: b}, path, rho_values=(0., .3, .5))
            destination = str(Path(tmp)/'saved')
            run_and_save_rho_star(normalized, destination, executor=Execution(), options=ExecutionOptions())
            with self.assertRaisesRegex(ValueError, 'different scientific inputs'):
                run_and_save_rho_star(physical, destination, executor=Execution(), options=ExecutionOptions())

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX is optional')
    def test_jax_matches_scipy_for_cycle_samples(self):
        import jax
        for normalization in ('none', 'characteristic_length'):
            with self.subTest(spatial_normalization=normalization):
                config = RhoStarConfig(rho_hat=.3, cycles=2, spatial_normalization=normalization)
                prepared = build_rho_star(field(), config, source_sha256='synthetic')
                scipy = run_rho_star(prepared, executor=Execution(), options=ExecutionOptions())
                with jax.experimental.enable_x64():
                    compiled = run_rho_star(prepared, executor=Execution(), options=ExecutionOptions(backend='jax'))
                np.testing.assert_allclose(compiled.solution.states, scipy.solution.states, rtol=1e-11, atol=1e-13)

    def test_confirmed_receipt_resumes_and_uncertain_submission_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'receipt.json'
            record = {'call_id': 'fc-test', 'app_name': 'gc2d-poincare-rho-sweep', 'function_name': 'integrate'}
            path.write_text(json.dumps(record))
            with patch('studies.poincare_rho_sweep.Execution_Modal.from_record') as resume:
                modal_rho_executor(tmp)
                resume.assert_called_once_with(path)
            record['call_id'] = None
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'Inspect existing Modal receipts'):
                modal_rho_executor(tmp)


if __name__ == '__main__':
    unittest.main()
