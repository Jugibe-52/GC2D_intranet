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
from diagnostics.persistence import load_solution
from execution.execution import Execution
from potential import Grid, Potential
from solution import Solution
from studies.dimensional_h5_midpoint import DimensionalH5Field
from studies.poincare_rho_sweep import (
    RhoStarConfig, build_rho_star, run_rho_star, run_and_save_rho_star,
    modal_rho_executor, folded_rho_positions,
    sample_rho_dynamics, validate_rho_solution,
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
    @unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX is optional')
    def test_selected_methods_match_scipy_and_keep_cycle_only_archives(self):
        """Exercise each new method through study assembly, JAX and persistence."""
        import jax
        config = RhoStarConfig(rho_hat=.3, cycles=1, steps_per_cycle=4,
                              particles=4, arms=2, hamiltonian_convention='radial')
        with tempfile.TemporaryDirectory() as tmp, jax.experimental.enable_x64():
            for name in ('BM4Implicit', 'RK4', 'GaussLegendre4'):
                with self.subTest(method=name):
                    prepared = build_rho_star(field(), replace(config, method=name), source_sha256='synthetic')
                    scipy = run_rho_star(prepared, executor=Execution(), options=ExecutionOptions())
                    compiled = run_and_save_rho_star(prepared, Path(tmp)/name, executor=Execution(),
                        options=ExecutionOptions(backend='jax'), save_folded_returns=True)
                    loaded = load_solution(Path(tmp)/name)
                    validate_rho_solution(loaded.solution, prepared, options=ExecutionOptions(backend='jax'))
                    np.testing.assert_allclose(compiled.solution.states, scipy.solution.states, rtol=1e-11, atol=1e-13)
                    np.testing.assert_array_equal(loaded.solution.states, compiled.solution.states)
                    self.assertEqual(loaded.metadata['method'], name)
                    self.assertEqual(loaded.solution.states.shape, (8, 2))
                    arrays = {key for key, value in loaded.solution.diagnostics.items() if isinstance(value, np.ndarray)}
                    self.assertEqual(arrays, {'cycle_positions_wrapped', 'cycle_positions_cell_fraction'})
                    if name != 'RK4':
                        self.assertEqual(loaded.solution.diagnostics['nonlinear_solves_per_step'], 1)
                    wrong = replace(prepared, config=replace(prepared.config, newton_absolute_tolerance=1e-8)
                                    if name != 'RK4' else replace(prepared.config, method='GaussLegendre4'))
                    with self.assertRaisesRegex(ValueError, 'implicit-method controls'):
                        validate_rho_solution(loaded.solution, wrong)

    def test_legacy_config_reuse_and_method_mismatch(self):
        """Old midpoint archives remain reusable; another method cannot reuse them."""
        prepared = build_rho_star(field(), RhoStarConfig(rho_hat=.3, cycles=1, steps_per_cycle=4),
                                  source_sha256='synthetic')
        stored = run_rho_star(prepared, executor=Execution(), options=ExecutionOptions())
        for key in tuple(stored.metadata['config']):
            if key == 'method' or key.startswith('newton_'):
                stored.metadata['config'].pop(key)
        stored.metadata['config']['source_selection'] = list(stored.metadata['config']['source_selection'])
        with patch('studies.poincare_rho_sweep.ArtifactStore.exists', return_value=True), \
                patch('studies.poincare_rho_sweep.load_solution', return_value=stored):
            same = run_and_save_rho_star(prepared, 'unused', executor=Execution(), options=ExecutionOptions())
            self.assertIs(same, stored)
            with self.assertRaisesRegex(ValueError, 'different scientific inputs'):
                run_and_save_rho_star(replace(prepared, config=replace(prepared.config, method='RK4')),
                                     'unused', executor=Execution(), options=ExecutionOptions())
        for kwargs in ({'method': 'GaussLegendre2'}, {'newton_max_iterations': 0},
                       {'newton_absolute_tolerance': 0}, {'newton_jacobian_method': 'finite_difference'},
                       {'method': 'RK4', 'coupling_frequency': 1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RhoStarConfig(rho_hat=.3, **kwargs)

    def test_phase_fields_match_gc_dynamics_and_close_at_one_cycle(self):
        """Check signs, gyro-radius dependence and both periodic endpoints."""
        prepared = build_rho_star(field(), RhoStarConfig(rho_hat=.3, cycles=2,
                                  hamiltonian_convention='radial'), source_sha256='synthetic')
        sampled = sample_rho_dynamics(prepared, grid_size=8, vector_grid_size=5)
        self.assertEqual(sampled['potential'].shape, (51, 8, 8))
        self.assertEqual(sampled['velocity'].shape, (51, 5, 5, 2))
        np.testing.assert_array_equal(sampled['velocity'][0], sampled['velocity'][-1])
        np.testing.assert_array_equal(sampled['potential'][0], sampled['potential'][-1])
        x0, y0, length = sampled['bounds']
        q = (np.arange(5) + .5)/5
        xx, yy = np.meshgrid(x0 + q*length, y0 + q*length)
        effective = prepared.problem.dynamics.effective_potential
        expected = np.stack((-effective.evaluate(.24, xx, yy, dy=1),
                              effective.evaluate(.24, xx, yy, dx=1)), axis=-1)
        np.testing.assert_allclose(sampled['velocity'][12], expected)
        zero = build_rho_star(field(), replace(prepared.config, rho_hat=0.), source_sha256='synthetic')
        zero_fields = sample_rho_dynamics(zero, grid_size=8, vector_grid_size=5)
        self.assertGreater(np.max(np.abs(zero_fields['potential']-sampled['potential'])), 0)
        self.assertGreater(np.max(np.abs(sampled['potential'][0]-sampled['potential'][25])), 0)

    def test_radial_hamiltonian_changes_drift_without_changing_forcing_or_geometry(self):
        """Match the radial Hamiltonian in both coordinate systems at fixed time."""
        source = field()
        config = RhoStarConfig(rho_hat=.3, cycles=2)
        cycle = build_rho_star(source, config, source_sha256='synthetic')
        radial = build_rho_star(source, replace(config, hamiltonian_convention='radial'),
                                source_sha256='synthetic')
        normalized = build_rho_star(source, replace(radial.config, spatial_normalization='characteristic_length'),
                                    source_sha256='synthetic')
        np.testing.assert_array_equal(cycle.problem.initial_state, radial.problem.initial_state)
        np.testing.assert_array_equal(cycle.request.output_times, radial.request.output_times)
        np.testing.assert_array_equal(cycle.problem.dynamics.potential.frequencies,
                                      radial.problem.dynamics.potential.frequencies)
        for time in (0., .173, .637):
            velocity = radial.problem.dynamics.vector_field(time, radial.problem.initial_state)
            np.testing.assert_allclose(velocity * (2*np.pi),
                cycle.problem.dynamics.vector_field(time, cycle.problem.initial_state), rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(normalized.problem.dynamics.vector_field(time, normalized.problem.initial_state),
                                      velocity * (2*np.pi/.06), rtol=1e-12, atol=1e-14)
        with tempfile.TemporaryDirectory() as tmp:
            dest = str(Path(tmp)/'radial')
            saved = run_and_save_rho_star(radial, dest, executor=Execution(), options=ExecutionOptions())
            with self.assertRaisesRegex(ValueError, 'different scientific inputs'):
                run_and_save_rho_star(cycle, dest, executor=Execution(), options=ExecutionOptions())
            original = run_rho_star(cycle, executor=Execution(), options=ExecutionOptions())
            original.metadata['config']['rho_hat'] = 0.
            with self.assertRaisesRegex(ValueError, 'same spatial units'):
                export_rho_sweep({0.: original, .3: saved}, Path(tmp)/'mixed.html', rho_values=(0., .3))

    def test_folded_archive_and_viewer_preserve_unwrapped_states(self):
        """Persist both representations and fold multi-cell excursions for display."""
        import base64
        prepared = build_rho_star(field(), RhoStarConfig(rho_hat=.3, cycles=2),
                                  source_sha256='synthetic')
        executor = Execution()
        with tempfile.TemporaryDirectory() as tmp:
            destination = str(Path(tmp) / 'folded')
            saved = run_and_save_rho_star(prepared, destination, executor=executor,
                                          options=ExecutionOptions(), save_folded_returns=True)
            loaded = load_solution(destination)
            np.testing.assert_array_equal(loaded.solution.states, saved.solution.states)
            for name in ('cycle_positions_wrapped', 'cycle_positions_cell_fraction'):
                np.testing.assert_array_equal(loaded.solution.diagnostics[name], saved.solution.diagnostics[name])
            with patch.object(executor, 'run', side_effect=AssertionError('must not resubmit')):
                run_and_save_rho_star(prepared, destination, executor=executor,
                                     options=ExecutionOptions(), save_folded_returns=True)
            # Use known returns on both sides of a nonzero-origin cell, including
            # an exact boundary. Every particle has the same synthetic excursion.
            states = saved.solution.states.copy()
            states[:40, 1:] = np.array([.09, .315])
            states[40:, 1:] = np.array([-.135, -.225])
            synthetic = Solution(t=saved.solution.t, states=states,
                                 source=saved.solution.source, diagnostics=saved.solution.diagnostics)
            wrapped, fractions = folded_rho_positions(synthetic, saved.metadata)
            np.testing.assert_allclose(wrapped[1:, 0], [[-.09, .045], [-.045, -.045]], atol=1e-15)
            np.testing.assert_allclose(fractions[1:, 0], [[0., .75], [.25, .25]], atol=1e-14)
            np.testing.assert_array_equal(synthetic.states, states)
            path = export_rho_sweep({.3: replace(saved, solution=synthetic)}, Path(tmp)/'folded.html',
                                    rho_values=(.3,), fold_to_cell=True)
            cfg = json.loads(re.search(r'const cfg=(.*);', path.read_text()).group(1))
            self.assertEqual(cfg['coordinateBounds'], {'x': 0., 'y': 0., 'span': 1.})
            self.assertEqual(cfg['axisLabels'], ['(R - R0) / L', '(Z - Z0) / L'])
            panel = cfg['panels'][1]
            displayed = np.frombuffer(base64.b64decode(panel['data']), dtype='<f4').reshape(panel['shape'])
            np.testing.assert_allclose(displayed, fractions, atol=1e-7)
            self.assertEqual(cfg['firstCycle'], 0)
            with patch('visualization.poincare_rho_sweep.prepare_rho_star', return_value=prepared):
                export_rho_sweep({.3: saved}, path, rho_values=(.3,), fold_to_cell=True,
                                 source='synthetic.h5', field_grid_size=8, vector_grid_size=5)
            cfg = json.loads(re.search(r'const cfg=(.*);', path.read_text()).group(1))
            encoded = cfg['panels'][0]['field']
            decoded = np.frombuffer(base64.b64decode(encoded['velocity']), dtype='<f4').reshape(encoded['vectorShape'])
            expected = sample_rho_dynamics(prepared, grid_size=8, vector_grid_size=5)
            np.testing.assert_allclose(decoded, expected['velocity']/expected['bounds'][2], rtol=1e-7, atol=1e-12)
            self.assertEqual(encoded['bounds'], {'x': 0., 'y': 0., 'span': 1.})
            self.assertEqual(len(encoded['phases']), 51)

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
            self.assertEqual(cfg['panels'][1]['shape'], [3, 40, 2])
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
            self.assertEqual(cfg['panels'][1]['shape'], [3, 40, 2])
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
