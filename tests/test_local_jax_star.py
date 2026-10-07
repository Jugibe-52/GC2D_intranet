"""Shared-center geometry, consecutive JAX integration and progress lifecycle."""

from dataclasses import replace
import importlib.util
import io
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from contracts.execution_options import ExecutionOptions
from contracts.request import SimulationRequest
from diagnostics.persistence import load_solution, save_solution
from diagnostics.run_progress import RunProgress, progress_log
from methods.extended.bm4 import BM4Midpoint
from potential.potential import Potential
from simulation.runner import simulate
from studies.local_jax_star import LocalJAXStarConfig, build_local_star, integrate_local_star, calculate_local_star
from visualization.local_jax_star import export_local_star_poincare, local_star_summary


def settings() -> LocalJAXStarConfig:
    """Keep the requested geometry but use a tiny, uneven final test block."""
    return LocalJAXStarConfig(1.5, .06, 3, .3, np.pi / 8,
                             8, 5, True, .5, .7, (.5, .5), 0., 3, 4, 2)


def field() -> Potential:
    base = Potential.random(A=.01, M=3, nx=16, ny=16, seed=27)
    return Potential(base.grid, mean=base.mean, modes=base.modes, frequencies=np.ones(1))


class LocalJAXStarTests(unittest.TestCase):
    def test_exactly_one_shared_center_and_five_radii_per_arm(self):
        potential = field()
        problem = build_local_star(potential, settings())
        x, y = problem.initial_configuration.layout.positions(problem.initial_state)
        relative = np.column_stack((x, y)) / potential.grid.period - .5
        np.testing.assert_array_equal(relative[0], [0., 0.])
        self.assertEqual(np.count_nonzero(np.all(relative == 0., axis=1)), 1)
        self.assertEqual(problem.particle_count, 41)
        radii = np.linalg.norm(relative[1:], axis=1).reshape(8, 5) / .5
        np.testing.assert_allclose(radii, np.tile([.14, .28, .42, .56, .7], (8, 1)))

    def test_invalid_blocks_and_out_of_cell_star_are_rejected(self):
        for change in (dict(block_cycles=0), dict(steps_per_cycle=True), dict(outer_radius_fraction=2.),
                       dict(center_fraction=(.1, .5)), dict(include_center='yes')):
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                build_local_star(field(), replace(settings(), **change))

    def test_progress_flushes_and_propagates_interruption_without_success(self):
        with tempfile.TemporaryDirectory() as temporary, patch('sys.stdout', new_callable=io.StringIO):
            path = Path(temporary) / 'run.log'
            with self.assertRaises(KeyboardInterrupt):
                with progress_log(path, total=3, heartbeat_seconds=.1) as progress:
                    progress.phase('Block complete', completed=2)
                    progress.report()
                    self.assertIn('cycles=2/3', path.read_text())
                    raise KeyboardInterrupt()
            content = path.read_text()
            self.assertIn('FAILED OR INTERRUPTED', content)
            self.assertNotIn('COMPLETED AND SAVED', content)
            self.assertFalse(progress.logger.handlers)

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX is optional')
    def test_blocks_match_complete_run_keep_every_step_and_roundtrip(self):
        import jax
        config, potential = settings(), field()
        progress = RunProgress(logging.getLogger('test.local-star'), config.cycles, 'cycles')
        with jax.experimental.enable_x64():
            result = integrate_local_star(potential, config, progress=progress)
            problem = build_local_star(potential, config)
            request = SimulationRequest((0., 3.), .25, np.arange(13) / 4)
            reference = simulate(problem, BM4Midpoint(), request, options=ExecutionOptions(backend='jax'))
        np.testing.assert_array_equal(result.solution.t, reference.t)
        np.testing.assert_allclose(result.solution.states, reference.states, rtol=1e-13, atol=1e-13)
        for name in ('step_start_times', 'step_times', 'step_sizes', 'copy_separation_norms'):
            np.testing.assert_allclose(result.solution.diagnostics[name], reference.diagnostics[name], atol=1e-13)
        self.assertEqual(result.solution.diagnostics['step_count'], 12)
        np.testing.assert_array_equal(result.solution.diagnostics['execution_block_end_cycles'], [2, 3])
        self.assertEqual(result.metadata['arm_ids'][0], -1)
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / 'archive'
            save_solution(result.solution, archive, potential=potential, metadata=result.metadata)
            saved = load_solution(archive)
            self.assertIn('Particles: 41', local_star_summary(saved))
            path = export_local_star_poincare(saved, Path(temporary) / 'viewer.html', cycles_per_frame=1)
            self.assertIn('Shared center', path.read_text())
            log = Path(temporary) / 'upload.log'
            with patch('studies.local_jax_star.prepare_verified_h5_field', return_value=(potential, {})), \
                 patch('studies.local_jax_star.integrate_local_star', return_value=result), \
                 patch('studies.local_jax_star.save_solution', side_effect=OSError('Upload failed')), \
                 patch('sys.stdout', new_callable=io.StringIO), self.assertRaises(OSError):
                calculate_local_star('unused', config, expected_source_sha256='unused', original_provenance='unused',
                                     destination='unused', log_path=log)
            self.assertIn('Upload failed', log.read_text())
            self.assertNotIn('COMPLETED AND SAVED', log.read_text())


if __name__ == '__main__':
    unittest.main()
