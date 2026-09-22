"""Probe integration, integrity and viewer alignment checks on short records."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from diagnostics.poincare_probe import load_rk4_probe, save_rk4_probe
from potential import Grid, Potential
from studies.poincare_probe import (
    RK4ProbeSettings, integrate_rk4_probe, probe_section, probe_comparison_panel,
)
from visualization.poincare_comparison import export_poincare_panel_comparison


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.settings = RK4ProbeSettings(49, (.25, .05), '#d000d0', 3, 20, 40, 0.)

    def test_linear_gc_drift_keeps_every_step_across_chunks_and_wraps_only_for_display(self):
        # Phi=cos(x) gives constant x and y'=-sin(x). At x=pi/2 it crosses y=0.
        grid = Grid.periodic(128, 16)
        potential = Potential(grid, mean=np.cos(grid.x)[:, None] * np.ones(grid.shape))
        result = integrate_rk4_probe(potential, self.settings)
        self.assertEqual(result['step_count'], 60)
        self.assertEqual(result['xy'].shape, (61, 1, 2))
        np.testing.assert_allclose(result['xy'][:, 0, 0], np.pi / 2, atol=1e-14)
        np.testing.assert_allclose(result['xy'][:, 0, 1], .05*2*np.pi - result['times'], atol=1e-6)
        section = probe_section(result, self.settings, grid.period)
        np.testing.assert_allclose(section[:, 0, 1], (.05 - np.arange(4)/(2*np.pi)) % 1, atol=2e-7)
        self.assertEqual(section.shape, (4, 1, 2))

    def test_storage_rejects_changed_contract_and_corrupted_trajectory(self):
        result = dict(times=np.arange(3, dtype=float), xy=np.zeros((3, 1, 2)),
                      step_count=2, runtime_seconds=.1)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_rk4_probe(directory, result, contract={'cycles': 2})
            loaded, _ = load_rk4_probe(directory, expected_contract={'cycles': 2})
            np.testing.assert_array_equal(loaded['xy'], result['xy'])
            with self.assertRaisesRegex(ValueError, 'different settings'):
                load_rk4_probe(directory, expected_contract={'cycles': 3})
            with self.assertRaises(FileExistsError):
                save_rk4_probe(directory, result, contract={'cycles': 2})
            with (directory/'trajectory.npz').open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                load_rk4_probe(directory)

    def test_panel_keeps_background_ids_colors_and_cycle_alignment(self):
        background = SimpleNamespace(positions=np.arange(16).reshape(4,2,2)/20,
                                     particle_ids=(1, 3), colors=('#111111', '#333333'))
        probe = np.full((4,1,2), .4)
        panel = probe_comparison_panel(background, probe, self.settings)
        np.testing.assert_array_equal(panel['coordinates'][:, :2], background.positions[1:])
        self.assertEqual(panel['particle_ids'], [1,3,49])
        self.assertEqual(panel['colors'], ['#111111','#333333','#d000d0'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'viewer.html'
            export_poincare_panel_comparison(path,[panel],initial_view=(.2,.3,.1),
                                            initial_cycle=3,highlight_particle=49,title='RK4 </script>')
            config = json.loads(path.read_text().split('const cfg=')[1].split(';\n')[0])
            self.assertEqual(config['initialCycle'],3)
            self.assertEqual(config['highlightParticle'],49)
            self.assertEqual(config['panels'][0]['shape'],[3,3,2])
            export_poincare_panel_comparison(path,[panel])
            for options in ({'initial_view':(.9,0,.2)}, {'initial_cycle':4}, {'highlight_particle':50}):
                with self.assertRaises(ValueError):
                    export_poincare_panel_comparison(path,[panel],**options)

    def test_invalid_geometry_and_cycle_controls(self):
        for options in ({'initial_xy_over_L':(np.nan,.4)}, {'cycles':0},
                        {'chunk_steps':21}, {'steps_per_cycle':True}):
            with self.assertRaises(ValueError):
                replace(self.settings, **options)


if __name__ == '__main__':
    unittest.main()
