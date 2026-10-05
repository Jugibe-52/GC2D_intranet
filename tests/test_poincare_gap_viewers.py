"""Saved gap probes extend viewer data without changing the original star."""

import base64
from copy import deepcopy
from dataclasses import replace
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
from studies.poincare_gap_probes import GapProbeSeeds, build_gap_probes
from studies.poincare_rho_sweep import RhoStarConfig, build_rho_star, folded_rho_positions, run_rho_star
from visualization.poincare_method_comparison import export_rho_method_comparison
from visualization.poincare_rho_sweep import export_rho_sweep, radius_colors


class GapProbeViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Compute only two cycles of a smooth test field for eight small pairs."""
        size, length = 8, .18
        grid = Grid(-length/2, -length/2, length/size, length/size, size, size, length)
        xx, yy = np.meshgrid(np.arange(size)*2*np.pi/size,
                             np.arange(size)*2*np.pi/size, indexing='ij')
        potential = Potential(grid, mean=1e-5*np.cos(xx)*np.cos(yy),
                              modes=np.asarray([1e-6*np.exp(1j*xx)]),
                              frequencies=np.array([1.]), interpolation_order=3)
        cls.field = DimensionalH5Field(potential, potential, 1., 1.5, (0, 1))
        cls.seeds = GapProbeSeeds(
            fractions=((.21, .24), (.23, .26), (.71, .24), (.73, .26),
                       (.21, .74), (.23, .76), (.71, .74), (.73, .76)),
            particle_ids=tuple(range(41, 49)),
            gap_names=('Lower left',)*2 + ('Lower right',)*2 + ('Upper left',)*2 + ('Upper right',)*2,
            selection_rho=.3)
        cls.runs, cls.probes = {}, {}
        for method in ('BM4Midpoint', 'BM4Implicit', 'RK4', 'GaussLegendre4'):
            cls.runs[method], cls.probes[method] = {}, {}
            for rho in (.3, .4):
                config = RhoStarConfig(rho_hat=rho, cycles=2, steps_per_cycle=4, method=method)
                original = cls.prepare('synthetic.h5', config)
                probe = build_gap_probes(cls.field, replace(config, particles=8, arms=8),
                                         source_sha256='synthetic', seeds=cls.seeds)
                cls.runs[method][rho] = run_rho_star(original, executor=Execution(), options=ExecutionOptions())
                cls.probes[method][rho] = run_rho_star(probe, executor=Execution(), options=ExecutionOptions())

    @classmethod
    def prepare(cls, source, config):
        """Reconstruct the same test field without involving an HDF5 file."""
        return build_rho_star(cls.field, config, source_sha256='synthetic')

    def export(self, probes=None, *, single=False, fold_to_cell=True):
        """Read serialized data from the actual HTML export."""
        probes = self.probes if probes is None else probes
        with tempfile.TemporaryDirectory() as tmp, patch(
                'visualization.poincare_method_comparison.prepare_rho_star', side_effect=self.prepare):
            path = Path(tmp)/'viewer.html'
            if single:
                export_rho_sweep(self.runs['RK4'], path, rho_values=(.3, .4),
                                 probe_runs=probes['RK4'], fold_to_cell=fold_to_cell,
                                 colormap='turbo', color_range=(.025, .975))
            else:
                export_rho_method_comparison(
                    self.runs, path, source='synthetic.h5', rho_values=(.3, .4),
                    probe_runs_by_method=probes, field_grid_size=4, vector_grid_size=3)
            return json.loads(re.search(r'const cfg=(.*);', path.read_text()).group(1))

    def test_four_methods_keep_all_40_original_colors_and_append_eight_saved_particles(self):
        config = self.export()
        self.assertEqual(config['fieldPlacement'], 'below_particles')
        self.assertIn('40-particle star + 8 gap probes', config['title'])
        self.assertIn('rho = 0.30', config['description'])
        for dataset, rho in zip(config['datasets'], (.3, .4)):
            for panel, method in zip(dataset['panels'][:4], self.runs):
                self.assertEqual(panel['shape'], [3, 48, 2])
                self.assertEqual(panel['ids'], list(range(1, 49)))
                self.assertEqual(panel['colors'][:40], radius_colors(40, colormap='turbo', color_range=(.025, .975)))
                self.assertEqual(panel['colors'][40:], config['panels'][0]['colors'][40:])
                xy = np.frombuffer(base64.b64decode(panel['data']), dtype='<f4').reshape(panel['shape'])
                original = self.runs[method][rho]
                probe = self.probes[method][rho]
                _, base_xy = folded_rho_positions(original.solution, original.metadata)
                _, probe_xy = folded_rho_positions(probe.solution, probe.metadata)
                np.testing.assert_array_equal(xy[:, :40], base_xy.astype('<f4'))
                np.testing.assert_array_equal(xy[:, 40:], probe_xy.astype('<f4'))
                self.assertEqual(original.solution.states.shape, (80, 3))
            self.assertEqual(dataset['panels'][-1]['shape'], [1, 48, 2])
        for arm, group in enumerate(config['particleGroups'][:8], 1):
            self.assertEqual(group['label'], f'Arm {arm}')
            self.assertEqual(group['ids'], list(range(arm, 41, 8)))
        self.assertEqual(config['particleGroups'][8]['ids'], list(range(41, 49)))
        self.assertEqual([group['ids'] for group in config['particleGroups'][9:]],
                         [[41, 42], [43, 44], [45, 46], [47, 48]])
        self.assertIn('Lower left', config['particleLabels']['41'])

    def test_single_method_keeps_raw_and_folded_coordinate_conventions(self):
        for folded in (False, True):
            config = self.export(single=True, fold_to_cell=folded)
            panel = config['panels'][1]
            self.assertEqual(panel['shape'], [3, 48, 2])
            self.assertEqual(panel['colors'][:40], radius_colors(40, colormap='turbo', color_range=(.025, .975)))
            xy = np.frombuffer(base64.b64decode(panel['data']), dtype='<f4').reshape(panel['shape'])
            if folded:
                expected = np.asarray(self.seeds.fractions)
            else:
                expected = self.probes['RK4'][.3].metadata['initial_positions']
            np.testing.assert_allclose(xy[0, 40:], expected, rtol=1e-6, atol=1e-9)

    def test_mismatched_source_controls_or_initial_seeds_are_rejected(self):
        for mutate in (
            lambda record: record.update(source_sha256='other'),
            lambda record: record['config'].update(steps_per_cycle=8),
            lambda record: record['probe_config']['fractions'][0].__setitem__(0, .9),
        ):
            probes = {method: dict(runs) for method, runs in self.probes.items()}
            saved = probes['RK4'][.4]
            metadata = deepcopy(saved.metadata)
            # Match the JSON-loaded representation used by real saved archives.
            metadata['probe_config'] = json.loads(json.dumps(metadata['probe_config']))
            mutate(metadata)
            probes['RK4'][.4] = replace(saved, metadata=metadata)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.export(probes)

    def test_incomplete_probe_grid_is_rejected(self):
        probes = {method: dict(runs) for method, runs in self.probes.items()}
        probes['RK4'].pop(.4)
        with self.assertRaisesRegex(ValueError, 'exactly the available'):
            self.export(probes)

    def test_probe_region_contract_is_frozen_across_methods_and_rho(self):
        probes = {method: dict(runs) for method, runs in self.probes.items()}
        saved = probes['GaussLegendre4'][.4]
        metadata = deepcopy(saved.metadata)
        metadata['probe_config']['gap_names'][0] = 'Different region'
        metadata['gap_names'][0] = 'Different region'
        probes['GaussLegendre4'][.4] = replace(saved, metadata=metadata)
        with self.assertRaisesRegex(ValueError, 'must remain fixed'):
            self.export(probes)


if __name__ == '__main__':
    unittest.main()
