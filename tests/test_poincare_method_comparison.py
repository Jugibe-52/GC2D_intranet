"""Scientific comparability and data retention in the four-method rho viewer."""

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
from studies.poincare_rho_sweep import (
    RhoStarConfig, build_rho_star, folded_rho_positions, run_rho_star, sample_rho_dynamics,
)
from visualization.poincare_method_comparison import export_rho_method_comparison


class PoincareMethodComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Save two tiny actual runs per method with the same periodic field."""
        size, length = 8, .18
        grid = Grid(-length/2, -length/2, length/size, length/size, size, size, length)
        xx, yy = np.meshgrid(np.arange(size)*2*np.pi/size,
                             np.arange(size)*2*np.pi/size, indexing='ij')
        potential = Potential(grid, mean=1e-5*np.cos(xx)*np.cos(yy),
                              modes=np.asarray([1e-6*np.exp(1j*xx)]),
                              frequencies=np.array([1.]), interpolation_order=3)
        cls.field = DimensionalH5Field(potential, potential, 1., 1.5, (15,))
        cls.runs = {}
        for method in ('BM4Midpoint', 'BM4Implicit', 'RK4', 'GaussLegendre4'):
            cls.runs[method] = {}
            for rho in (.3, .4):
                prepared = cls.prepare('synthetic.h5', RhoStarConfig(
                    rho_hat=rho, particles=8, cycles=2, steps_per_cycle=4, method=method))
                cls.runs[method][rho] = run_rho_star(
                    prepared, executor=Execution(), options=ExecutionOptions())

    @classmethod
    def prepare(cls, source, config):
        """Supply a deterministic original field in place of an on-disk HDF5."""
        return build_rho_star(cls.field, config, source_sha256='synthetic')

    def export(self, runs=None):
        """Decode the HTML payload while recording the actual field sampling."""
        module = 'visualization.poincare_method_comparison'
        with tempfile.TemporaryDirectory() as tmp, \
                patch(f'{module}.prepare_rho_star', side_effect=self.prepare), \
                patch(f'{module}.sample_rho_dynamics', wraps=sample_rho_dynamics) as sample:
            path = export_rho_method_comparison(
                self.runs if runs is None else runs, Path(tmp)/'viewer.html',
                source='synthetic.h5', rho_values=(.3, .4, .5),
                field_grid_size=4, vector_grid_size=3)
            config = json.loads(re.search(r'const cfg=(.*);', path.read_text()).group(1))
            return config, sample.call_count

    def changed_metadata(self, method, rho, update):
        """Copy one metadata record without changing the shared test archives."""
        runs = {name: dict(values) for name, values in self.runs.items()}
        saved = runs[method][rho]
        metadata = deepcopy(saved.metadata)
        update(metadata)
        runs[method][rho] = replace(saved, metadata=metadata)
        return runs

    def test_four_panels_preserve_all_cycles_and_share_one_field_per_rho(self):
        config, sample_count = self.export()
        self.assertEqual(sample_count, 2)
        self.assertEqual(config['fieldPlacement'], 'below_particles')
        self.assertEqual(config['firstCycle'], 0)
        self.assertEqual(config['initialCycle'], 2)
        self.assertEqual([panel['title'] for panel in config['panels'][:4]], list(self.runs))
        self.assertEqual([item['key'] for item in config['datasets']], ['0.30', '0.40', '0.50'])
        self.assertNotIn('panels', config['datasets'][-1])
        for dataset, rho in zip(config['datasets'], (.3, .4)):
            self.assertEqual(len(dataset['panels']), 5)
            for panel, method in zip(dataset['panels'][:4], self.runs):
                self.assertEqual(panel['shape'], [3, 8, 2])
                self.assertEqual(panel['ids'], config['panels'][0]['ids'])
                self.assertEqual(panel['colors'], config['panels'][0]['colors'])
                self.assertNotIn('field', panel)
                encoded = np.frombuffer(base64.b64decode(panel['data']), dtype='<f4').reshape(panel['shape'])
                saved = self.runs[method][rho]
                _, expected = folded_rho_positions(saved.solution, saved.metadata)
                np.testing.assert_array_equal(encoded, expected.astype('<f4'))
            field_panel = dataset['panels'][-1]
            self.assertTrue(field_panel['static'])
            field = field_panel['field']
            self.assertEqual(field['shape'], [51, 4, 4])
            self.assertEqual(field['vectorShape'], [51, 3, 3, 2])
            self.assertEqual(field['bounds'], {'x': 0., 'y': 0., 'span': 1.})
            potential = np.frombuffer(base64.b64decode(field['potential']), dtype='<f4').reshape(field['shape'])
            velocity = np.frombuffer(base64.b64decode(field['velocity']), dtype='<f4').reshape(field['vectorShape'])
            np.testing.assert_array_equal(potential[0], potential[-1])
            np.testing.assert_array_equal(velocity[0], velocity[-1])
            prepared = self.prepare('synthetic.h5', RhoStarConfig(**self.runs['RK4'][rho].metadata['config']))
            sampled = sample_rho_dynamics(prepared, grid_size=4, vector_grid_size=3)
            length = prepared.metadata['cell_bounds'][2]
            np.testing.assert_array_equal(potential, (sampled['potential']/length**2).astype('<f4'))
            np.testing.assert_array_equal(velocity, (sampled['velocity']/length).astype('<f4'))

    def test_different_physical_settings_and_metadata_are_rejected(self):
        for mutate in (
            lambda metadata: metadata['config'].update(magnetic_field=2.),
            lambda metadata: metadata.update(time_unit_seconds=2.),
            lambda metadata: metadata.update(source_sha256='different'),
            lambda metadata: metadata['initial_positions_m'].__setitem__((0, 0), .01),
        ):
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, 'scientific|physical|geometry'):
                self.export(self.changed_metadata('RK4', .4, mutate))

    def test_implicit_controls_match_but_unused_explicit_controls_may_differ(self):
        mutate = lambda metadata: metadata['config'].update(newton_absolute_tolerance=1e-10)
        with self.assertRaisesRegex(ValueError, 'Newton controls'):
            self.export(self.changed_metadata('GaussLegendre4', .4, mutate))
        config, _ = self.export(self.changed_metadata('RK4', .4, mutate))
        self.assertEqual(len(config['panels']), 5)

    def test_incomplete_method_rho_grid_is_rejected(self):
        runs = {name: dict(values) for name, values in self.runs.items()}
        runs['GaussLegendre4'].pop(.4)
        with self.assertRaisesRegex(ValueError, 'share a nonempty set'):
            self.export(runs)


if __name__ == '__main__':
    unittest.main()
