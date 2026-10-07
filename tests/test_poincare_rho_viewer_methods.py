"""Method identity and exact rho selection in saved Poincare viewers."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from contracts.execution_options import ExecutionOptions
from execution.execution import Execution
from potential import Grid, Potential
from solution import Solution
from studies.dimensional_h5_midpoint import DimensionalH5Field
from studies.poincare_rho_sweep import RhoStarConfig, build_rho_star, run_rho_star
from visualization.poincare_rho_sweep import export_rho_sweep


class RhoViewerMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Compute inexpensive actual returns for all supported star methods."""
        size, length = 8, .18
        grid = Grid(-length/2, -length/2, length/size, length/size, size, size, length)
        xx, yy = np.meshgrid(np.arange(size)*2*np.pi/size,
                             np.arange(size)*2*np.pi/size, indexing='ij')
        raw = Potential(grid, mean=1e-5*np.cos(xx)*np.cos(yy),
                        modes=np.asarray([1e-6*np.exp(1j*xx)]),
                        frequencies=np.array([1.]), interpolation_order=3)
        field = DimensionalH5Field(raw, raw, 1., 1.5, (15,))
        cls.runs = {}
        for method in ('BM4Midpoint', 'BM4Implicit', 'RK4', 'GaussLegendre4'):
            config = RhoStarConfig(rho_hat=.3, particles=8, cycles=1,
                                   steps_per_cycle=4, method=method)
            prepared = build_rho_star(field, config, source_sha256='synthetic')
            cls.runs[method] = run_rho_star(prepared, executor=Execution(),
                                           options=ExecutionOptions())

    def viewer_config(self, runs, rho_values=(.3,)):
        """Read the actual serialized selector and panel configuration."""
        with tempfile.TemporaryDirectory() as tmp:
            path = export_rho_sweep(runs, Path(tmp)/'viewer.html', rho_values=rho_values,
                                    fold_to_cell=True)
            return json.loads(re.search(r'const cfg=(.*);', path.read_text()).group(1))

    def test_each_method_labels_its_own_returns(self):
        for method, saved in self.runs.items():
            with self.subTest(method=method):
                config = self.viewer_config({.3: saved})
                self.assertEqual(config['panels'][1]['title'], f'{method} · rho = 0.30')
                self.assertIn(f'{method} · 8-particle star', config['title'])
                self.assertEqual(config['firstCycle'], 0)
                self.assertEqual(config['panels'][1]['shape'], [2, 8, 2])

    def test_legacy_midpoint_config_matches_current_defaults(self):
        saved = self.runs['BM4Midpoint']
        metadata = deepcopy(saved.metadata)
        for name in tuple(metadata['config']):
            if name == 'method' or name.startswith('newton_'):
                metadata['config'].pop(name)
        metadata['config']['rho_hat'] = .31
        config = self.viewer_config({.3: saved, .31: replace(saved, metadata=metadata)}, (.3, .31))
        self.assertEqual(len(config['datasets']), 2)

    def test_mixed_methods_and_mislabeled_metadata_are_rejected(self):
        other = self.runs['RK4']
        metadata = deepcopy(other.metadata)
        metadata['config']['rho_hat'] = .31
        with self.assertRaisesRegex(ValueError, 'one method'):
            self.viewer_config({.3: self.runs['BM4Implicit'],
                                .31: replace(other, metadata=metadata)}, (.3, .31))
        metadata['config']['rho_hat'] = .3
        metadata['method'] = 'BM4Implicit'
        with self.assertRaisesRegex(ValueError, 'metadata do not agree'):
            self.viewer_config({.3: replace(other, metadata=metadata)})

    def test_relabeling_implicit_diagnostics_as_rk4_is_rejected(self):
        saved = self.runs['BM4Implicit']
        metadata = deepcopy(saved.metadata)
        metadata['method'] = metadata['config']['method'] = 'RK4'
        with self.assertRaisesRegex(ValueError, 'diagnostics do not match'):
            self.viewer_config({.3: replace(saved, metadata=metadata)})
        saved = self.runs['GaussLegendre4']
        diagnostics = dict(saved.solution.diagnostics)
        diagnostics['stage_count'] = 1
        solution = Solution(t=saved.solution.t, states=saved.solution.states,
                            source=saved.solution.source, diagnostics=diagnostics)
        with self.assertRaisesRegex(ValueError, 'diagnostics do not match'):
            self.viewer_config({.3: replace(saved, solution=solution)})

    def test_fine_rho_values_remain_distinct_and_selectable(self):
        saved = self.runs['RK4']
        metadata = deepcopy(saved.metadata)
        metadata['config']['rho_hat'] = .3001
        rho_values = (.3, .3001, .3002, .3157894736842105)
        config = self.viewer_config({.3: saved, .3001: replace(saved, metadata=metadata)},
                                     rho_values)
        keys = [dataset['key'] for dataset in config['datasets']]
        self.assertEqual(len(set(keys)), len(rho_values))
        self.assertEqual(tuple(float(key) for key in keys), rho_values)
        self.assertEqual(float(config['selectedDataset']), .3)


if __name__ == '__main__':
    unittest.main()
