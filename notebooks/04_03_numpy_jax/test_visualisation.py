"""Offline parity and damaged-input checks for the standalone section-4.3 viewer."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare_public_data import prepare_bundle

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
ARCHIVE = ROOT / 'outputs/studies/bm4_numpy_jax/methods-20261119-union16-rho0015-c10-30-90'


class ViewerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='gc2d-43-public-check-')
        cls.bundle = Path(cls.temporary.name) / 'v1'
        cls.pin = prepare_bundle(ARCHIVE, cls.bundle)
        notebook = json.loads((DIRECTORY / 'visualisation.ipynb').read_text())
        cls.namespace = {'__name__': 'viewer_test'}
        first = next(c for c in notebook['cells'] if c['cell_type'] == 'code')
        exec(compile(''.join(first['source']), 'viewer-definitions', 'exec'), cls.namespace)
        cls.namespace['download_bytes'] = lambda name: (cls.bundle / name).read_bytes()
        cls.study, cls.timings, cls.particles, cls.trajectories, cls.manifest = cls.namespace['load_public_data']()
        cls.reference = json.loads((ARCHIVE / 'summary.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_exact_archive_and_figure_parity(self):
        self.assertEqual(self.pin, self.namespace['EXPECTED_MANIFEST_SHA256'])
        summary = self.namespace['summarize_saved'](self.study, self.timings, self.particles, self.trajectories)
        self.assertEqual(summary, self.reference)
        for name in ('comparacion_numpy_jax_fr.json', 'concordancia_numpy_jax_fr.json'):
            original = json.loads((ROOT.parent / 'TFM_redaccion/figuras' / name).read_text())
            self.assertEqual(summary['rows'], original['rows'])

    def test_rejects_download_corruption(self):
        original = self.namespace['download_bytes']
        try:
            for damaged in ('manifest.json', 'trajectories.npz', 'timings.json'):
                with self.subTest(damaged=damaged):
                    self.namespace['download_bytes'] = lambda name: original(name) + (b' ' if name == damaged else b'')
                    with self.assertRaises(ValueError):
                        self.namespace['load_public_data']()
        finally:
            self.namespace['download_bytes'] = original

    def test_rejects_duplicate_wrong_phase_cache_and_nonfinite_timings(self):
        for mode in ('duplicate', 'phase', 'cache', 'nonfinite'):
            changed = copy.deepcopy(self.timings)
            rows = changed['BM4Implicit-10']['pair']['measurements']
            if mode == 'duplicate': rows[1] = copy.deepcopy(rows[0])
            if mode == 'phase': rows[0]['phase'] = 'warm'
            if mode == 'cache': rows[1]['integration_cache_misses'] = 0
            if mode == 'nonfinite': rows[0]['seconds'] = float('nan')
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.namespace['summarize_saved'](self.study, changed, self.particles, self.trajectories)

    def test_rejects_changed_states_times_and_membership(self):
        for mode in ('states', 'times', 'membership'):
            arrays = {name: value.copy() for name, value in self.trajectories.items()}
            groups = copy.deepcopy(self.particles)
            if mode == 'states': arrays['BM4Implicit-10/scipy/states'][1, 0] += .1
            if mode == 'times': arrays['BM4Implicit-10/scipy/t'][1] += .01
            if mode == 'membership': groups['chaotic']['record']['particles'][0]['id'] = 999
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.namespace['summarize_saved'](self.study, self.timings, groups, arrays)

    def test_bundle_rebuild_is_immutable(self):
        self.assertEqual(prepare_bundle(ARCHIVE, self.bundle), self.pin)
        with tempfile.TemporaryDirectory() as temporary:
            occupied = Path(temporary)
            (occupied / 'unrelated.txt').write_text('Existing data must survive.')
            with self.assertRaises(FileExistsError):
                prepare_bundle(ARCHIVE, occupied)
            self.assertEqual((occupied / 'unrelated.txt').read_text(), 'Existing data must survive.')


if __name__ == '__main__':
    unittest.main(verbosity=2)
