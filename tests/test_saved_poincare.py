"""Check cycle-only viewing, particle ordering and saved-product integrity."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from diagnostics.poincare_section import load_poincare_cycle_csv
from visualization.saved_poincare import render_saved_section


class SavedPoincareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'resultados' / 'sample'
        self.run.mkdir(parents=True)
        metadata = dict(run_id='sample', particle_ids=[7, 2], particle_count=2,
                        cycles=2, method='RK4', steps_per_cycle=20)
        (self.run / 'metadata.json').write_text(json.dumps(metadata))
        self.csv = self.run / 'positions_after_each_cycle.csv'
        self.csv.write_text('cycle,particle,color,x_over_L,y_over_L\n'
                            '2,2,#123456,0.2,0.3\n1,7,#654321,0.4,0.5\n'
                            '1,2,#123456,0.6,0.7\n2,7,#654321,0.8,0.9\n')
        self.publish()

    def publish(self):
        hashes = {name: hashlib.sha256((self.run / name).read_bytes()).hexdigest()
                  for name in ('metadata.json', 'positions_after_each_cycle.csv')}
        (self.run / 'COMPLETE.json').write_text(json.dumps(
            dict(schema_version=1, run_id='sample', sha256=hashes)))

    def test_render_without_trajectory_or_initial_files(self):
        metadata, points, colors = load_poincare_cycle_csv(self.root, 'sample')
        self.assertEqual(metadata['particle_ids'], [7, 2])
        self.assertEqual(colors, ['#654321', '#123456'])
        np.testing.assert_array_equal(points[:, 0], [[.4, .5], [.8, .9]])
        selector = render_saved_section(self.root, 'sample')
        self.assertIn('"ids": [7, 2]', selector.read_text())
        self.assertTrue(selector.with_name('poincare_section.png').is_file())

    def test_reject_modified_csv(self):
        self.csv.write_text(self.csv.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            load_poincare_cycle_csv(self.root, 'sample')

    def test_reject_missing_or_duplicate_returns(self):
        original = self.csv.read_text()
        for text in (original.rsplit('\n', 2)[0] + '\n',
                     original + '2,7,#654321,0.8,0.9\n'):
            with self.subTest(text=text):
                self.csv.write_text(text)
                self.publish()
                with self.assertRaisesRegex(ValueError, 'Incomplete|Duplicate'):
                    load_poincare_cycle_csv(self.root, 'sample')
