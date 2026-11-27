"""Parity, scientific invariants and corruption checks for the section-4.4 viewer."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np

from prepare_public_data import ARCHIVE, ROOT, RUN_ID, prepare_bundle

DIRECTORY = Path(__file__).resolve().parent
BUNDLE = ROOT / 'outputs/colab-public/04_04_evolution_energetique' / RUN_ID / 'v1'


class ViewerChecks(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.notebook = json.loads((DIRECTORY / 'visualisation.ipynb').read_text())
		cls.namespace = {'__name__': 'viewer_test'}
		first = next(cell for cell in cls.notebook['cells'] if cell['cell_type'] == 'code')
		exec(compile(''.join(first['source']), 'viewer-definitions', 'exec'), cls.namespace)
		cls.namespace['download_bytes'] = lambda name: (BUNDLE / name).read_bytes()
		cls.study, cls.arrays, cls.manifest = cls.namespace['load_public_data']()
		cls.reference = json.loads((ROOT.parent / 'TFM_redaccion/figuras/balance_energetico_en_linea_fr.json').read_text())

	def test_exact_retained_figure_statistics(self):
		summary = self.namespace['summarize_saved'](self.study, self.arrays)
		self.assertEqual(summary['series'], self.reference['series'])
		self.assertEqual(summary['global_metrics'], self.reference['global_metrics'])
		self.assertEqual(len(summary['series']), 32)
		self.assertEqual(len(summary['global_metrics']), 16)

	def test_rejects_download_corruption(self):
		original = self.namespace['download_bytes']
		try:
			for damaged in ('manifest.json', 'study.json', 'diagnostics.npz'):
				self.namespace['download_bytes'] = lambda name: original(name) + (b' ' if name == damaged else b'')
				with self.subTest(damaged=damaged), self.assertRaises(ValueError):
					self.namespace['load_public_data']()
		finally:
			self.namespace['download_bytes'] = original

	def test_rejects_inconsistent_diagnostics_cycles_and_ids(self):
		for mode in ('residual', 'integral_coarse', 'hamiltonian', 'nonfinite', 'cycle', 'particle_ids'):
			arrays = self.arrays.copy()
			key = mode if mode in ('cycle', 'particle_ids') else 'RK2/' + ('residual' if mode == 'nonfinite' else mode)
			arrays[key] = arrays[key].copy()
			if arrays[key].ndim == 1:
				arrays[key][1] += 1
			else:
				arrays[key][1, 0] += np.nan if mode == 'nonfinite' else .01
			with self.subTest(mode=mode), self.assertRaises((ValueError, AssertionError)):
				self.namespace['summarize_saved'](self.study, arrays)

	def test_rejects_wrong_groups_and_protocol(self):
		for mode in ('groups', 'rho', 'steps_per_cycle', 'integrated_particle_count'):
			study = copy.deepcopy(self.study)
			if mode == 'groups': study[mode]['chaotic8'][0] = 1
			else: study[mode] *= 2
			with self.subTest(mode=mode), self.assertRaises(ValueError):
				self.namespace['summarize_saved'](study, self.arrays)

	def test_rejects_self_consistent_but_different_scientific_values(self):
		arrays = self.arrays.copy()
		for key in ('hamiltonian', 'delta_h', 'residual'):
			name = 'RK2/' + key
			arrays[name] = arrays[name].copy()
			arrays[name][1, 0] += .01
		# Recompute identities so only independent thesis-statistics verification catches this.
		arrays['RK2/delta_h'] = arrays['RK2/hamiltonian'] - arrays['RK2/hamiltonian'][0]
		arrays['RK2/residual'] = arrays['RK2/delta_h'] - arrays['RK2/integral_simpson']
		with self.assertRaises(ValueError):
			self.namespace['summarize_saved'](self.study, arrays)

	def test_four_panels_and_source_parity(self):
		namespace = self.namespace.copy()
		namespace['summary'] = namespace['summarize_saved'](self.study, self.arrays)
		cell = next(cell for cell in self.notebook['cells'] if cell['cell_type'] == 'code' and 'def plot_balance' in ''.join(cell['source']))
		exec(compile(''.join(cell['source']), 'plot-cell', 'exec'), namespace)
		figure = namespace['figure']
		self.assertEqual(len(figure.axes), 4)
		for index, axis in enumerate(figure.axes):
			self.assertEqual(axis.get_yscale(), 'log')
			self.assertEqual(len(axis.lines), 8)
			for line, row in zip(axis.lines, self.reference['series'][index * 8:(index + 1) * 8]):
				np.testing.assert_array_equal(line.get_ydata(), row['values'])
				np.testing.assert_array_equal(line.get_xdata(), row['cycle_centers'])
		self.assertEqual(figure.axes[0].get_ylim(), figure.axes[1].get_ylim())
		self.assertEqual(figure.axes[2].get_ylim(), figure.axes[3].get_ylim())
		namespace['plt'].close(figure)

	def test_bundle_rebuild_is_immutable(self):
		self.assertEqual(prepare_bundle(ARCHIVE, BUNDLE), self.namespace['EXPECTED_MANIFEST_SHA256'])
		with tempfile.TemporaryDirectory() as temporary:
			occupied = Path(temporary)
			(occupied / 'unrelated.txt').write_text('Existing data must survive.')
			with self.assertRaises(FileExistsError):
				prepare_bundle(ARCHIVE, occupied)
			self.assertEqual((occupied / 'unrelated.txt').read_text(), 'Existing data must survive.')


if __name__ == '__main__':
	unittest.main(verbosity=2)
