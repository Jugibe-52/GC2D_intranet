"""Exercise corruption handling and the scientific missing-case/plot contracts."""
from __future__ import annotations

import base64
import copy
import json
import os
from pathlib import Path
import unittest
import zlib

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np


class ProjectionViewerTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		notebook = json.loads(Path(__file__).with_name('visualisation.ipynb').read_text())
		code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
		cls.namespace = {'__name__': 'viewer_tests'}
		exec(compile(code[0], 'viewer-definitions', 'exec'), cls.namespace)
		exec(compile(code[2].split('\ncorrection_figure =', 1)[0], 'viewer-plot-definitions', 'exec'), cls.namespace)
		cls.summary, cls.provenance, cls.manifest = cls.namespace['load_data']()
		cls.objects = {name: base64.b64decode(value) for name, value in json.loads(
			zlib.decompress(base64.b64decode(cls.namespace['EMBEDDED_SNAPSHOT_B64']))).items()}

	def tearDown(self) -> None:
		self.namespace['DATA_SOURCE'] = 'snapshot'
		plt.close('all')

	def test_reference_series_and_explicit_missing_case(self) -> None:
		series = self.namespace['verify_summary'](self.summary, self.provenance)
		for statistic in self.namespace['STATISTICS']:
			self.assertTrue(np.isnan(series['ABBA2Implicit'][statistic][0]))
			self.assertEqual(np.count_nonzero(np.isfinite(series['ABBA2Implicit'][statistic])), 10)
			self.assertTrue(np.isfinite(series['BM4Implicit'][statistic]).all())
		self.assertEqual(len(self.provenance['cases']), 21)
		self.assertEqual(sum(len(case['arrays']) for case in self.provenance['cases']), 147)
		self.assertEqual(self.provenance['archive_solution_bytes'], 6108791916)

	def test_tampered_manifest_rejected_without_snapshot_fallback(self) -> None:
		objects = dict(self.objects, **{'manifest.json': self.objects['manifest.json'] + b' '})
		self.namespace['DATA_SOURCE'] = 'public_https'
		self.namespace['download_bytes'] = objects.__getitem__
		with self.assertRaisesRegex(ValueError, 'pinned version'):
			self.namespace['load_data']()

	def test_tampered_summary_rejected_without_snapshot_fallback(self) -> None:
		objects = dict(self.objects, **{'summary.json': self.objects['summary.json'].replace(b'5000', b'5001', 1)})
		self.namespace['DATA_SOURCE'] = 'public_https'
		self.namespace['download_bytes'] = objects.__getitem__
		with self.assertRaisesRegex(ValueError, 'Size or SHA-256 mismatch'):
			self.namespace['load_data']()

	def test_transport_failure_propagates(self) -> None:
		def fail(name: str) -> bytes:
			raise RuntimeError('Unavailable public object')
		self.namespace['DATA_SOURCE'] = 'public_https'
		self.namespace['download_bytes'] = fail
		with self.assertRaisesRegex(RuntimeError, 'Unavailable public object'):
			self.namespace['load_data']()

	def test_zero_filling_the_unavailable_case_rejected(self) -> None:
		summary = copy.deepcopy(self.summary)
		case = summary['cases'][0]
		case.update(status='complete', **{name: 0 for name in self.namespace['STATISTICS']})
		with self.assertRaisesRegex(ValueError, 'unavailable case'):
			self.namespace['verify_summary'](summary, self.provenance)

	def test_additional_radius_rescaling_rejected(self) -> None:
		summary = copy.deepcopy(self.summary)
		summary['rho_values'] = [rho / 20 for rho in summary['rho_values']]
		with self.assertRaisesRegex(ValueError, 'rho_values'):
			self.namespace['verify_summary'](summary, self.provenance)

	def test_changed_statistic_rejected_against_thesis_series(self) -> None:
		summary = copy.deepcopy(self.summary)
		summary['cases'][1]['sampled_median'] *= 2
		with self.assertRaises(AssertionError):
			self.namespace['verify_summary'](summary, self.provenance)

	def test_four_panels_preserve_scales_sampled_points_and_canvas(self) -> None:
		series = self.namespace['verify_summary'](self.summary, self.provenance)
		for name, expected_scale in (('correction', 'log'), ('newton', 'linear')):
			figure = self.namespace['plot_comparison'](self.summary, series, name)
			self.assertEqual(len(figure.axes), 2)
			for axis in figure.axes:
				self.assertEqual(axis.get_yscale(), expected_scale)
				self.assertEqual(len(axis.lines), 3)
				np.testing.assert_array_equal(axis.lines[0].get_xdata(), np.arange(11) / 400)
				self.assertTrue(np.isnan(axis.lines[0].get_ydata()[0]))
				self.assertIn('Rayon normalisé', axis.get_xlabel())
			renderer = figure.canvas.get_renderer()
			boxes = [axis.get_tightbbox(renderer) for axis in figure.axes]
			boxes += [item.get_window_extent(renderer) for item in (*figure.legends, *figure.texts)]
			for box in boxes:
				self.assertGreaterEqual(box.x0, 0)
				self.assertGreaterEqual(box.y0, 0)
				self.assertLessEqual(box.x1, figure.bbox.width)
				self.assertLessEqual(box.y1, figure.bbox.height)
			self.assertLess(boxes[0].x1, boxes[1].x0)
			self.assertGreater(boxes[2].y0, max(box.y1 for box in boxes[:2]))
			self.assertLess(boxes[3].y1, min(box.y0 for box in boxes[:2]))


if __name__ == '__main__':
	unittest.main()
