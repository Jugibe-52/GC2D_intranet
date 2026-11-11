"""Identity and numerical consistency of full-integration viewer annotations."""

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from visualization.poincare_comparison import export_poincare_panel_comparison
from visualization.poincare_transport import (
	attach_horizontal_transport, export_horizontal_transport_viewer,
)


METHODS = ('BM4Midpoint', 'BM4Implicit', 'RK4', 'GaussLegendre4')


class HorizontalTransportViewerTests(unittest.TestCase):
	def setUp(self):
		self.positions = np.array([[[.2, .3], [.4, .5]],
			[[.25, .35], [.45, .55]], [[.3, .4], [.5, .6]]])
		self.panels = [dict(title=method, coordinates=self.positions,
			particle_ids=[1, 2], colors=['blue', 'red']) for method in METHODS]
		self.datasets = [dict(key=f'{rho:.2f}', label=f'{rho:.2f}', panels=self.panels)
			for rho in (.3, .4)]
		self.results = dict(schema_version=1, metric=dict(
			coordinate='physical_unwrapped_x', integral='accepted_endpoint_trapezoid_abs_vx', units='m'),
			cases=[dict(rho=rho, method=method, particle_ids=[2, 1],
				q_x=[None, 1.5], travel_x=[0., 3.], width_x=[0., 1.],
				x_min=[.5, -.5], x_max=[.5, .5], cycles=2, steps_per_cycle=50)
				for rho in (.3, .4) for method in METHODS])

	def export(self, path, results=None):
		return export_poincare_panel_comparison(path, self.panels,
			first_cycle=0, initial_cycle=2, datasets=self.datasets,
			selected_dataset='0.30', horizontal_transport=results)

	def read(self, path):
		return json.loads(re.search(r'^const cfg=(.*);$', path.read_text(), re.MULTILINE).group(1))

	def test_optional_metrics_match_by_id_and_dataset_without_altering_trajectories(self):
		with tempfile.TemporaryDirectory() as temporary:
			plain = self.read(self.export(Path(temporary) / 'plain.html'))
			annotated = self.read(self.export(Path(temporary) / 'annotated.html', self.results))
		self.assertNotIn('horizontalTransportMetric', plain)
		self.assertNotIn('horizontalTransport', plain['datasets'][0])
		for before, after in zip(plain['datasets'], annotated['datasets']):
			self.assertEqual(before['panels'], after['panels'])
			metric = after['horizontalTransport']
			self.assertEqual(metric['cycles'], 2)
			self.assertEqual([item['method'] for item in metric['methods']], list(METHODS))
			self.assertEqual(metric['methods'][0]['particles']['1']['q'], 1.5)
			self.assertIsNone(metric['methods'][0]['particles']['2']['q'])

	def test_incomplete_mislabeled_or_inconsistent_results_are_rejected_before_writing(self):
		mutations = (
			lambda result: result['cases'].pop(),
			lambda result: result['cases'].append(deepcopy(result['cases'][0])),
			lambda result: result['cases'][0].update(particle_ids=[1, 3]),
			lambda result: result['cases'][0].update(cycles=3),
			lambda result: result['cases'][0].update(q_x=[None, 9.]),
			lambda result: result['cases'][0].update(width_x=[0., 2.]),
			lambda result: result['cases'][0].update(q_x=[0., 1.5]),
			lambda result: result['metric'].update(coordinate='folded_x'),
		)
		with tempfile.TemporaryDirectory() as temporary:
			for index, mutate in enumerate(mutations):
				result = deepcopy(self.results)
				mutate(result)
				path = Path(temporary) / f'bad-{index}.html'
				with self.subTest(index=index), self.assertRaises(ValueError):
					self.export(path, result)
				self.assertFalse(path.exists())

	def test_repetition_export_replaces_every_method_preserving_initials_and_field_metadata(self):
		with tempfile.TemporaryDirectory() as temporary:
			source = self.export(Path(temporary) / 'baseline.html')
			original = self.read(source)
			self.results['metric']['source_html_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
			positions = {(rho, method): self.positions.copy() for rho in (.3, .4) for method in METHODS}
			for values in positions.values():
				values[1:] += .1
			path = export_horizontal_transport_viewer(source, self.results,
				Path(temporary) / 'rerun.html', cycle_positions=positions)
			config = self.read(path)
			for dataset in config['datasets']:
				for panel in dataset['panels']:
					decoded = np.frombuffer(base64.b64decode(panel['data']), dtype='<f4').reshape(panel['shape'])
					np.testing.assert_array_equal(decoded, positions[(float(dataset['key']), panel['title'])].astype('<f4'))
			self.assertEqual(config['panels'], config['datasets'][0]['panels'])
			self.assertEqual(config['coordinateBounds'], original['coordinateBounds'])
			positions.pop((.3, 'RK4'))
			with self.assertRaisesRegex(ValueError, 'Missing rerun'):
				export_horizontal_transport_viewer(source, self.results,
					Path(temporary) / 'missing.html', cycle_positions=positions)
			self.results['metric']['source_html_sha256'] = 'incorrect'
			with self.assertRaisesRegex(ValueError, 'fingerprint'):
				export_horizontal_transport_viewer(source, self.results,
					Path(temporary) / 'incorrect.html', cycle_positions=positions)


if __name__ == '__main__':
	unittest.main()
