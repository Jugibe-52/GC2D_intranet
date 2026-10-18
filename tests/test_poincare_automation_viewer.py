"""Verify the shared HTML payload and its physical field without reintegration."""

from dataclasses import replace
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_poincare_automation import create_field
from studies.poincare_automation import AutomationConfig, load_automation_field, prepare_automation_case
from studies.poincare_automation_io import prepare_automation, run_automation
from visualization.poincare_automation import particle_presentation, sample_automation_field, export_automation_viewer


def payload(path):
	text=path.read_text()
	return json.loads(text.split('const cfg=',1)[1].split(';\nif(cfg.title)',1)[0])


class AutomationViewerTests(unittest.TestCase):
	def setUp(self):
		self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
		self.source=self.root/'field.h5'; create_field(self.source)
		self.config=AutomationConfig(rhos=(0.,.3),methods=('RK4','BM4Midpoint'),magnetic_fields=(1.5,-2.),
			source=str(self.source),source_index=1,cycles=1,steps_per_cycle=8,field_grid_size=8,vector_grid_size=5)

	def tearDown(self): self.tmp.cleanup()

	def test_all_cases_in_one_viewer_with_original_colors_and_no_integration(self):
		d=prepare_automation(self.config,self.root,self.root/'study'); run_automation(d)
		with patch('studies.poincare_automation.simulate',side_effect=AssertionError('view must not integrate')):
			p=payload(export_automation_viewer(d))
		self.assertEqual(p['firstCycle'],0); self.assertEqual(p['initialCycle'],0)
		self.assertEqual(len(p['datasets']),8)
		self.assertEqual(p['datasetLabel'],'Case')
		colors=particle_presentation()['colors']
		self.assertEqual(colors[40:],['#111827','#7c3aed','#e6007e','#92400e','#0891b2','#0f766e','#d97706','#be123c'])
		for dataset in p['datasets']:
			self.assertIn('rho=',dataset['label']); self.assertIn('B=',dataset['label'])
			self.assertEqual(len(dataset['panels']),2)
			field,returns=dataset['panels']
			self.assertEqual(returns['ids'],list(range(1,49)))
			self.assertEqual(returns['colors'],colors)
			self.assertEqual(returns['shape'],[2,48,2])
			xy=np.frombuffer(base64.b64decode(returns['data']),dtype='<f4').reshape(2,48,2)
			self.assertTrue(np.all((xy>=0)&(xy<=1)))
			self.assertEqual(field['field']['shape'],[51,8,8])
			pot=np.frombuffer(base64.b64decode(field['field']['potential']),dtype='<f4').reshape(51,8,8)
			np.testing.assert_array_equal(pot[0],pot[-1])

	def test_display_field_agrees_across_raw_and_normalized_cases(self):
		baseline=None
		for spatial in (True,False):
			for temporal in (True,False):
				c=replace(self.config,spatial_rescaling=spatial,temporal_rescaling=temporal)
				p=prepare_automation_case(c,c.cases()[0],load_automation_field(c,self.root))
				field=sample_automation_field(p)
				if baseline is None: baseline=field
				for name in ('potential','velocity'):
					np.testing.assert_allclose(field[name],baseline[name],rtol=1e-10,atol=1e-12)

	def test_missing_cases_remain_unavailable_and_source_mismatch_fails(self):
		d=prepare_automation(self.config,self.root,self.root/'partial')
		with self.assertRaisesRegex(ValueError,'No completed'): export_automation_viewer(d)
		from studies.poincare_automation import run_automation_case
		def some(prepared,**kw):
			if prepared.case.rho==.3: raise RuntimeError('missing case')
			return run_automation_case(prepared,**kw)
		with patch('studies.poincare_automation_io.run_automation_case',side_effect=some): run_automation(d)
		p=payload(export_automation_viewer(d))
		self.assertEqual(len(p['datasets']),8)
		self.assertEqual(sum('panels' in row for row in p['datasets']),4)
		with self.source.open('ab') as f: f.write(b'changed source')
		with self.assertRaisesRegex(ValueError,'source file changed'): export_automation_viewer(d)


if __name__=='__main__': unittest.main()
