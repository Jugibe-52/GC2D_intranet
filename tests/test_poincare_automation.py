"""Independent geometry, physical scaling, numerical references and archive reuse."""

from dataclasses import replace
import importlib.util
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scipy.integrate import solve_ivp

from dynamics.gc import GuidingCenterDynamics
from potential.potential import Potential
from studies.poincare_automation import (
	AutomationConfig, METHODS, fixed_particle_fractions, load_automation_field,
	prepare_automation_case, run_automation_case, validate_saved_automation,
)
from studies.poincare_automation_io import (
	case_key, prepare_automation, read_automation, run_automation, load_completed_case,
)


def create_field(path):
	"""Non-unit cell, offset origin and 3 Hz cyclic forcing with a small smooth drift."""
	L, n = .5, 16
	q = np.arange(n)/n
	x, y = np.meshgrid(q, q, indexing='ij')
	mean = .002 * (np.cos(2*np.pi*x) + .3*np.sin(2*np.pi*y))
	mode = .001 * np.exp(2j*np.pi*(x+2*y))
	with h5py.File(path, 'w') as f:
		f['Rcells'], f['Zcells'] = .2+L*q, -.1+L*q
		f['freqs'] = [0., 3.]
		f['fields'] = np.stack((mean.astype(complex), mode))


class AutomationTests(unittest.TestCase):
	def setUp(self):
		self.tmp = tempfile.TemporaryDirectory()
		self.root = Path(self.tmp.name)
		self.source = self.root/'field.h5'
		create_field(self.source)
		self.config = AutomationConfig(rhos=(.3,), methods=('RK4',), source=str(self.source),
			source_index=1, cycles=2, steps_per_cycle=50, field_grid_size=8, vector_grid_size=5)

	def tearDown(self):
		self.tmp.cleanup()

	def prepared(self, config=None):
		config = self.config if config is None else config
		return prepare_automation_case(config, config.cases()[0], load_automation_field(config, self.root))

	def test_exact_reference_particles_and_default_schedule(self):
		i = np.arange(40)
		expected = .5 + (.425*i/39)[:,None]*np.column_stack((np.cos(2*np.pi*(i%8)/8),np.sin(2*np.pi*(i%8)/8)))
		actual = fixed_particle_fractions()
		np.testing.assert_allclose(actual[:40], expected, rtol=0, atol=2e-16)
		np.testing.assert_array_equal(actual[40:], [[.4407,.5817],[.4417,.5896],[.6073,.5675],[.603,.5753],
			[.4317,.3856],[.4376,.3814],[.5749,.3911],[.579,.3918]])
		config = AutomationConfig(rhos=(0.,.3), methods=('BM4Midpoint',), source=str(self.source), source_index=1)
		p = self.prepared(config)
		self.assertEqual(p.request.max_step, 1/50)
		self.assertEqual(p.metadata['expected_step_count'], 250000)
		np.testing.assert_array_equal(p.request.output_times, np.arange(5001))
		self.assertEqual(p.problem.initial_state.shape, (96,))
		self.assertEqual(p.case.B, 1.5)
		self.assertTrue(config.spatial_rescaling and config.temporal_rescaling)
		self.assertEqual(p.metadata['time_convention'], 'physical_cycles')

	def test_invalid_choices_and_collision_free_case_matrix(self):
		for kw in [dict(rhos=()),dict(rhos=(.3,.3)),dict(rhos=(-1.,)),dict(rhos=(True,)),
			dict(rhos=(np.nan,)),dict(magnetic_fields=(0.,)),dict(methods=('FC',)),dict(methods=()),
			dict(cycles=True),dict(steps_per_cycle=1.5),dict(spatial_rescaling=1),dict(sigma=-1),
			dict(coupling_frequency=.3),dict(source_index=0),dict(backend='other'),dict(device='gpu')]:
			with self.subTest(kw=kw), self.assertRaises((TypeError, ValueError)):
				replace(self.config, **kw)
		c = replace(self.config, rhos=(.3,.300000001), magnetic_fields=(1.5,-2.), methods=METHODS)
		self.assertEqual(len(c.cases()),16)
		self.assertEqual(len({case_key(c, case) for case in c.cases()}),16)
		self.assertNotEqual(case_key(c,c.cases()[0]),case_key(replace(c,cycles=3),c.cases()[0]))

	def test_fields_rho_and_time_scale_for_all_representations(self):
		raw = Potential.load(self.source, indx=(1,))
		q = fixed_particle_fractions(); X = np.array((.2,-.1))+.5*q
		state = np.r_[X[:,0],X[:,1]]
		for spatial,temporal,rho,B in product((False,True),(False,True),(0.,.3),(1.5,-2.)):
			c=replace(self.config, spatial_rescaling=spatial, temporal_rescaling=temporal,
				rhos=(rho,), magnetic_fields=(B,))
			p=self.prepared(c); s=.5 if spatial else 1.; T=1/3 if temporal else 1.
			d=GuidingCenterDynamics(raw,rho=rho*.06/(2*np.pi),B=B)
			actual=p.dynamics.vector_field(.173/T,p.problem.initial_state)
			expected=T/s*d.vector_field(.173,state)
			np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12*np.max(abs(expected)))
			np.testing.assert_allclose(p.dynamics.hamiltonian(.173/T,p.problem.initial_state),
				T/s**2*d.hamiltonian(.173,state),rtol=1e-10,atol=1e-12)
			self.assertAlmostEqual(p.dynamics.rho,rho*.06/(2*np.pi*s))
			self.assertAlmostEqual(p.request.output_times[-1],2/(3*T))

	def test_all_methods_equivalent_and_audited_reference(self):
		base=self.prepared()
		rhs=base.dynamics.vector_field
		reference=solve_ivp(rhs,(0.,2.),base.problem.initial_state,method='DOP853',
			t_eval=[0.,1.,2.],rtol=1e-11,atol=1e-13,max_step=.005)
		audit=solve_ivp(rhs,(0.,2.),base.problem.initial_state,method='Radau',
			t_eval=[0.,1.,2.],rtol=1e-12,atol=1e-14,max_step=.0025)
		self.assertTrue(reference.success and audit.success)
		floor=float(np.max(abs(reference.y-audit.y)))
		self.assertLessEqual(floor,1e-10)
		maximum=equivalence=0.
		for method in METHODS:
			baseline=None
			for spatial,temporal in product((True,False),(True,False)):
				c=replace(self.config,methods=(method,),spatial_rescaling=spatial,temporal_rescaling=temporal,
					newton_absolute_tolerance=1e-13 if spatial else 5e-14,newton_relative_tolerance=1e-13)
				p=self.prepared(c); saved=run_automation_case(p)
				self.assertEqual(saved.solution.states.shape,(96,3))
				self.assertEqual(saved.solution.diagnostics['step_count'],100)
				q=saved.solution.states if spatial else (saved.solution.states-np.repeat([.2,-.1],48)[:,None])/.5
				if baseline is None: baseline=q
				equivalence=max(equivalence,float(np.max(abs(q-baseline))))
				maximum=max(maximum,float(np.max(abs(q-reference.y))))
		self.assertLessEqual(equivalence,1e-8)
		self.assertLessEqual(maximum,1e-8)
		print(f'Automation reference floor={floor:.3e}, method error={maximum:.3e}, coordinate equivalence={equivalence:.3e}')

	def test_prepare_without_source_reuse_and_mismatch_rejection(self):
		missing=replace(self.config,source=str(self.root/'missing.h5'))
		with patch('studies.poincare_automation.load_automation_field',side_effect=AssertionError):
			prepare_automation(missing,self.root,self.root/'only-prepare')
		directory=prepare_automation(self.config,self.root,self.root/'study')
		self.assertEqual(prepare_automation(self.config,self.root,directory),directory)
		with self.assertRaises(FileExistsError):
			prepare_automation(replace(self.config,rhos=(.4,)),self.root,directory)
		manifest=run_automation(directory)
		self.assertTrue(all(r['status']=='complete' for r in manifest['cases'].values()))
		with patch('studies.poincare_automation.simulate',side_effect=AssertionError('must reuse')):
			self.assertEqual(run_automation(directory),manifest)
		p=self.prepared(); saved=load_completed_case(directory,p)
		for key,value in [('source_sha256','bad'),('rho_runtime',.99),('B',2.),
			('particle_ids',list(range(48))),('time_convention','legacy_cycles')]:
			with self.subTest(key=key),self.assertRaises(ValueError):
				validate_saved_automation(replace(saved,metadata={**saved.metadata,key:value}),p)
		with h5py.File(self.source,'r+') as f: f['fields'][0,0,0]=.123
		with self.assertRaisesRegex(ValueError,'source file changed'): run_automation(directory)

	def test_failures_preserve_successful_cases_and_retry_partial_outputs(self):
		c=replace(self.config,rhos=(0.,.3),cycles=1)
		d=prepare_automation(c,self.root,self.root/'partial')
		from studies.poincare_automation import run_automation_case as run
		def first_pass(prepared,**kw):
			if prepared.case.rho==.3: raise RuntimeError('test interruption')
			return run(prepared,**kw)
		with patch('studies.poincare_automation_io.run_automation_case',side_effect=first_pass):
			manifest=run_automation(d)
		self.assertEqual(sorted(r['status'] for r in manifest['cases'].values()),['complete','failed'])
		with patch('studies.poincare_automation_io.run_automation_case',wraps=run) as mocked:
			manifest=run_automation(d)
			self.assertEqual(mocked.call_count,1)
		self.assertTrue(all(r['status']=='complete' for r in manifest['cases'].values()))

	def test_partial_archive_is_preserved_before_retry(self):
		c=replace(self.config,cycles=1)
		d=prepare_automation(c,self.root,self.root/'partial-archive')
		p=self.prepared(c)
		archive=d/'cases'/case_key(c,c.cases()[0])/p.metadata['input_digest']
		archive.mkdir(parents=True)
		(archive/'interrupted.txt').write_text('preserve this partial output')
		manifest=run_automation(d)
		self.assertTrue(all(r['status']=='complete' for r in manifest['cases'].values()))
		preserved=list(archive.parent.glob('*.incomplete-*/interrupted.txt'))
		self.assertEqual(len(preserved),1)
		self.assertEqual(preserved[0].read_text(),'preserve this partial output')
		load_completed_case(d,p)

	def test_coupling_rate_transforms_with_runtime_period(self):
		from studies.poincare_automation import automation_method
		for temporal in (True,False):
			c=replace(self.config,methods=('BM4Midpoint',),coupling_frequency=.25,temporal_rescaling=temporal)
			p=self.prepared(c)
			self.assertAlmostEqual(automation_method(p).coupling_frequency*p.metadata['runtime_cycle_duration'],.25)

	@unittest.skipUnless(importlib.util.find_spec('jax'),'JAX is optional')
	def test_unavailable_backend_preflight_never_integrates(self):
		c=replace(self.config,backend='jax')
		d=prepare_automation(c,self.root,self.root/'unavailable')
		with patch('execution._jax.resolve_device',side_effect=RuntimeError('unavailable')):
			with patch('studies.poincare_automation_io.verified_automation_field',side_effect=AssertionError('no field read')):
				with self.assertRaisesRegex(RuntimeError,'unavailable'):
					run_automation(d)
		self.assertTrue(all(row['status']=='requested' for row in read_automation(d)[2]['cases'].values()))

	@unittest.skipUnless(importlib.util.find_spec('jax'),'JAX is optional')
	def test_jax_matches_numpy_for_joint_particles(self):
		import jax
		with jax.experimental.enable_x64():
			c=replace(self.config,cycles=1,steps_per_cycle=8)
			cpu=run_automation_case(self.prepared(c))
			jax_result=run_automation_case(self.prepared(replace(c,backend='jax')))
			np.testing.assert_allclose(jax_result.solution.states,cpu.solution.states,rtol=1e-11,atol=1e-13)

	def test_cli_prepare_run_view_and_status(self):
		script=Path(__file__).resolve().parents[1]/'scripts/run_poincare_automation.py'
		d=self.root/'cli'
		args=['prepare','--rho','0','.3','--method','RK4','--source',str(self.source),
			'--source-index','1','--cycles','1','--steps-per-cycle','8','--output',str(d),
			'--field-grid-size','8','--vector-grid-size','5']
		for command in [args,['run',str(d)],['view',str(d)],['status',str(d)]]:
			result=subprocess.run([sys.executable,str(script),*command],capture_output=True,text=True)
			self.assertEqual(result.returncode,0,result.stdout+result.stderr)
		self.assertTrue((d/'viewer.html').is_file())
		result=subprocess.run([sys.executable,str(script),'prepare','--rho','.3'],capture_output=True)
		self.assertNotEqual(result.returncode,0)


if __name__=='__main__': unittest.main()
