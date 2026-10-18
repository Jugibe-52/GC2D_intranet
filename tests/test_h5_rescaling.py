"""Physical cyclic-frequency import and equivalent coordinate transformations."""

from pathlib import Path
from dataclasses import replace
import pickle
import tempfile
import unittest

import h5py
import numpy as np
from scipy.integrate import solve_ivp

from dynamics import GuidingCenterDynamics
from potential import Potential, JaxPotential
from studies._potential_snapshot import _H5PotentialSnapshot


class H5RescalingTests(unittest.TestCase):
	def setUp(self) -> None:
		self.temporary = tempfile.TemporaryDirectory()
		self.path = Path(self.temporary.name) / 'field.h5'
		self.L = 0.5
		self.x = 0.2 + self.L * np.arange(16) / 16
		self.y = -0.1 + self.L * np.arange(16) / 16
		qx, qy = np.meshgrid(np.arange(16)/16, np.arange(16)/16, indexing='ij')
		self.mean = 0.002 * (np.cos(2*np.pi*qx) + 0.3*np.sin(2*np.pi*qy))
		self.modes = np.array([0.001*np.exp(2j*np.pi*(qx+2*qy)),
			0.0005*np.exp(2j*np.pi*(2*qx-qy))])
		self.frequency = np.array([3., 3.*np.sqrt(2.)])
		with h5py.File(self.path, 'w') as f:
			f['Rcells'], f['Zcells'] = self.x, self.y
			f['freqs'] = np.r_[0., self.frequency]
			f['fields'] = np.concatenate((self.mean[None].astype(complex), self.modes))
		self.raw = Potential.load(self.path, indx=(1,2))

	def tearDown(self) -> None:
		self.temporary.cleanup()

	def test_default_preserves_source_and_physical_phase(self) -> None:
		np.testing.assert_array_equal(self.raw.mean, self.mean)
		np.testing.assert_array_equal(self.raw.modes, self.modes)
		np.testing.assert_array_equal(self.raw.frequencies, self.frequency)
		np.testing.assert_allclose(self.raw.grid.x, self.x, rtol=1e-12, atol=1e-12)
		np.testing.assert_allclose(self.raw.grid.y, self.y, rtol=1e-12, atol=1e-12)
		self.assertEqual(self.raw.metadata.normalization_factor, 1.)
		self.assertEqual(self.raw.metadata.source_frequency_unit, 'Hz')
		self.assertEqual(self.raw.metadata.time_convention, 'physical_seconds')
		# Independent real trigonometric reconstruction, not the evaluator's exponential.
		t = .43
		for dt in (0,1,2):
			expected = self.mean.copy() if dt == 0 else np.zeros_like(self.mean)
			for c,w in zip(self.modes,2*np.pi*self.frequency):
				phase = w*t + dt*np.pi/2
				expected += 2*w**dt*(c.real*np.cos(phase)-c.imag*np.sin(phase))
			np.testing.assert_allclose(self.raw.evaluate_grid(t,dt=dt),expected,rtol=1e-12,atol=1e-12)

	def test_four_changes_of_variables_include_amplitudes_and_derivatives(self) -> None:
		x,y,t = np.array([.247,.431]), np.array([-.037,.261]), .173
		for spatial in (False,True):
			for temporal in (False,True):
				with self.subTest(spatial=spatial,temporal=temporal):
					p = Potential.load(self.path,indx=(1,2),spatial_rescaling=spatial,temporal_rescaling=temporal)
					s = self.L if spatial else 1.
					T = 1/self.frequency[0] if temporal else 1.
					qx = (x-self.x[0])/s if spatial else x
					qy = (y-self.y[0])/s if spatial else y
					self.assertEqual(p.metadata.time_convention, "physical_cycles" if temporal else "physical_seconds")
					factor = T/s**2
					np.testing.assert_allclose(p.mean,self.mean*factor)
					np.testing.assert_allclose(p.modes,self.modes*factor)
					np.testing.assert_allclose(p.frequencies,self.frequency*T)
					for dx,dy,dt in ((0,0,0),(1,0,0),(0,1,0),(2,0,0),(1,1,0),(0,0,1),(0,0,2),(1,0,1)):
						actual=p.evaluate(t/T,qx,qy,dx=dx,dy=dy,dt=dt)
						expected=factor*s**(dx+dy)*T**dt*self.raw.evaluate(t,x,y,dx=dx,dy=dy,dt=dt)
						np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
					averaged=p.gyroaverage(.025/s)
					np.testing.assert_allclose(averaged.evaluate(t/T,qx,qy),factor*self.raw.gyroaverage(.025).evaluate(t,x,y),rtol=1e-10,atol=1e-10)

	def test_reference_cycle_and_constant_only_policy(self) -> None:
		p=Potential.load(self.path,indx=(2,),temporal_rescaling=True)
		self.assertEqual(p.metadata.time_scale, 1/self.frequency[1])
		pair=Potential.load(self.path,indx=(2,1),temporal_rescaling=True)
		np.testing.assert_allclose(pair.frequencies,self.frequency[::-1]/self.frequency[1])
		np.testing.assert_allclose(p.evaluate_grid(.23),p.evaluate_grid(1.23),rtol=1e-12,atol=1e-12)
		p=Potential.load(self.path,indx=(1,2),temporal_rescaling=True)
		self.assertGreater(np.max(abs(p.evaluate_grid(.23)-p.evaluate_grid(1.23))),1e-5)
		constant=Potential.load(self.path,indx=())
		np.testing.assert_array_equal(constant.mean,self.mean)
		np.testing.assert_array_equal(constant.evaluate_grid(2.,dt=1),np.zeros_like(self.mean))
		with self.assertRaisesRegex(ValueError,'positive-frequency'):
			Potential.load(self.path,indx=(),temporal_rescaling=True)
		with self.assertRaises(TypeError):
			Potential.load(self.path,indx=(1,),spatial_rescaling=1)
		for kw in ('B','characteristic_length','characteristic_frequency','nx','ny'):
			with self.assertRaises(TypeError): Potential.load(self.path,**{kw:1})

	def test_B_is_applied_consistently_by_dynamics(self) -> None:
		state=np.array([.27,.13])
		base=GuidingCenterDynamics(self.raw,rho=.025)
		for B in (1.5,-2.):
			d=GuidingCenterDynamics(self.raw,rho=.025,B=B)
			for name in ('vector_field','particle_vector_field_jacobians','hamiltonian','extended_momentum_derivative'):
				np.testing.assert_allclose(getattr(d,name)(.2,state),getattr(base,name)(.2,state)/B,rtol=1e-12,atol=1e-12)
		for B in (0.,np.nan,np.inf):
			with self.assertRaises(ValueError): GuidingCenterDynamics(self.raw,B=B)
	def test_periodic_shear_against_analytic_time_integral(self) -> None:
		"""Separate time/coordinate errors from the spatial spline approximation."""
		with h5py.File(self.path, 'r+') as f:
			q = np.arange(16) / 16
			mean = np.broadcast_to(.002*np.cos(2*np.pi*q)[:,None], (16,16))
			mode = np.broadcast_to(.001*np.sin(2*np.pi*q)[:,None], (16,16))
			f['fields'][0] = mean.astype(complex)
			f['fields'][1] = mode.astype(complex)
		B, omega = 1.5, 2*np.pi*self.frequency[0]
		period = 2*np.pi/omega
		times = np.linspace(0, 2*period, 21)
		initial = np.array([.26,.38,.59,-.02,.16,.32])
		raw = Potential.load(self.path, indx=(1,))
		# Spline derivatives define the spatial reference; time is integrated exactly.
		mean_dx = raw._splines[0].evaluate(initial[:3], initial[3:], dx=1).real
		mode_dx = raw._splines[1].evaluate(initial[:3], initial[3:], dx=1).real
		expected = np.broadcast_to(initial[:,None], (6,times.size)).copy()
		expected[3:] += (mean_dx[:,None]*times + 2*mode_dx[:,None]*np.sin(omega*times)/omega)/B
		maximum = 0.
		for spatial in (False,True):
			for temporal in (False,True):
				p = Potential.load(self.path, indx=(1,), spatial_rescaling=spatial, temporal_rescaling=temporal)
				s, T = self.L if spatial else 1., period if temporal else 1.
				origin = np.repeat([self.x[0],self.y[0]],3) if spatial else np.zeros(6)
				dynamics = GuidingCenterDynamics(p, B=B)
				sol = solve_ivp(dynamics.vector_field, (0,2*period/T), (initial-origin)/s,
					t_eval=times/T, method='DOP853', rtol=1e-11, atol=1e-13, max_step=period/(100*T))
				self.assertTrue(sol.success)
				maximum = max(maximum, float(np.max(abs(sol.y*s+origin[:,None]-expected)))/self.L)
		self.assertLessEqual(maximum, 1e-9)
		print(f'Analytic shear: max coordinate/L={maximum:.3e}')

	def test_orientation_selection_filtering_and_owned_metadata(self) -> None:
		from scipy.ndimage import gaussian_filter
		p = JaxPotential.load(self.path, indx=(2,1), sigma=.5)
		self.assertIsInstance(p,JaxPotential)
		np.testing.assert_array_equal(p.frequencies,self.frequency[::-1])
		np.testing.assert_allclose(p.mean,gaussian_filter(self.mean,.5))
		for actual,source in zip(p.modes,self.modes[::-1]):
			np.testing.assert_allclose(actual,gaussian_filter(source.real,.5)+1j*gaussian_filter(source.imag,.5))
		self.assertFalse(p.metadata.source_x.flags.writeable)
		self.assertFalse(p.metadata.source_frequencies.flags.writeable)
		with h5py.File(self.path, 'r+') as f:
			del f['Zcells']; f['Zcells'] = self.y[:8]
			del f['fields']; f['fields'] = np.zeros((3,8,16),complex)
		with self.assertRaisesRegex(ValueError,'square'):
			Potential.load(self.path,indx=(1,))

	def test_equivalent_trajectories_with_independent_reference_audit(self) -> None:
		T=1/self.frequency[0]
		times=np.linspace(0,2*T,41)
		initial=np.array([.26,.38,.59,-.02,.16,.32])
		maximum=0.; floor=0.
		for rho in (0.,.025):
			d=GuidingCenterDynamics(self.raw,rho=rho,B=1.5)
			f=lambda t,z:d.vector_field(t,z)
			reference=solve_ivp(f,(0,2*T),initial,t_eval=times,method='DOP853',rtol=1e-11,atol=1e-13,max_step=T/100)
			audit=solve_ivp(f,(0,2*T),initial,t_eval=times,method='Radau',rtol=1e-12,atol=1e-14,max_step=T/200)
			self.assertTrue(reference.success and audit.success)
			floor=max(floor,float(np.max(np.linalg.norm((reference.y-audit.y).reshape(2,3,-1),axis=0)))/self.L)
			for spatial in (False,True):
				for temporal in (False,True):
					p=Potential.load(self.path,indx=(1,2),spatial_rescaling=spatial,temporal_rescaling=temporal)
					s=self.L if spatial else 1.; scale_t=T if temporal else 1.
					origin=np.repeat([self.x[0],self.y[0]],3) if spatial else np.zeros(6)
					field=GuidingCenterDynamics(p,rho=rho/s,B=1.5)
					sol=solve_ivp(field.vector_field,(0,2*T/scale_t),(initial-origin)/s,t_eval=times/scale_t,method='DOP853',rtol=1e-11,atol=1e-13,max_step=T/(100*scale_t))
					self.assertTrue(sol.success)
					error=sol.y*s+origin[:,None]-reference.y
					maximum=max(maximum,float(np.max(np.linalg.norm(error.reshape(2,3,-1),axis=0)))/self.L)
		self.assertLessEqual(floor,1e-9)
		self.assertLessEqual(maximum,1e-8)
		print(f'GC equivalence: max position/L={maximum:.3e}; reference floor/L={floor:.3e}')

	def test_jax_and_serialization_use_cyclic_rates(self) -> None:
		import jax
		jax.config.update('jax_enable_x64',True)
		p=JaxPotential.load(self.path,indx=(1,2),spatial_rescaling=True,temporal_rescaling=True)
		x,y=np.array([.13,.52]),np.array([.72,.21])
		for dt in (0,1,2):
			np.testing.assert_allclose(p.evaluate(.31,x,y,dt=dt),Potential.evaluate(p,.31,x,y,dt=dt),rtol=1e-10,atol=1e-10)
		for restored in (pickle.loads(pickle.dumps(self.raw)),_H5PotentialSnapshot.from_potential(self.raw).restore()):
			np.testing.assert_array_equal(restored.frequencies,self.frequency)
			np.testing.assert_allclose(restored.evaluate_grid(.31),self.raw.evaluate_grid(.31))
		state=self.raw.__getstate__(); state.pop('frequency_convention')
		state['frequencies']=self.raw.frequencies.copy()
		old=object.__new__(Potential); old.__setstate__(state)
		np.testing.assert_allclose(old.evaluate_grid(.31),self.raw.evaluate_grid(.31),rtol=1e-12,atol=1e-12)
		angular_state=self.raw.__getstate__(); angular_state['frequency_convention']='angular'
		angular_state['frequencies']=2*np.pi*self.raw.frequencies
		angular=object.__new__(Potential); angular.__setstate__(angular_state)
		np.testing.assert_allclose(angular.evaluate_grid(.31),self.raw.evaluate_grid(.31),rtol=1e-12,atol=1e-12)
		self.assertEqual(pickle.loads(pickle.dumps(self.raw)).metadata.source_frequency_unit, 'Hz')
		legacy_args=self.raw.metadata.__reduce__()[1][:14]
		legacy_metadata=type(self.raw.metadata)(*legacy_args)
		self.assertEqual(legacy_metadata.time_convention,'legacy_unspecified')
		for changes in ({'time_convention':'unknown'},{'source_frequency_unit':'unknown'}):
			with self.assertRaises(ValueError): replace(self.raw.metadata,**changes)
		snapshot = _H5PotentialSnapshot.from_potential(self.raw)
		legacy = replace(snapshot, frequencies=2*np.pi*self.frequency, frequency_convention='angular')
		np.testing.assert_allclose(legacy.restore().evaluate_grid(.31),self.raw.evaluate_grid(.31),rtol=1e-12,atol=1e-12)
		with self.assertRaisesRegex(ValueError,'frequency convention'):
			replace(snapshot, frequency_convention='unknown').restore()
		state['frequency_convention'] = 'unknown'
		with self.assertRaisesRegex(ValueError,'frequency convention'): old.__setstate__(state)


if __name__ == '__main__': unittest.main()
