"""Method-owned nominal steps, final clipping and independent sampling."""

from dataclasses import replace
import unittest

import numpy as np

from dynamics import GuidingCenterDynamics
from execution.options import ExecutionOptions
from diagnostics.methods.energy import EnergyObserver
from execution._host import _fixed_schedule
from potential import Potential
from simulation import (
	InitialValueProblem, SimulationRequest, simulate, ExplicitEuler, RK4, GaussLegendre4, SDIRK4,
	HBVM42, ABBA2Implicit, ABBA4Implicit, ABBA2Midpoint, BM4Implicit, BM4Midpoint, DOP853, Radau,
)

FIXED = (ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42,
		 ABBA2Implicit, ABBA4Implicit, ABBA2Midpoint, BM4Implicit, BM4Midpoint)


class TimeField:
	"""Euler exposes the exact durations and evaluation times used by its driver."""

	state_dimension = 2
	coordinates = GuidingCenterDynamics.coordinates

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		return np.array([t, 1.])


class MethodStepTests(unittest.TestCase):
	def test_constructor_validation_and_request_boundary(self) -> None:
		for cls in FIXED:
			self.assertEqual(cls().step, .1)
			for invalid in (0., -1., np.inf, np.nan, True, np.bool_(False)):
				with self.subTest(method=cls.__name__, value=invalid):
					with self.assertRaises(ValueError):
						cls(step=invalid)
		for cls in (DOP853, Radau):
			self.assertFalse(hasattr(cls(), 'step'))
			with self.assertRaises(ValueError):
				cls(max_step=False)
		request = SimulationRequest.uniform()
		self.assertFalse(hasattr(request, 'max_step'))
		with self.assertRaises(TypeError):
			SimulationRequest.uniform(max_step=.1)

	def test_only_last_main_step_is_shortened_and_configuration_is_reusable(self) -> None:
		p = InitialValueProblem(TimeField(), np.array([0., 0.]))
		events = []
		method = ExplicitEuler(step=.3)
		r = SimulationRequest((0., 1.), np.array([0., .1, .3, .55, .9, 1.]))
		result = simulate(p.with_request(r), method, observers=(events.append,))
		np.testing.assert_allclose(result.diagnostics['step_sizes'], [.3, .3, .3, .1], atol=2e-16, rtol=0)
		np.testing.assert_allclose(result.diagnostics['step_times'], [.3, .6, .9, 1.], atol=2e-16, rtol=0)
		np.testing.assert_allclose(result.states[:, -1], [.36, 1.], atol=2e-16, rtol=0)
		self.assertEqual(result.diagnostics['step_times'][-1], 1.)
		self.assertEqual(len(events), 4)
		self.assertEqual(method.step, .3)
		sparse = simulate(p.with_request(SimulationRequest((0., 1.), np.array([0., 1.]))), method)
		np.testing.assert_array_equal(sparse.states[:, -1], result.states[:, -1])
		short = simulate(p.with_request(SimulationRequest((2., 2.1), np.array([2., 2.1]))), method)
		self.assertEqual(short.diagnostics['step_count'], 1)
		self.assertEqual(method.step, .3)

	def test_roundoff_and_unrepresentable_grids(self) -> None:
		for span, step, count in (((.3, .34), .02, 2), ((0., .3), .1, 3), ((-1., -.4), .3, 2)):
			starts, ends, sizes = _fixed_schedule(span, step)
			self.assertEqual(len(starts), count)
			self.assertEqual(ends[-1], span[1])
			np.testing.assert_allclose(sizes, step, rtol=0, atol=2e-16)
		with self.assertRaisesRegex(ValueError, 'representable time'):
			_fixed_schedule((1e16, 1e16 + 4), .1)

	def test_all_fixed_methods_match_jax_with_a_shortened_final_step(self) -> None:
		try:
			import jax
		except ImportError:
			self.skipTest('Optional JAX is unavailable')
		potential = Potential.random(A=.08, M=3, nx=16, ny=16, seed=27, interpolation_order=5)
		p = InitialValueProblem(GuidingCenterDynamics(potential, rho=.05), np.array([1., 1.2]))
		r = SimulationRequest((.3, .355), np.array([.3, .305, .32, .345, .355]))
		with jax.experimental.enable_x64():
			for cls in FIXED:
				with self.subTest(method=cls.__name__):
					method = cls(step=.02)
					host = simulate(p.with_request(r), method, observers=(EnergyObserver(),))
					device = simulate(p.with_request(r), method, options=ExecutionOptions(backend='jax'), observers=(EnergyObserver(),))
					np.testing.assert_allclose(host.diagnostics['step_sizes'], [.02, .02, .015], atol=1e-16, rtol=0)
					for key in ('step_start_times', 'step_times', 'step_sizes'):
						np.testing.assert_array_equal(host.diagnostics[key], device.diagnostics[key])
					np.testing.assert_allclose(device.states, host.states, rtol=3e-10, atol=3e-11)
					np.testing.assert_allclose(device.diagnostics['extended_momentum'], host.diagnostics['extended_momentum'], rtol=3e-9, atol=3e-11)
					if 'nonlinear_iterations' in host.diagnostics:
						np.testing.assert_array_equal(host.diagnostics['nonlinear_iterations'], device.diagnostics['nonlinear_iterations'])
					self.assertEqual(method.step, .02)


if __name__ == '__main__':
	unittest.main()
