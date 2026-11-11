"""Physical horizontal travel, full-step sampling and backend agreement."""

import importlib.util
import unittest

import numpy as np

from contracts.methods.step import StepInfo
from diagnostics import HorizontalTransportObserver
from diagnostics.execution.horizontal_transport import HORIZONTAL_METRIC_ARRAYS
from diagnostics.methods.energy import EnergyObserver
from dynamics.coordinates import GCCoordinates
from dynamics.gc import GuidingCenterDynamics
from execution._modal_worker import encode_job, execute_cycle_payload
from execution.options import ExecutionOptions
from methods import BM4Implicit, BM4Midpoint, GaussLegendre4, RK4
from potential import Potential
from problem import InitialValueProblem
from request import SimulationRequest
from simulation import simulate


class PrescribedHorizontalVelocity:
	state_dimension = 2
	coordinates = GCCoordinates()

	def __init__(self, mode="constant"):
		self.mode = mode

	def vector_field(self, t, state):
		n = state.size // 2
		factor = {"constant": 1., "quadratic": t*t,
			"oscillatory": np.cos(2*np.pi*t)}[self.mode]
		return np.concatenate((np.resize([1., -2., 0.], n)*factor, np.zeros(n)))


def prescribed_problem(mode="constant", outputs=None):
	return InitialValueProblem(PrescribedHorizontalVelocity(mode), np.zeros(6),
		request=SimulationRequest((0., 1.), np.array([0., 1.]) if outputs is None else outputs))


def physical_problem():
	potential = Potential.random(A=.05, M=3, nx=16, ny=16, seed=27)
	return InitialValueProblem(GuidingCenterDynamics(potential, rho=.3),
		np.array([.2, .4, .3, .5]),
		request=SimulationRequest((0., .13), np.array([0., .025, .13])))


class HorizontalTransportTests(unittest.TestCase):
	def test_constant_velocity_unwrapped_width_and_zero_width_policy(self):
		result = simulate(prescribed_problem(), RK4(step=.13),
			observers=(HorizontalTransportObserver(cycle_duration=.5),))
		d = result.diagnostics
		np.testing.assert_allclose(d["horizontal_travel"], [1., 2., 0.])
		np.testing.assert_allclose(d["horizontal_width"], [1., 2., 0.])
		np.testing.assert_allclose(d["horizontal_min"], [0., -2., 0.])
		np.testing.assert_allclose(d["horizontal_max"], [1., 0., 0.])
		np.testing.assert_allclose(d["horizontal_transport_rate"][:2], [.5, .5])
		self.assertTrue(np.isnan(d["horizontal_transport_rate"][2]))
		self.assertEqual(d["horizontal_cycles"], 2.)
		self.assertEqual(d["horizontal_observed_steps"], 8)
		self.assertEqual(d["horizontal_zero_width"], "undefined_nan")

	def test_oscillations_between_poincare_returns_and_quadrature_convergence(self):
		errors = []
		for step in (.05, .025, .0125):
			result = simulate(prescribed_problem("oscillatory"), RK4(step=step),
				observers=(HorizontalTransportObserver(),))
			d = result.diagnostics
			# One-period endpoint displacement is zero despite positive total travel.
			self.assertLess(abs(result.states[-1, 0]), 1.e-14)
			self.assertGreater(d["horizontal_travel"][0], .6)
			errors.append(abs(d["horizontal_travel"][0] - 2/np.pi))
			np.testing.assert_allclose(d["horizontal_transport_rate"][:2], [2., 2.], atol=.02)
		self.assertGreater(errors[0] / errors[1], 3.9)
		self.assertGreater(errors[1] / errors[2], 3.9)

	def test_velocity_quadrature_is_not_endpoint_distance(self):
		result = simulate(prescribed_problem("quadratic"), RK4(step=.2),
			observers=(HorizontalTransportObserver(),))
		self.assertAlmostEqual(result.states[-1, 0], 1/3)
		self.assertAlmostEqual(result.diagnostics["horizontal_travel"][0], .34)

	def test_observation_is_passive_fresh_and_independent_of_saved_samples(self):
		observer = HorizontalTransportObserver()
		method = RK4(step=.13)
		plain = simulate(prescribed_problem("oscillatory"), method)
		observed = simulate(prescribed_problem("oscillatory"), method, observers=(observer,))
		dense = simulate(prescribed_problem("oscillatory", np.array([0., .01, .2, .67, 1.])),
			method, observers=(observer,))
		np.testing.assert_array_equal(plain.states, observed.states)
		for key in HORIZONTAL_METRIC_ARRAYS:
			np.testing.assert_array_equal(observed.diagnostics[key], dense.diagnostics[key])
			self.assertEqual(observed.diagnostics[key].shape, (3,))
		self.assertEqual(observed.diagnostics["horizontal_observed_steps"], 8)

	def test_configuration_and_nonfinite_velocity_are_rejected(self):
		for duration in (0., -1., np.inf, np.nan, True):
			with self.assertRaises(ValueError):
				HorizontalTransportObserver(cycle_duration=duration)
		with self.assertRaisesRegex(ValueError, "at most one"):
			simulate(prescribed_problem(), RK4(), observers=(HorizontalTransportObserver(),)*2)
		run = HorizontalTransportObserver().new_run(lambda t, z: np.array([np.inf, 0.]),
			GCCoordinates().positions, particle_count=1)
		with self.assertRaisesRegex(ValueError, "finite"):
			run.accept(StepInfo(0, 0., .1, .1, np.zeros(2)), np.zeros(2))

	def test_modal_cycle_payload_retains_metric_vectors_and_discards_step_histories(self):
		problem = physical_problem()
		observer = HorizontalTransportObserver()
		payload = encode_job(problem, RK4(step=.02), None, (observer, EnergyObserver()))
		result = execute_cycle_payload(payload)[2]
		local = simulate(problem, RK4(step=.02), observers=(observer,))
		for key in HORIZONTAL_METRIC_ARRAYS:
			np.testing.assert_array_equal(result.diagnostics[key], local.diagnostics[key])
		self.assertNotIn("step_times", result.diagnostics)
		self.assertNotIn("physical_hamiltonian", result.diagnostics)
		self.assertEqual(result.diagnostics["horizontal_observed_steps"], 7)


@unittest.skipUnless(importlib.util.find_spec("jax") is not None, "JAX optional")
class HorizontalTransportJaxTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		import jax
		jax.config.update("jax_enable_x64", True)

	def test_all_four_methods_match_host_and_keep_physical_solutions(self):
		problem = physical_problem()
		for method_type in (BM4Midpoint, BM4Implicit, RK4, GaussLegendre4):
			with self.subTest(method=method_type.__name__):
				method = method_type(step=.03)
				observers = (HorizontalTransportObserver(), EnergyObserver())
				host = simulate(problem, method, observers=observers)
				device = simulate(problem, method, observers=observers,
					options=ExecutionOptions(backend="jax"))
				plain = simulate(problem, method, options=ExecutionOptions(backend="jax"))
				np.testing.assert_allclose(device.states, plain.states, rtol=1.e-13, atol=1.e-14)
				for key in HORIZONTAL_METRIC_ARRAYS:
					np.testing.assert_allclose(device.diagnostics[key], host.diagnostics[key], rtol=2.e-11, atol=1.e-13)
				self.assertEqual(device.diagnostics["horizontal_observed_steps"], 5)

	def test_jax_modal_payload_preserves_compiled_observation(self):
		payload = encode_job(physical_problem(), RK4(step=.03),
			ExecutionOptions(backend="jax"), (HorizontalTransportObserver(),))
		result = execute_cycle_payload(payload)[2]
		self.assertEqual(result.diagnostics["execution_backend"], "jax")
		self.assertEqual(result.diagnostics["horizontal_travel"].shape, (2,))
		self.assertTrue(np.all(result.diagnostics["horizontal_travel"] > 0))

	def test_jax_stationary_width_remains_explicitly_undefined(self):
		potential = Potential.random(A=0., M=2, nx=12, ny=12, seed=19)
		problem = InitialValueProblem(GuidingCenterDynamics(potential),
			np.array([.2, .4, .3, .5]),
			request=SimulationRequest((0., .1), np.array([0., .1])))
		result = simulate(problem, RK4(step=.03), observers=(HorizontalTransportObserver(),),
			options=ExecutionOptions(backend="jax"))
		np.testing.assert_array_equal(result.diagnostics["horizontal_width"], [0., 0.])
		np.testing.assert_array_equal(result.diagnostics["horizontal_travel"], [0., 0.])
		self.assertTrue(np.all(np.isnan(result.diagnostics["horizontal_transport_rate"])))


if __name__ == "__main__":
	unittest.main()
