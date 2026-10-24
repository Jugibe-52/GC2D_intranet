"""Accepted energy streams, independent sampling and backend-neutral quadrature."""

import unittest

import numpy as np

from contracts.methods.quadrature import InterpolantQuadrature, StageQuadrature
from diagnostics.methods.energy import EnergyObserver, momentum_increment


class EnergyObservationTests(unittest.TestCase):
	"""The observer changes passive momentum only after accepted complete steps."""

	def test_signed_doubled_weights_keep_physical_normalization(self) -> None:
		quadrature = StageQuadrature(
			(0., 1., 2.),
			(np.array([2., 4.]), np.array([8., 10.]), np.array([6., 12.])),
			(.5 * .25, .5 * -.5, .5 * .75),
			reduction_scale=2.,
		)
		actual = momentum_increment(lambda t, z: z + t, quadrature, 2)
		np.testing.assert_array_equal(actual, np.array([1., 3.]))

	def test_doubled_reduction_retains_overflow_before_physical_normalization(self) -> None:
		run = EnergyObserver().new_run(lambda t, z: z,
			lambda t, z: np.array([np.finfo(float).max]), particle_count=1, initial_time=0.)
		quadrature = StageQuadrature((0., 1.), (np.ones(1), np.ones(1)), (.5, .5),
			reduction_scale=2.)
		with self.assertRaisesRegex(ValueError, "increment became non-finite"):
			run.accept(0, 0., 1., quadrature)
		np.testing.assert_array_equal(run.momentum, [0.])

	def test_configuration_and_repeated_runs_own_initial_momentum(self) -> None:
		initial = np.array([2., 3.])
		configuration = EnergyObserver(initial)
		initial[:] = 20
		first = configuration.new_run(lambda t, z: z, lambda t, z: z,
			particle_count=2, initial_time=0.)
		second = configuration.new_run(lambda t, z: z, lambda t, z: z,
			particle_count=2, initial_time=0.)
		first.accept(0, 0., 1., StageQuadrature((0.,), (np.ones(2),), (1.,)))
		np.testing.assert_array_equal(first.momentum, [3., 4.])
		np.testing.assert_array_equal(second.momentum, [2., 3.])
		first.momentum[:] = 99
		np.testing.assert_array_equal(first.momentum, [3., 4.])
		self.assertFalse(configuration.initial_momentum.flags.writeable)

	def test_fixed_shadow_samples_start_from_left_and_never_accumulate(self) -> None:
		calls = []
		def rate(time: float, state: np.ndarray) -> np.ndarray:
			calls.append(time)
			return state
		run = EnergyObserver(np.array([4.])).new_run(lambda t, z: z, rate,
			particle_count=1, initial_time=0.)
		self.assertEqual(calls, [])
		run.accept(0, 0., 1., StageQuadrature((0.,), (np.array([2.]),), (1.,)))
		shadow = StageQuadrature((0.,), (np.array([2.]),), (.25,))
		np.testing.assert_array_equal(run.sample(.25, shadow), [4.5])
		np.testing.assert_array_equal(run.sample(.25, shadow), [4.5])
		np.testing.assert_array_equal(run.momentum, [6.])
		before = len(calls)
		np.testing.assert_array_equal(run.sample(np.array([0., 1., 1.])), [[4., 6., 6.]])
		self.assertEqual(len(calls), before)
		with self.assertRaisesRegex(ValueError, "shortened quadrature"):
			run.sample(.25)
		with self.assertRaisesRegex(ValueError, "inside"):
			run.sample(1.1, shadow)

	def test_adaptive_samples_use_eight_nodes_of_the_accepted_interpolant(self) -> None:
		calls = []
		def rate(time: float, state: np.ndarray) -> np.ndarray:
			calls.append(time)
			return -state ** np.array([6, 14])
		run = EnergyObserver().new_run(lambda t, z: z, rate,
			particle_count=2, initial_time=0.)
		quadrature = InterpolantQuadrature(0., 1., lambda time: np.array([time, time]))
		run.accept(0, 0., 1., quadrature)
		np.testing.assert_allclose(run.momentum, [-1 / 7, -1 / 15], rtol=3e-14, atol=1e-16)
		self.assertEqual(len(calls), 8)
		np.testing.assert_array_equal(run.sample(np.array([0., 1.])),
			np.column_stack((np.zeros(2), run.momentum)))
		self.assertEqual(len(calls), 8)
		expected = -np.array([.25 ** 7 / 7, .25 ** 15 / 15])
		np.testing.assert_allclose(run.sample(.25), expected, rtol=3e-14, atol=1e-20)
		np.testing.assert_allclose(run.sample(.25), expected, rtol=3e-14, atol=1e-20)
		self.assertEqual(len(calls), 24)
		np.testing.assert_allclose(run.momentum, [-1 / 7, -1 / 15], rtol=3e-14, atol=1e-16)
		self.assertEqual(run.sample(np.empty((0,))).shape, (2, 0))
		self.assertEqual(len(calls), 24)

	def test_duplicate_acceptance_and_failed_evaluation_do_not_change_state(self) -> None:
		run = EnergyObserver().new_run(lambda t, z: z, lambda t, z: z,
			particle_count=1, initial_time=0.)
		valid = StageQuadrature((0.,), (np.ones(1),), (1.,))
		run.accept(0, 0., 1., valid)
		with self.assertRaisesRegex(ValueError, "once"):
			run.accept(0, 0., 1., valid)
		invalid = StageQuadrature((1.,), (np.array([np.nan]),), (1.,))
		with self.assertRaisesRegex(ValueError, "derivative"):
			run.accept(1, 1., 2., invalid)
		np.testing.assert_array_equal(run.momentum, [1.])
		run.accept(1, 1., 2., valid)
		np.testing.assert_array_equal(run.momentum, [2.])

	def test_energy_output_preserves_schema_and_evaluates_h_once(self) -> None:
		calls = []
		def hamiltonian(times: np.ndarray, physical: np.ndarray) -> np.ndarray:
			calls.append(times)
			return physical[:1]
		run = EnergyObserver(np.array([2.])).new_run(hamiltonian, lambda t, z: z,
			particle_count=1, initial_time=0.)
		times, physical = np.array([0., 1.]), np.array([[3., 4.], [0., 0.]])
		auxiliary = {"extended_momentum": np.array([[2., 1.]])}
		result = run.diagnose_history(times, physical, auxiliary)
		self.assertEqual(len(calls), 1)
		self.assertEqual(result["extended_momentum_normalization"], "physical_kappa")
		np.testing.assert_array_equal(result["generalized_energy"], [[5., 5.]])
		np.testing.assert_array_equal(result["energy_drift"], [[0., 1.]])
		self.assertEqual(result["energy_error"], 0.)
		run.diagnose_history(times, physical, auxiliary, energy=physical[:1])
		self.assertEqual(len(calls), 1)

	def test_invalid_momentum_and_quadrature_are_rejected(self) -> None:
		for invalid in (np.array([[1.]]), np.array([np.inf])):
			with self.assertRaisesRegex(ValueError, "finite one-dimensional"):
				EnergyObserver(invalid)
		with self.assertRaisesRegex(ValueError, "one value per particle"):
			EnergyObserver(np.ones(2)).new_run(lambda t, z: z, lambda t, z: z,
				particle_count=1, initial_time=0.)
		with self.assertRaises(ValueError):
			momentum_increment(lambda t, z: z, StageQuadrature((0.,), (), (1.,)), 1)

	def test_weighted_quadrature_is_a_compilable_jax_pytree(self) -> None:
		try:
			import jax
			import jax.numpy as jnp
		except ImportError:
			self.skipTest("Optional JAX runtime is unavailable.")
		with jax.experimental.enable_x64():
			quadrature = StageQuadrature((jnp.array(.1), jnp.array(.4)),
				(jnp.array([1., 2.]), jnp.array([3., 4.])), (jnp.array(.3), jnp.array(-.2)),
				reduction_scale=2.)
			compiled = jax.jit(lambda data: momentum_increment(lambda t, z: t + z, data, 2))
			result = compiled(quadrature)
			np.testing.assert_allclose(result, [-.35, -.25], rtol=2e-15, atol=2e-16)
			self.assertIsInstance(result, jax.Array)


if __name__ == "__main__":
	unittest.main()
