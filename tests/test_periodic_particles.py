"""Shooting residuals, unwrapped closure and phase-duplicate rejection."""
import unittest

import numpy as np

from diagnostics.studies.periodic_particles import distinct_orbit, orbit_metrics
from studies.periodic_particles import refine_points


class AffineMap:
	def __call__(self, points):
		return np.stack((points, .5 * points + np.array((.2, .1))))


class PeriodicParticleTests(unittest.TestCase):
	def test_batched_newton_reaches_known_root_without_mutating_seeds(self):
		seeds = np.array(((.39, .19), (.42, .23)))
		original = seeds.copy()
		points, returns, trace = refine_points(AffineMap(), seeds)
		np.testing.assert_allclose(points, np.tile((.4, .2), (2, 1)), atol=2e-11, rtol=0)
		np.testing.assert_array_equal(seeds, original)
		self.assertLess(np.max(np.abs(returns[-1] - points)), 2e-11)
		self.assertGreater(len(trace), 1)

	def test_closure_does_not_hide_nonzero_winding(self):
		metrics = orbit_metrics(np.array(((.2, .3), (2.2, .3))))
		self.assertAlmostEqual(metrics['closure'], 2.)

	def test_earlier_returns_expose_nonminimal_period(self):
		orbit = np.array(((.2, .3), (.4, .6), (.2, .3), (.4, .6), (.2, .3)))
		self.assertEqual(orbit_metrics(orbit)['earlier_return_min'], 0.)

	def test_nonfinite_returns_cannot_pass_acceptance(self):
		with self.assertRaises(ValueError):
			orbit_metrics(np.array(((.2, .3), (float('nan'), .3))))

	def test_cyclic_and_period_two_copies_are_not_distinct(self):
		first = np.array(((.2, .3), (.4, .6), (.2, .3)))
		shift = np.array(((.4, .6), (.2, .3), (.4, .6)))
		self.assertFalse(distinct_orbit(first, shift))
		self.assertFalse(distinct_orbit(first, shift + (2., 0.)))
		self.assertTrue(distinct_orbit(first, first + (.1, .1)))


if __name__ == '__main__':
	unittest.main()
