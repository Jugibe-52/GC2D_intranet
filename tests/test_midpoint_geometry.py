"""Independent mathematical checks for the midpoint geometry measurements."""
from __future__ import annotations

import unittest
import numpy as np

from dynamics.coordinates import GCCoordinates
from methods.palindromic.core.midpoint import midpoint_step
from methods.palindromic.maps import GCDoubledMaps
from studies.midpoint_geometry import J, RECIPES, midpoint_geometry, projected_geometry


def nonlinear(time: float, state: np.ndarray) -> np.ndarray:
	x, y = state.reshape(2, -1)
	return (1+.2*np.cos(time))*np.concatenate((-y*(1+.5*x*x), x*(1+.5*y*y)))


def derivative(time: float, state: np.ndarray) -> np.ndarray:
	x, y = state.reshape(2, -1)
	return (1+.2*np.cos(time))*np.stack((np.stack((-x*y, -1-.5*x*x), axis=-1),
		np.stack((1+.5*y*y, x*y), axis=-1)), axis=-2)


class MidpointGeometryTests(unittest.TestCase):
	def setUp(self) -> None:
		self.state = np.array([.7, .3])
		self.maps = GCDoubledMaps(nonlinear, 2, coordinates=GCCoordinates(),
			jacobian=derivative, coupling_frequency=None)

	def test_harmonic_oscillator_closed_forms(self) -> None:
		maps = GCDoubledMaps(lambda t, y: J@y, 2, coordinates=GCCoordinates(),
			jacobian=lambda t, y: J[None], coupling_frequency=None)
		for h in (.5, .25, .125):
			with self.subTest(h=h):
				actual = midpoint_geometry(maps, RECIPES[0], .4, self.state, h)
				expected = np.eye(2)+h*J+(h*J)@(h*J)/2+np.linalg.matrix_power(h*J, 3)/8
				np.testing.assert_allclose(actual['tangent'][0], expected, rtol=0, atol=2e-16)
				np.testing.assert_allclose(actual['det_defect'], h**6/64, rtol=0, atol=3e-16)
				np.testing.assert_allclose(actual['separation_det'], h**6/64, rtol=1e-12, atol=1e-19)
				np.testing.assert_allclose(actual['return_vector'], (h**6/64)*self.state, rtol=0, atol=3e-16)

	def test_analytic_tangents_against_independent_finite_differences(self) -> None:
		for recipe in RECIPES:
			with self.subTest(stages=len(recipe)):
				actual = midpoint_geometry(self.maps, recipe, .7, self.state, .08)
				columns = []
				delta = 1e-5
				for axis in np.eye(2):
					plus = midpoint_step(self.maps, recipe, .7, self.state+delta*axis, .08).state
					minus = midpoint_step(self.maps, recipe, .7, self.state-delta*axis, .08).state
					columns.append((plus-minus)/(2*delta))
				np.testing.assert_allclose(actual['tangent'][0], np.column_stack(columns), rtol=2e-9, atol=2e-10)
				self.assertLess(float(np.max(np.abs(actual['identity_residual']))), 2e-15)

	def test_projected_tangent_and_genuine_non_autonomous_reversal(self) -> None:
		for recipe in RECIPES:
			with self.subTest(stages=len(recipe)):
				actual = projected_geometry(self.maps, recipe, .7, self.state, .1, 1e-14)
				self.assertLess(float(np.max(actual['return_norm'])), 2e-13)
				self.assertLess(float(np.max(np.abs(actual['det_defect']))), 2e-14)
				columns = []
				delta = 1e-5
				for axis in np.eye(2):
					plus = projected_geometry(self.maps, recipe, .7, self.state+delta*axis, .1, 1e-14)['state']
					minus = projected_geometry(self.maps, recipe, .7, self.state-delta*axis, .1, 1e-14)['state']
					columns.append((plus-minus)/(2*delta))
				np.testing.assert_allclose(actual['tangent'][0], np.column_stack(columns), rtol=2e-9, atol=2e-10)

	def test_signed_defect_and_log_sum_match_composed_tangent(self) -> None:
		state = self.state.copy()
		total = np.eye(2)
		logdet = 0.
		for index in range(20):
			step = midpoint_geometry(self.maps, RECIPES[0], index*.05, state, .05, reverse=False)
			state = step['state']
			total = step['tangent'][0]@total
			logdet += np.log1p(step['det_defect'][0])
		self.assertAlmostEqual(float(logdet), float(np.linalg.slogdet(total)[1]), delta=3e-15)
		np.testing.assert_array_equal(self.state, np.array([.7, .3]))


if __name__ == '__main__':
	unittest.main()
