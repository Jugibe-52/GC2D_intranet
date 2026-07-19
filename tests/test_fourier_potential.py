from __future__ import annotations

import unittest

import numpy as np
from scipy.special import jv

from classes import FourierPotential, Potential


class FourierPotentialTests(unittest.TestCase):
	@staticmethod
	def single_mode() -> FourierPotential:
		coefficients = np.zeros((2, 2), dtype=np.complex128)
		coefficients[1, 1] = 1.0
		return FourierPotential(1.0, 1, coefficients=coefficients)

	def test_potential_is_an_abstract_contract(self) -> None:
		with self.assertRaises(TypeError):
			Potential()  # type: ignore[abstract]

	def test_single_mode_and_derivatives_are_evaluated_analytically(self) -> None:
		potential = self.single_mode()
		x = np.array([0.2, 0.7, 1.1])
		y = np.array([0.4, 0.3, -0.2])
		t = 0.25
		phase = x + y - t

		np.testing.assert_allclose(potential.field_at_time(t, x, y), np.sin(phase))
		np.testing.assert_allclose(potential.field_at_time(t, x, y, dx=1), np.cos(phase))
		np.testing.assert_allclose(potential.field_at_time(t, x, y, dy=1), np.cos(phase))
		np.testing.assert_allclose(potential.field_at_time(t, x, y, dt=1), -np.cos(phase))
		np.testing.assert_allclose(potential.field_at_time(t, x, y, dx=2), -np.sin(phase))

	def test_electric_field_is_minus_the_analytic_gradient(self) -> None:
		potential = self.single_mode()
		x = np.array([0.2, 0.7])
		y = np.array([0.4, 0.3])
		t = 0.25

		ex, ey = potential.electric_field(t, x, y)

		expected = -np.cos(x + y - t)
		np.testing.assert_allclose(ex, expected)
		np.testing.assert_allclose(ey, expected)

	def test_gyroaverage_attenuates_each_mode_by_the_bessel_factor(self) -> None:
		potential = self.single_mode()
		rho = 0.3

		averaged = potential.gyroaveraged(rho)

		self.assertIsInstance(averaged, FourierPotential)
		np.testing.assert_allclose(
			averaged.coefficients[1, 1],
			jv(0, rho * np.sqrt(2)),
		)
		np.testing.assert_array_equal(potential.coefficients[1, 1], 1.0)

	def test_seeded_generation_and_copy_are_independent(self) -> None:
		first = FourierPotential(0.2, 4, seed=31)
		second = FourierPotential(0.2, 4, seed=31)
		copied = first.copy()

		np.testing.assert_array_equal(first.coefficients, second.coefficients)
		np.testing.assert_array_equal(first.coefficients, copied.coefficients)
		copied.coefficients[1, 1] += 1
		self.assertNotEqual(first.coefficients[1, 1], copied.coefficients[1, 1])


if __name__ == "__main__":
	unittest.main()
