"""Check the closed-cell cubic representation independently of spatial folding."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.interpolate import BSpline

from potential._reflected_spline import _build_reflected_spline


class ReflectedSplineTests(unittest.TestCase):
	"""Preserve samples, normal derivatives, and exported tensor coefficients."""

	def test_complex_samples_and_zero_normal_derivatives_on_complete_edges(self) -> None:
		x: np.ndarray = np.linspace(-0.4, 1.8, 9)
		y: np.ndarray = np.linspace(0.7, 2.9, 7)
		values: np.ndarray = np.random.default_rng(518).normal(size=(9, 7, 2))
		field: np.ndarray = values[..., 0] + 1j * values[..., 1]
		spline = _build_reflected_spline(x, y, field)
		x_mesh: np.ndarray
		y_mesh: np.ndarray
		x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
		np.testing.assert_allclose(spline.evaluate(x_mesh, y_mesh), field, atol=2e-14)
		# Include the corners and points between nodes, rather than only sampled edges.
		x_edge: np.ndarray = np.linspace(x[0], x[-1], 41)
		y_edge: np.ndarray = np.linspace(y[0], y[-1], 43)
		for x_boundary in (x[0], x[-1]):
			for tangent_order in (0, 1, 2):
				np.testing.assert_allclose(
					spline.evaluate(np.full_like(y_edge, x_boundary), y_edge, dx=1, dy=tangent_order),
					0.0, atol=2e-11,
				)
		for y_boundary in (y[0], y[-1]):
			for tangent_order in (0, 1, 2):
				np.testing.assert_allclose(
					spline.evaluate(x_edge, np.full_like(x_edge, y_boundary), dx=tangent_order, dy=1),
					0.0, atol=2e-11,
				)

	def test_cubic_polynomial_and_derivatives_are_exact(self) -> None:
		# p(u)=3u²-2u³ has zero endpoint slopes but distinct endpoint values.
		x: np.ndarray = np.linspace(0.0, 1.0, 6)
		y: np.ndarray = np.linspace(0.0, 1.0, 8)
		p_x: np.ndarray = 3 * x**2 - 2 * x**3
		p_y: np.ndarray = 3 * y**2 - 2 * y**3
		field: np.ndarray = (1.0 + 2.0j) * p_x[:, None] * p_y[None, :]
		spline = _build_reflected_spline(x, y, field)
		points_x: np.ndarray = np.array([0.0, 0.12, 0.49, 0.83, 1.0])
		points_y: np.ndarray = np.array([1.0, 0.36, 0.61, 0.18, 0.0])
		x_derivatives: tuple[np.ndarray, ...] = (
			3 * points_x**2 - 2 * points_x**3,
			6 * points_x - 6 * points_x**2,
			6 - 12 * points_x,
		)
		y_derivatives: tuple[np.ndarray, ...] = (
			3 * points_y**2 - 2 * points_y**3,
			6 * points_y - 6 * points_y**2,
			6 - 12 * points_y,
		)
		for dx in range(3):
			for dy in range(3):
				np.testing.assert_allclose(
					spline.evaluate(points_x, points_y, dx=dx, dy=dy),
					(1.0 + 2.0j) * x_derivatives[dx] * y_derivatives[dy],
					atol=2e-11,
				)

	def test_exported_coefficients_use_standard_tensor_basis_and_order(self) -> None:
		x: np.ndarray = np.linspace(-1.0, 2.0, 7)
		y: np.ndarray = np.linspace(0.5, 3.5, 5)
		field: np.ndarray = np.random.default_rng(23).normal(size=(7, 5)) * (1.0 - 0.7j)
		spline = _build_reflected_spline(x, y, field)
		tx: np.ndarray
		ty: np.ndarray
		tx, ty = spline.real.get_knots()
		np.testing.assert_array_equal(tx, spline.imag.get_knots()[0])
		np.testing.assert_array_equal(ty, spline.imag.get_knots()[1])
		coefficients: np.ndarray = (
			spline.real.get_coeffs() + 1j * spline.imag.get_coeffs()
		).reshape(tx.size - 4, ty.size - 4)
		points_x: np.ndarray = np.array([x[0], -0.27, 0.58, 1.31, x[-1]])
		points_y: np.ndarray = np.array([y[-1], 1.04, 2.19, 3.07, y[0]])
		basis_x: BSpline = BSpline(tx, np.eye(tx.size - 4), 3)
		basis_y: BSpline = BSpline(ty, np.eye(ty.size - 4), 3)
		for dx in range(3):
			for dy in range(3):
				expected: np.ndarray = np.einsum(
					"pi,ij,pj->p", basis_x(points_x, nu=dx), coefficients, basis_y(points_y, nu=dy),
				)
				np.testing.assert_allclose(
					spline.evaluate(points_x, points_y, dx=dx, dy=dy), expected, atol=2e-12,
				)


if __name__ == "__main__":
	unittest.main()
