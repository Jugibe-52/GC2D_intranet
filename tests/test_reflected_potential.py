"""Scientific contracts for the reflected Hamiltonian and its legacy alternative."""

from __future__ import annotations

import importlib.util
import pickle
import unittest

import numpy as np
from scipy.special import j0

from dynamics.gc import GuidingCenterDynamics
from potential import Grid, Potential
from potential._periodic_spline import _build_periodic_spline


def _polynomial(values: np.ndarray, order: int = 0) -> np.ndarray:
	"""Smoothstep and its first two derivatives, with zero endpoint slopes."""
	if order == 0:
		return 3 * values**2 - 2 * values**3
	if order == 1:
		return 6 * values - 6 * values**2
	return 6 - 12 * values


def _polynomial_potential() -> Potential:
	"""A clamped tensor polynomial gives exact off-grid derivative expectations."""
	grid: Grid = Grid(-0.5, 0.25, 2 / 8, 2 / 6, 8, 6, 2.0)
	u: np.ndarray = np.linspace(0.0, 1.0, grid.nx + 1)[:, None]
	v: np.ndarray = np.linspace(0.0, 1.0, grid.ny + 1)[None, :]
	mean: np.ndarray = _polynomial(u) + 0.3 * _polynomial(v) + 0.2 * _polynomial(u) * _polynomial(v)
	closed: np.ndarray = np.stack((mean, (0.17 + 0.11j) * mean))
	return Potential(
		grid, mean=mean[:-1, :-1], modes=closed[1:, :-1, :-1],
		frequencies=np.array([0.37]), reflected_samples=closed,
	)


def _polynomial_expected(
	t: float | np.ndarray, x: np.ndarray, y: np.ndarray,
	*, dx: int = 0, dy: int = 0, dt: int = 0,
) -> np.ndarray:
	"""Analytic reflected polynomial, with the physical derivative scale included."""
	u: np.ndarray = ((x + 0.5) / 2.0) % 2.0
	v: np.ndarray = ((y - 0.25) / 2.0) % 2.0
	sign_x: np.ndarray = np.where(u <= 1, 1.0, -1.0)
	sign_y: np.ndarray = np.where(v <= 1, 1.0, -1.0)
	u = 1 - np.abs(u - 1)
	v = 1 - np.abs(v - 1)
	px: np.ndarray = _polynomial(u)
	py: np.ndarray = _polynomial(v)
	spatial: np.ndarray
	if dx == 0 and dy == 0:
		spatial = px + 0.3 * py + 0.2 * px * py
	elif dy == 0:
		spatial = _polynomial(u, dx) * (1 + 0.2 * py)
	elif dx == 0:
		spatial = _polynomial(v, dy) * (0.3 + 0.2 * px)
	else:
		spatial = 0.2 * _polynomial(u, dx) * _polynomial(v, dy)
	phase: np.ndarray = (0.17 + 0.11j) * np.exp(2j * np.pi * 0.37 * np.asarray(t))
	temporal: np.ndarray = np.asarray(2 * np.real(phase * (2j * np.pi * 0.37)**dt))
	if dt == 0:
		temporal = temporal + 1.0
	return temporal * spatial * sign_x**dx * sign_y**dy / 2.0**(dx + dy)


def _cosine_potential() -> Potential:
	"""Two resolvable cosine modes distinguish L from the actual period 2L."""
	grid: Grid = Grid(-0.5, 0.25, 2 / 18, 2 / 14, 18, 14, 2.0)
	u: np.ndarray = np.linspace(0.0, 1.0, grid.nx + 1)[:, None]
	v: np.ndarray = np.linspace(0.0, 1.0, grid.ny + 1)[None, :]
	mean: np.ndarray = np.broadcast_to(np.cos(np.pi * u), (grid.nx + 1, grid.ny + 1)).copy()
	mode: np.ndarray = (0.13 + 0.07j) * np.cos(2 * np.pi * u) * np.cos(np.pi * v)
	closed: np.ndarray = np.stack((mean, mode))
	return Potential(
		grid, mean=mean[:-1, :-1], modes=mode[None, :-1, :-1],
		frequencies=np.array([0.37]), reflected_samples=closed,
	)


class ReflectedPotentialTests(unittest.TestCase):
	"""Check geometry, boundary regularity, averaging and persisted semantics."""

	def test_default_completion_and_degree_validation(self) -> None:
		grid: Grid = Grid.periodic(5, 6, period=1.0)
		mean: np.ndarray = np.arange(30, dtype=float).reshape(5, 6) / 30
		original: np.ndarray = mean.copy()
		potential: Potential = Potential(grid, mean=mean)
		self.assertEqual(potential.spatial_extension, "reflected")
		self.assertEqual(potential.spatial_period, 2.0)
		self.assertIs(potential.grid, grid)
		self.assertIs(potential.gyroaverage(0.0), potential)
		self.assertIsNotNone(potential.reflected_samples)
		np.testing.assert_array_equal(potential.reflected_samples[0], np.pad(mean, ((0, 1), (0, 1)), mode="wrap"))
		np.testing.assert_array_equal(mean, original)
		self.assertFalse(potential.reflected_samples.flags.writeable)
		mean[:] = -1
		np.testing.assert_array_equal(potential.mean, original)
		self.assertEqual(Potential.random(A=0.1, M=2, nx=5, ny=6).spatial_extension, "reflected")
		for degree in (2, 4, 5):
			with self.subTest(degree=degree):
				with self.assertRaisesRegex(ValueError, "cubic"):
					Potential(grid, interpolation_order=degree)
				self.assertEqual(Potential(grid, interpolation_order=degree, spatial_extension="periodic").interpolation_order, degree)
		with self.assertRaisesRegex(ValueError, "spatial_extension"):
			Potential(grid, spatial_extension="mirror")

	def test_periodic_option_reproduces_existing_scipy_construction(self) -> None:
		grid: Grid = Grid(-0.5, 0.25, 2 / 8, 2 / 10, 8, 10, 2.0)
		x_mesh: np.ndarray
		y_mesh: np.ndarray
		x_mesh, y_mesh = np.meshgrid(grid.x, grid.y, indexing="ij")
		mean: np.ndarray = np.cos(x_mesh) + np.sin(2 * y_mesh)
		potential: Potential = Potential(grid, mean=mean, spatial_extension="periodic")
		spline = _build_periodic_spline(grid.x, grid.y, mean, spacing=(grid.dx, grid.dy), interpolation_order=3)
		x: np.ndarray = np.array([-6.1, -0.5, 0.27, 1.5, 7.2])
		y: np.ndarray = np.array([4.1, 2.25, 0.94, -3.2, 1.77])
		for dx, dy in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
			np.testing.assert_array_equal(
				potential.evaluate(0.0, x, y, dx=dx, dy=dy),
				spline.evaluate((x - grid.x0) % 2 + grid.x0, (y - grid.y0) % 2 + grid.y0, dx=dx, dy=dy).real,
			)
		self.assertEqual(potential.spatial_period, grid.period)
		self.assertIsNone(potential.reflected_samples)

	def test_shifted_negative_and_multiple_cells_preserve_analytic_derivatives(self) -> None:
		potential: Potential = _polynomial_potential()
		x: np.ndarray = np.array([-8.5, -4.27, -0.5, 0.19, 1.5, 2.61, 3.5, 7.83])
		y: np.ndarray = np.array([6.25, -4.63, 0.25, 2.25, -1.75, 1.38, 4.25, 5.61])
		for t in (0.0, 0.31, 1.7):
			for dx, dy in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
				for dt in (0, 1, 2):
					np.testing.assert_allclose(
						potential.evaluate(t, x, y, dx=dx, dy=dy, dt=dt),
						_polynomial_expected(t, x, y, dx=dx, dy=dy, dt=dt),
						atol=2e-12,
					)
			np.testing.assert_allclose(potential.evaluate(t, x, y), potential.evaluate(t, x + 4, y - 8), atol=1e-14)

	def test_normal_derivatives_vanish_for_all_edges_and_times(self) -> None:
		potential: Potential = _polynomial_potential().gyroaverage(0.23)
		x: np.ndarray = np.linspace(-0.5, 1.5, 37)
		y: np.ndarray = np.linspace(0.25, 2.25, 37)
		for t in (-0.8, 0.0, 0.43, 2.7):
			for dt in (0, 1, 2):
				for edge in (-4.5, -0.5, 1.5, 3.5):
					np.testing.assert_allclose(potential.evaluate(t, np.full_like(y, edge), y, dx=1, dt=dt), 0, atol=2e-13)
				for edge in (-3.75, 0.25, 2.25, 4.25):
					np.testing.assert_allclose(potential.evaluate(t, x, np.full_like(x, edge), dy=1, dt=dt), 0, atol=2e-13)

	def test_seam_field_and_jacobian_limits_allow_transversal_crossing(self) -> None:
		potential: Potential = _polynomial_potential()
		dynamics: GuidingCenterDynamics = GuidingCenterDynamics(potential)
		# This smooth field has nonzero normal speed at each open edge. Its
		# first two scalar derivatives are bounded, so O(epsilon) limits apply.
		epsilon: float = 1e-7
		for state, normal in (
			(np.array([-0.5, 1.25]), 0), (np.array([1.5, 1.25]), 0),
			(np.array([0.5, 0.25]), 1), (np.array([0.5, 2.25]), 1),
		):
			velocity: np.ndarray = dynamics.vector_field(0.31, state)
			self.assertGreater(abs(velocity[normal]), 0.1)
			self.assertLess(abs(velocity[1 - normal]), 1e-13)
			for sign in (-1, 1):
				nearby: np.ndarray = state.copy()
				nearby[normal] += sign * epsilon
				np.testing.assert_allclose(dynamics.vector_field(0.31, nearby), velocity, atol=4e-7)
				np.testing.assert_allclose(
					dynamics.particle_vector_field_jacobians(0.31, nearby),
					dynamics.particle_vector_field_jacobians(0.31, state), atol=5e-7,
				)
				np.testing.assert_allclose(potential.evaluate(0.31, *nearby), potential.evaluate(0.31, *state), atol=1e-13)
		for corner in ((-0.5, 0.25), (-0.5, 2.25), (1.5, 0.25), (1.5, 2.25)):
			np.testing.assert_allclose(dynamics.vector_field(0.31, np.asarray(corner)), 0.0, atol=1e-13)
		# At y=1.25 on the left wall, x decreases with nonzero speed. A single
		# ordinary step therefore crosses that wall without a reflecting dynamics law.
		initial: np.ndarray = np.array([-0.5 + 1e-5, 1.25])
		crossed: np.ndarray = initial + 1e-3 * dynamics.vector_field(0.31, initial)
		self.assertLess(crossed[0], -0.5)

	def test_gyroaverage_matches_cosine_multipliers_and_preserves_high_edges(self) -> None:
		potential: Potential = _cosine_potential()
		rho: float = 0.37
		averaged: Potential = potential.gyroaverage(rho)
		expected: np.ndarray = potential.reflected_samples * np.array([
			j0(np.pi * rho / 2), j0(np.sqrt(5) * np.pi * rho / 2),
		])[:, None, None]
		np.testing.assert_allclose(averaged.reflected_samples, expected, atol=2e-15)
		self.assertEqual(averaged.spatial_extension, "reflected")
		self.assertIs(averaged.grid, potential.grid)
		self.assertIs(averaged.gyroaverage(0.0), averaged)
		self.assertGreater(abs(averaged.reflected_samples[0, 0, 0] - averaged.reflected_samples[0, -1, 0]), 1.0)
		x: np.ndarray
		y: np.ndarray
		x, y = np.meshgrid(np.append(potential.grid.x, 1.5), np.append(potential.grid.y, 2.25), indexing="ij")
		for t in (0.0, 0.43):
			phase: complex = np.exp(2j * np.pi * 0.37 * t)
			np.testing.assert_allclose(averaged.evaluate(t, x, y), expected[0].real + 2 * np.real(expected[1] * phase), atol=2e-14)
		second_radius: float = 0.19
		sequential: Potential = averaged.gyroaverage(second_radius)
		expected *= np.array([
			j0(np.pi * second_radius / 2), j0(np.sqrt(5) * np.pi * second_radius / 2),
		])[:, None, None]
		np.testing.assert_allclose(sequential.reflected_samples, expected, atol=2e-15)

	def test_magnetic_scaling_preserves_averaged_boundaries(self) -> None:
		potential: Potential = _cosine_potential()
		averaged: Potential = potential.gyroaverage(0.37)
		for magnetic_field in (1.5, -2.3):
			dynamics: GuidingCenterDynamics = GuidingCenterDynamics(potential, rho=0.37, B=magnetic_field)
			self.assertEqual(dynamics.hamiltonian_potential.spatial_extension, "reflected")
			np.testing.assert_allclose(
				dynamics.hamiltonian_potential.reflected_samples,
				averaged.reflected_samples / magnetic_field, atol=2e-16,
			)

	def test_pickle_preserves_model_and_missing_historical_tag_is_periodic(self) -> None:
		potential: Potential = _cosine_potential().gyroaverage(0.37)
		restored: Potential = pickle.loads(pickle.dumps(potential))
		self.assertEqual(restored.spatial_extension, "reflected")
		np.testing.assert_array_equal(restored.reflected_samples, potential.reflected_samples)
		self.assertFalse(restored.reflected_samples.flags.writeable)
		legacy: Potential = Potential(potential.grid, mean=potential.mean, spatial_extension="periodic")
		state = legacy.__getstate__()
		state.pop("spatial_extension")
		state.pop("reflected_samples")
		old: Potential = object.__new__(Potential)
		old.__setstate__(state)
		self.assertEqual(old.spatial_extension, "periodic")
		self.assertIsNone(old.reflected_samples)
		np.testing.assert_array_equal(old.evaluate(0.0, np.array([-0.5, 1.5]), np.array([0.25, 2.25])), legacy.evaluate(0.0, np.array([-0.5, 1.5]), np.array([0.25, 2.25])))


@unittest.skipUnless(importlib.util.find_spec("jax"), "JAX is optional")
class ReflectedJaxTests(unittest.TestCase):
	"""Cross-check explicit derivatives and automatic differentiation on seams."""

	@classmethod
	def setUpClass(cls) -> None:
		import jax
		jax.config.update("jax_enable_x64", True)

	def test_values_derivatives_and_time_reconstruction_agree_with_cpu(self) -> None:
		potential: Potential = _polynomial_potential()
		prepared = potential._as_jax()
		self.assertEqual(prepared.spatial_extension, "reflected")
		self.assertIs(prepared.reflected_samples, potential.reflected_samples)
		self.assertIs(prepared._splines, potential._splines)
		x: np.ndarray = np.array([-4.5, -0.5, 0.17, 1.5, 2.71, 3.5, 8.23])
		y: np.ndarray = np.array([2.25, 0.25, -0.46, 2.25, 1.31, 4.25, -3.11])
		for dx, dy, dt in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (1, 1, 0), (0, 2, 0), (0, 0, 1), (0, 0, 2), (1, 0, 1)):
			np.testing.assert_allclose(
				prepared.evaluate(0.31, x, y, dx=dx, dy=dy, dt=dt),
				potential.evaluate(0.31, x, y, dx=dx, dy=dy, dt=dt), atol=2e-12,
			)
		averaged = prepared.gyroaverage(0.23)
		self.assertIs(type(averaged), type(prepared))
		np.testing.assert_allclose(averaged.evaluate(0.31, x, y), potential.gyroaverage(0.23).evaluate(0.31, x, y), atol=2e-13)

	def test_autodiff_hessians_at_exact_edges_corners_and_other_cells(self) -> None:
		import jax
		import jax.numpy as jnp
		potential = _polynomial_potential()._as_jax()
		points: np.ndarray = np.array([
			[-0.5, 0.25], [-0.5, 1.25], [1.5, 0.25], [1.5, 2.25],
			[1.5, 1.25], [3.5, 4.25], [-4.5, -3.75], [-1.17, 3.11],
		])

		def value(state: jax.Array) -> jax.Array:
			return potential.evaluate(0.31, state[0], state[1])

		def gradient(state: jax.Array) -> jax.Array:
			return jnp.stack((potential.evaluate(0.31, state[0], state[1], dx=1), potential.evaluate(0.31, state[0], state[1], dy=1)))

		from_values: np.ndarray = np.asarray(jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(value))))(points))
		from_gradient: np.ndarray = np.asarray(jax.jit(jax.vmap(jax.jacfwd(gradient)))(points))
		x: np.ndarray = points[:, 0]
		y: np.ndarray = points[:, 1]
		xx: np.ndarray = _polynomial_expected(0.31, x, y, dx=2)
		xy: np.ndarray = _polynomial_expected(0.31, x, y, dx=1, dy=1)
		yy: np.ndarray = _polynomial_expected(0.31, x, y, dy=2)
		expected: np.ndarray = np.stack((np.stack((xx, xy), axis=-1), np.stack((xy, yy), axis=-1)), axis=-2)
		np.testing.assert_allclose(from_values, expected, atol=2e-12)
		np.testing.assert_allclose(from_gradient, expected, atol=2e-12)


if __name__ == "__main__":
	unittest.main()
