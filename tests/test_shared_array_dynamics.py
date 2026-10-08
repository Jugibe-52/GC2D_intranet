"""Backend-preserving physical layouts and shared GC/FC dynamics."""

from __future__ import annotations

import importlib.util
from typing import Any, ClassVar
import unittest

import numpy as np

from contracts.state_layout import FCStateLayout, GCStateLayout
from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics
from execution._jax import prepare_dynamics
from potential.grid import Grid
from potential.potential import Potential


@unittest.skipUnless(importlib.util.find_spec("jax"), "Optional JAX is not installed")
class SharedArrayDynamicsTests(unittest.TestCase):
	"""The same physical objects support eager, traced, and differentiated arrays."""

	jax: ClassVar[Any]
	jnp: ClassVar[Any]
	previous_x64: ClassVar[bool]
	potential: ClassVar[Potential]
	gc: ClassVar[GuidingCenterDynamics]
	fc: ClassVar[FullCyclotronDynamics]

	@classmethod
	def setUpClass(cls) -> None:
		import jax
		import jax.numpy as jnp

		cls.jax = jax
		cls.jnp = jnp
		cls.previous_x64 = jax.config.read("jax_enable_x64")
		jax.config.update("jax_enable_x64", True)
		grid = Grid.periodic(12, 14)
		x, y = np.meshgrid(grid.x, grid.y, indexing="ij")
		cls.potential = Potential(
			grid, mean=np.sin(x) * np.cos(y),
			modes=(0.2 * np.cos(x + y) + 0.1j * np.sin(x - y))[None],
			frequencies=np.array([0.7]),
		)
		cls.gc = GuidingCenterDynamics(cls.potential, rho=0.1)
		cls.fc = FullCyclotronDynamics(cls.potential, rho=0.3, eta=-0.4)

	@classmethod
	def tearDownClass(cls) -> None:
		cls.jax.config.update("jax_enable_x64", cls.previous_x64)

	def _state(self, dimension: int) -> np.ndarray:
		"""Two particles with non-grid coordinates, including a periodic image."""
		blocks = np.array([[0.23, 6.5], [1.14, -0.4], [0.3, -0.2], [-0.1, 0.5]])
		return blocks[:dimension].reshape(-1)

	def test_layout_round_trip_preserves_backend_and_sample_axes(self) -> None:
		for layout in (GCStateLayout(), FCStateLayout()):
			state = np.arange(layout.state_dimension * 12.0).reshape(-1, 3, 2)
			for value in (state, self.jnp.asarray(state)):
				with self.subTest(layout=type(layout).__name__, backend=type(value).__name__):
					self.assertIs(layout.validate_packed_state_layout(value), value)
					self.assertEqual(layout.particle_count(value), 2)
					blocks = layout.as_blocks(value)
					self.assertEqual(blocks.shape, (layout.state_dimension, 2, 3, 2))
					packed = layout.pack_components(*layout.split(value))
					self.assertIsInstance(packed, np.ndarray if isinstance(value, np.ndarray) else self.jax.Array)
					np.testing.assert_array_equal(packed, state)
					np.testing.assert_array_equal(layout.from_blocks(blocks), state)
			compiled = self.jax.jit(lambda value: layout.pack_components(*layout.split(value)))
			np.testing.assert_array_equal(compiled(self.jnp.asarray(state)), state)
			mixed = list(layout.split(state))
			mixed[-1] = self.jnp.asarray(mixed[-1])
			self.assertIsInstance(layout.pack_components(*mixed), self.jax.Array)

	def test_layout_rejects_invalid_shapes_during_tracing(self) -> None:
		layout = GCStateLayout()
		for state in (self.jnp.array(1.0), self.jnp.ones(3), self.jnp.ones((0, 2))):
			with self.subTest(shape=state.shape), self.assertRaisesRegex(ValueError, "first state dimension"):
				self.jax.jit(layout.as_blocks)(state)
		with self.assertRaisesRegex(ValueError, "same shape"):
			self.jax.jit(layout.pack_components)(self.jnp.ones(2), self.jnp.ones(3))

	def test_numpy_jax_and_mixed_time_agree_for_both_dynamics(self) -> None:
		for dynamics in (self.gc, self.fc):
			state = self._state(dynamics.state_dimension)
			operations = [dynamics.vector_field, dynamics.hamiltonian, dynamics.extended_momentum_derivative]
			if isinstance(dynamics, GuidingCenterDynamics):
				operations.append(dynamics.particle_vector_field_jacobians)
			for operation in operations:
				with self.subTest(dynamics=type(dynamics).__name__, operation=operation.__name__):
					expected = operation(0.17, state)
					self.assertIsInstance(expected, np.ndarray)
					for time, value in ((0.17, self.jnp.asarray(state)), (self.jnp.array(0.17), state)):
						actual = getattr(prepare_dynamics(dynamics), operation.__name__)(time, value)
						self.assertIsInstance(actual, self.jax.Array)
						np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
					compiled = self.jax.jit(getattr(prepare_dynamics(dynamics), operation.__name__))(self.jnp.array(0.17), self.jnp.asarray(state))
					np.testing.assert_allclose(compiled, expected, rtol=2e-11, atol=2e-12)

	def test_history_broadcasting_and_vmap_preserve_component_order(self) -> None:
		times = np.array([0.1, 0.23, 0.4])
		for dynamics in (self.gc, self.fc):
			state = self._state(dynamics.state_dimension)
			history = state[:, None] + np.arange(3)[None, :] * 0.03
			for operation in (dynamics.vector_field, dynamics.hamiltonian, dynamics.extended_momentum_derivative):
				with self.subTest(dynamics=type(dynamics).__name__, operation=operation.__name__):
					expected = np.stack([operation(t, column) for t, column in zip(times, history.T)], axis=1)
					device_operation = getattr(prepare_dynamics(dynamics), operation.__name__)
					direct = device_operation(self.jnp.asarray(times), self.jnp.asarray(history))
					mapped = self.jax.jit(self.jax.vmap(device_operation, in_axes=(0, 1), out_axes=1))(
						self.jnp.asarray(times), self.jnp.asarray(history),
					)
					np.testing.assert_allclose(direct, expected, rtol=2e-11, atol=2e-12)
					np.testing.assert_allclose(mapped, expected, rtol=2e-11, atol=2e-12)

	def test_energy_time_gradient_matches_extended_momentum_rate(self) -> None:
		for dynamics in (self.gc, self.fc):
			state = self._state(dynamics.state_dimension)
			# Only time is traced: NumPy coordinates must not force NumPy evaluation.
			derivative = self.jax.jit(self.jax.grad(lambda t: self.jnp.sum(prepare_dynamics(dynamics).hamiltonian(t, state))))(0.17)
			expected = -np.sum(dynamics.extended_momentum_derivative(0.17, state))
			np.testing.assert_allclose(derivative, expected, rtol=2e-11, atol=2e-12)

	def test_gc_analytic_jacobians_match_automatic_differentiation(self) -> None:
		state = self.jnp.asarray(self._state(2))
		full = self.jax.jit(self.jax.jacfwd(prepare_dynamics(self.gc).vector_field, argnums=1))(0.17, state)
		blocks = self.gc.particle_vector_field_jacobians(0.17, state)
		expected = np.zeros((4, 4))
		for particle in range(2):
			indices = np.array([particle, particle + 2])
			expected[np.ix_(indices, indices)] = np.asarray(blocks[particle])
		np.testing.assert_allclose(full, expected, rtol=2e-10, atol=2e-11)


if __name__ == "__main__":
	unittest.main()
