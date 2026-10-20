"""Unified snapshots, external generators and independent geometry contracts."""

from dataclasses import FrozenInstanceError
import importlib.util
import pickle
import unittest

import numpy as np

from states import State, GCState, GCStateLayout, FCState, FCStateLayout
from states.generators import circle, square
from diagnostics.geometry import calculate_area


class StateTests(unittest.TestCase):
	def test_snapshot_owns_values_components_and_mutable_layout(self) -> None:
		class MutableLayout(GCStateLayout):
			def __init__(self):
				self.shift = 0.0

			def positions(self, values):
				x, y = super().positions(values)
				return x + self.shift, y + self.shift

		values = np.array([1., 2., 3., 4.])
		layout = MutableLayout()
		state = State(values, layout=layout)
		values[:] = -1
		layout.shift = 5
		state.values[:] = -2
		state.initial_state[:] = -3
		state.split()[0][:] = -4
		state.positions()[0][:] = -5
		state.layout.shift = 6
		np.testing.assert_array_equal(state.values, [1, 2, 3, 4])
		np.testing.assert_array_equal(state.positions()[0], [1, 2])
		self.assertEqual(state.particle_count, 2)
		with self.assertRaises((FrozenInstanceError, TypeError)):
			state.values = values

	def test_one_class_supports_both_models_and_pickle(self) -> None:
		for layout, component_type in [(GCStateLayout(), GCState), (FCStateLayout(), FCState)]:
			components = tuple(np.arange(3.) + i for i in range(layout.state_dimension))
			state = State.from_components(*components, layout=layout)
			self.assertIs(type(state), State)
			self.assertIsInstance(state.split(), component_type)
			np.testing.assert_array_equal(state.values, np.concatenate(components))
			restored = pickle.loads(pickle.dumps(state))
			np.testing.assert_array_equal(restored.values, state.values)
			if layout.state_dimension == 4:
				np.testing.assert_array_equal(state.velocities(), components[2:])
			else:
				with self.assertRaisesRegex(TypeError, "velocity"):
					state.velocities()

	def test_rejects_empty_nonfinite_and_noninstantaneous_states(self) -> None:
		for values in ([], [1., 2., 3.], [np.nan, 0], [np.inf, 0], [[1., 2.]]):
			with self.subTest(values=values), self.assertRaises(ValueError):
				State(np.asarray(values), layout=GCStateLayout())
		with self.assertRaises(TypeError):
			State(np.array([1., 2.]), layout=object())
		with self.assertRaises(ValueError):
			State.from_components(np.ones((2, 3)), np.ones((2, 3)), layout=GCStateLayout())

	def test_layout_history_roundtrips_preserve_all_axes(self) -> None:
		for layout in (GCStateLayout(), FCStateLayout()):
			values = np.arange(layout.state_dimension * 2 * 3 * 4).reshape(-1, 3, 4)
			blocks = layout.as_blocks(values)
			self.assertEqual(blocks.shape, (layout.state_dimension, 2, 3, 4))
			np.testing.assert_array_equal(layout.from_blocks(blocks), values)
			np.testing.assert_array_equal(layout.pack_components(*layout.split(values)), values)

	@unittest.skipUnless(importlib.util.find_spec("jax"), "JAX is optional")
	def test_layout_operations_remain_traceable(self) -> None:
		import jax
		import jax.numpy as jnp
		for layout in (GCStateLayout(), FCStateLayout()):
			values = jnp.arange(layout.state_dimension * 6., dtype=jnp.float32).reshape(-1, 3)
			@jax.jit
			def roundtrip(value):
				return layout.from_blocks(layout.as_blocks(layout.pack_components(*layout.split(value))))
			actual = roundtrip(values)
			self.assertIsInstance(actual, jax.Array)
			np.testing.assert_array_equal(actual, values)

	def test_generators_return_plain_states_and_area_is_external(self) -> None:
		for contour in (circle(center=(0., 0.), radius=.5, points=16),
				square(center=(0., 0.), side=1., points_per_side=4)):
			self.assertIs(type(contour), State)
			self.assertFalse(hasattr(contour, "shape"))
			self.assertFalse(hasattr(contour, "calculate_area"))
			positive = calculate_area(contour.values, layout=contour.layout)
			x, y = contour.positions()
			reversed_values = GCStateLayout.pack_components(x[::-1], y[::-1])
			self.assertAlmostEqual(float(calculate_area(reversed_values, layout=contour.layout)), -float(positive))
		with self.assertRaises(ValueError):
			calculate_area(np.zeros(12), layout=FCStateLayout())
		with self.assertRaises(ValueError):
			calculate_area(np.zeros(4), layout=GCStateLayout())

	def test_retired_packages_are_absent(self) -> None:
		self.assertIsNone(importlib.util.find_spec("initial_conditions"))
		self.assertIsNone(importlib.util.find_spec("contracts.state_layout"))


if __name__ == "__main__":
	unittest.main()
