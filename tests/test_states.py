"""Physical recording lifecycle, sample ownership, layouts and external generators."""

import importlib.util
import pickle
import unittest
import numpy as np
from dynamics.coordinates import PhysicalCoordinates
from dynamics.gc import GuidingCenterDynamics
from dynamics.fc import FullCyclotronDynamics
from states import States
from execution.state import IntegrationState
from contracts.methods.step import StepInfo
from states.generators import circle, square
from diagnostics.states.geometry import calculate_area


class StateTests(unittest.TestCase):
	def test_snapshot_owns_values_components_and_mutable_layout(self) -> None:
		from tests.test_state_ownership import _ExternalGCLayout
		from dataclasses import FrozenInstanceError
		values = np.array([1., 2., 3., 4.]); layout = _ExternalGCLayout()
		state = States.prepare(np.array([.3,.7]), values, layout=layout)
		state.record(.3, values)
		values[:] = -1; layout.state_dimension = 7
		exposed = state.values
		with self.assertRaises(ValueError): exposed[:] = 0
		state.record(.7, np.array([2.,3.,4.,5.]))
		self.assertIs(state.finalize(), state)
		np.testing.assert_array_equal(state.values[:,0], [1,2,3,4])
		with self.assertRaises(AttributeError): state.layout.state_dimension = 7
		np.testing.assert_array_equal(state.positions()[0][:,0], [1,2])
		self.assertEqual(state.particle_count, 2)
		with self.assertRaises(RuntimeError): state.record(.7, values)
		with self.assertRaises(RuntimeError): state.finalize()

	def test_one_class_supports_both_models_and_pickle(self) -> None:
		for layout in (GuidingCenterDynamics.coordinates, FullCyclotronDynamics.coordinates):
			values = layout.pack_components(*(np.arange(3.)+i for i in range(layout.state_dimension)))
			state = States(np.array([5.]), values[:,None], layout=layout, initial_state=values)
			self.assertIs(type(state), States)
			self.assertIs(type(state.components()), tuple)
			restored = pickle.loads(pickle.dumps(state))
			np.testing.assert_array_equal(restored.values, state.values)
			self.assertFalse(restored.values.flags.writeable)
			self.assertFalse(restored.times.flags.writeable)
			if layout.state_dimension == 4:
				np.testing.assert_array_equal(state.velocities()[0][:,0], np.arange(3.)+2)
			else:
				with self.assertRaisesRegex(TypeError, 'velocity'): state.velocities()

	def test_rejects_empty_nonfinite_and_noninstantaneous_states(self) -> None:
		for values in ([], [np.nan,0], [np.inf,0], [[1.,2.]]):
			with self.subTest(values=values), self.assertRaises(ValueError):
				IntegrationState(0., np.asarray(values))
		for values in ([], [1.,2.,3.], [np.nan,0], [[1.,2.]]):
			with self.subTest(values=values), self.assertRaises(ValueError):
				States.prepare(np.array([0.,1.]), np.asarray(values), layout=GuidingCenterDynamics.coordinates)
		with self.assertRaises(ValueError): IntegrationState(np.nan, np.ones(2))

	def test_recording_does_not_publish_unfilled_columns_or_finalize_partial_result(self) -> None:
		result = States.prepare(np.array([2.,2.3,3.]), np.array([1.,2.]), layout=GuidingCenterDynamics.coordinates)
		self.assertEqual(result.values.shape, (2,0))
		with self.assertRaises(RuntimeError): result.finalize()
		with self.assertRaises(ValueError): result.record(2.3, np.ones(2))
		result.record(2., np.ones(2), auxiliary={'momentum':np.zeros(1)})
		with self.assertRaises(ValueError): result.record(2.3, np.ones(2))
		with self.assertRaises(ValueError): result.record(2.3, np.ones(3), auxiliary={'momentum':np.zeros(1)})
		self.assertEqual(result.times.tolist(), [2.])

	def test_bulk_recording_owns_history_and_keeps_finalization_explicit(self) -> None:
		times = np.array([.3, .5, .9])
		values = np.array([[1., 1.5, 2.], [2., 2.5, 3.]])
		for layout in (GuidingCenterDynamics.coordinates, FullCyclotronDynamics.coordinates):
			history = np.tile(values, (layout.state_dimension // 2, 1))
			original = history.copy()
			result = States.prepare(times, history[:, 0], layout=layout)
			result.record_history(history)
			history[:] = -999
			self.assertFalse(result.complete)
			np.testing.assert_array_equal(result.values, original)
			with self.assertRaises(TypeError): pickle.dumps(result)
			self.assertIs(result.finalize({'step_count': 2}), result)
			self.assertFalse(result.values.flags.writeable)
			with self.assertRaises(RuntimeError): result.record_history(original)

	def test_invalid_bulk_recording_leaves_the_recorder_empty(self) -> None:
		times = np.array([0., .5, 1.])
		result = States.prepare(times, np.array([1., 2.]), layout=GuidingCenterDynamics.coordinates)
		for invalid in (np.zeros((3, 3)), np.zeros((2, 2)), np.zeros((6,)), np.full((2, 3), np.nan)):
			with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
				result.record_history(invalid)
			self.assertEqual(result.values.shape, (2, 0))
			self.assertFalse(result.complete)
		result.record(0., np.array([1., 2.]))
		with self.assertRaisesRegex(RuntimeError, "empty"):
			result.record_history(np.ones((2, 3)))
		np.testing.assert_array_equal(result.values, [[1.], [2.]])

	def test_sample_and_main_step_axes_remain_independent(self) -> None:
		values=np.array([1.,2.]); result=States.prepare(np.array([0.,.03,.07,.1]),values,layout=GuidingCenterDynamics.coordinates)
		for time in [0.,.03,.07,.1]: result.record(time, values+time)
		metric=np.array([3.]);result.record_step(StepInfo(0,0.,.1,.1,values),{'work':metric})
		metric[:]=-1
		result.finalize()
		self.assertEqual(result.n_steps,1)
		self.assertEqual(result.values.shape,(2,4))
		np.testing.assert_array_equal(result.diagnostics['work'],[[3.]])
		with self.assertRaises(ValueError): result.diagnostics['work'][:]=0
		np.testing.assert_array_equal(result.select([1,3]).times,[.03,.1])

	def test_integer_initial_values_do_not_truncate_physical_samples(self) -> None:
		result = States.prepare(np.array([0., 1.]), np.array([1, 2]), layout=GuidingCenterDynamics.coordinates)
		result.record(0., np.array([1, 2]))
		result.record(1., np.array([1.5, 2.5]))
		result.finalize()
		np.testing.assert_array_equal(result.values[:, 1], [1.5, 2.5])

	def test_internal_snapshot_does_not_alias_input(self) -> None:
		values=np.arange(6.);current=IntegrationState(.3,values);values[:]=-1
		np.testing.assert_array_equal(current.values,np.arange(6.))
		with self.assertRaises(ValueError): current.values[:]=0

	def test_layout_history_roundtrips_preserve_all_axes(self) -> None:
		for layout in (GuidingCenterDynamics.coordinates, FullCyclotronDynamics.coordinates):
			values=np.arange(layout.state_dimension*2*3*4).reshape(-1,3,4)
			blocks=layout.as_blocks(values)
			self.assertEqual(blocks.shape,(layout.state_dimension,2,3,4))
			np.testing.assert_array_equal(layout.from_blocks(blocks),values)
			np.testing.assert_array_equal(layout.pack_components(*layout.split(values)),values)

	@unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX is optional')
	def test_layout_operations_remain_traceable(self) -> None:
		import jax
		import jax.numpy as jnp
		from dynamics.jax_dynamics import JaxGuidingCenterDynamics, JaxFullCyclotronDynamics
		for layout in (JaxGuidingCenterDynamics.coordinates, JaxFullCyclotronDynamics.coordinates):
			values=jnp.arange(layout.state_dimension*6.,dtype=jnp.float32).reshape(-1,3)
			@jax.jit
			def roundtrip(value):return layout.from_blocks(layout.as_blocks(layout.pack_components(*layout.split(value))))
			actual=roundtrip(values)
			self.assertIsInstance(actual,jax.Array)
			np.testing.assert_array_equal(actual,values)

	def test_generators_return_plain_states_and_area_is_external(self) -> None:
		layout=GuidingCenterDynamics.coordinates
		for contour in (circle(center=(0.,0.),radius=.5,points=16),square(center=(0.,0.),side=1.,points_per_side=4)):
			self.assertIs(type(contour),np.ndarray)
			positive=calculate_area(contour,layout=layout)
			x,y=layout.positions(contour)
			reversed_values=layout.pack_components(x[::-1],y[::-1])
			self.assertAlmostEqual(float(calculate_area(reversed_values,layout=layout)),-float(positive))
		with self.assertRaises(ValueError):calculate_area(np.zeros(12),layout=FullCyclotronDynamics.coordinates)
		with self.assertRaises(ValueError):calculate_area(np.zeros(4),layout=layout)

	def test_retired_packages_are_absent(self) -> None:
		for name in ('initial_conditions','contracts.state_layout','states.state','states.layout','solution','contracts.request'):
			with self.subTest(name=name): self.assertIsNone(importlib.util.find_spec(name))
