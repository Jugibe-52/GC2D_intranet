"""Physical coordinate identity and independently owned auxiliary snapshots."""
from dynamics import GuidingCenterDynamics
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from dynamics.coordinates import PhysicalCoordinates, GCCoordinates, FCCoordinates, describe_coordinates
from tests.coordinate_fixtures import ThreeCoordinates, FiveCoordinates, UnclassifiedCoordinates, ReorderedCoordinates
from dynamics.fc import FullCyclotronDynamics
from execution.state import IntegrationState
from diagnostics.methods.energy import EnergyObserver, momentum_increment
from execution._host import FixedStepController
from simulation import InitialConditions, InitialValueProblem, RK4, ABBA2Implicit, SimulationRequest, simulate
from tests.test_method_integration import _problem


class ExternalThreeComponentSystem:
	state_dimension = 3
	coordinates = ThreeCoordinates()

	def vector_field(self, t, state):
		return np.ones_like(state)


class PhysicalWorkspaceTests(unittest.TestCase):
	def test_fixed_coordinates_have_no_configurable_fields(self) -> None:
		from dataclasses import is_dataclass
		with self.assertRaises(TypeError):
			PhysicalCoordinates()
		for cls, dimension in ((GCCoordinates, 2), (FCCoordinates, 4)):
			layout = cls()
			self.assertEqual(layout.state_dimension, dimension)
			self.assertFalse(is_dataclass(layout))
			self.assertFalse(hasattr(layout, '__dict__'))
			for field in ('component_names', 'position_names', 'velocity_names', 'position_indices', 'velocity_indices'):
				self.assertFalse(hasattr(layout, field))
				with self.assertRaises(TypeError):
					cls(**{field: ()})
			with self.assertRaises(TypeError):
				cls(dimension)
			with self.assertRaises(AttributeError):
				layout.state_dimension = 7
			self.assertEqual(pickle.loads(pickle.dumps(layout)), layout)

	def test_fixed_roles_and_plain_component_views(self) -> None:
		for layout in (GCCoordinates(), FCCoordinates()):
			state = np.arange(layout.state_dimension * 2.)
			parts = layout.split(state)
			self.assertIs(type(parts), tuple)
			self.assertTrue(all(np.shares_memory(part, state) for part in parts))
			np.testing.assert_array_equal(layout.pack_components(*parts), state)
			np.testing.assert_array_equal(layout.positions(state), parts[:2])
			if isinstance(layout, FCCoordinates):
				np.testing.assert_array_equal(layout.velocities(state), parts[2:])
			else:
				with self.assertRaisesRegex(ValueError, 'velocity'):
					layout.velocities(state)

	def test_external_snapshot_must_explicitly_preserve_its_convention(self) -> None:
		from tests.test_state_ownership import _ExternalGCLayout
		source = _ExternalGCLayout()
		owned = describe_coordinates(source)
		self.assertIs(type(owned), GCCoordinates)
		source.state_dimension = 3
		with self.assertRaisesRegex(ValueError, 'preserve'):
			describe_coordinates(source)
		self.assertEqual(owned.state_dimension, 2)
		source.state_dimension = 2
		source.snapshot = lambda: source
		with self.assertRaisesRegex(TypeError, 'NumPy PhysicalCoordinates'):
			describe_coordinates(source)

	def test_method_guards_require_explicit_gc_or_fc_conventions(self) -> None:
		from methods.extended.maps import GCDoubledMaps
		from methods.splitting import FCSplitMaps
		for layout in (UnclassifiedCoordinates(), ReorderedCoordinates(), FCCoordinates()):
			with self.subTest(layout=type(layout)), self.assertRaises(TypeError):
				GCDoubledMaps(lambda t, z: z, 2, coordinates=layout)
		for layout in (GCCoordinates(), UnclassifiedCoordinates(), FiveCoordinates()):
			with self.subTest(layout=type(layout)), self.assertRaises(TypeError):
				FCSplitMaps(lambda t, z, h: z, lambda t, z, h: z, 4, coordinates=layout)

	def test_external_coordinates_and_explicit_layout_do_not_infer_a_model(self):
		for count in (1, 3):
			dynamics = ExternalThreeComponentSystem()
			initial = np.arange(3 * count, dtype=float)
			problem = InitialValueProblem(dynamics, initial)
			result = simulate(problem, RK4(step=.02), SimulationRequest((.15, .255), np.array([.15, .153, .255])))
			np.testing.assert_allclose(result.values, initial[:, None] + result.times - .15, rtol=0, atol=4e-15)
			self.assertEqual(result.layout.state_dimension, 3)
			self.assertIs(type(result.layout), ThreeCoordinates)
			self.assertEqual(result.diagnostics['accepted_internal_state_dimension'], initial.size)
			class Undeclared:
				state_dimension = 3
				def vector_field(self, t, z): return np.ones_like(z)
			with self.assertRaisesRegex(TypeError, 'explicit layout'):
				InitialValueProblem(Undeclared(), initial)
			self.assertEqual(InitialValueProblem(Undeclared(), initial, layout=dynamics.coordinates).particle_count, count)
		with self.assertRaises(ValueError): dynamics.coordinates.split(np.zeros(4))

	def test_same_dimension_does_not_establish_coordinate_compatibility(self):
		problem = _problem()
		for layout in (ReorderedCoordinates(),
		               UnclassifiedCoordinates()):
			with self.assertRaises(TypeError):
				InitialValueProblem(problem.dynamics, problem.initial_state, layout=layout)
			with self.assertRaises(TypeError):
				InitialValueProblem(problem.dynamics, InitialConditions(problem.initial_state, layout=layout))
		with self.assertRaises(AttributeError): problem.layout.state_dimension = 7

	def test_role_accessors_and_sample_axes_are_generic(self):
		layout = FiveCoordinates()
		physical = np.arange(5 * 3 * 2 * 4).reshape(15, 2, 4)
		components = layout.split(physical)
		np.testing.assert_array_equal(layout.positions(physical)[0], components[1])
		np.testing.assert_array_equal(layout.velocities(physical)[0], components[3])
		np.testing.assert_array_equal(layout.pack_components(*components), physical)
		np.testing.assert_array_equal(pickle.loads(pickle.dumps(components))[1], components[1])

	def test_archive_load_needs_no_dynamics_or_field_construction(self):
		from diagnostics.io.persistence import save_solution
		from states import States
		with tempfile.TemporaryDirectory() as directory:
			for name, layout in (('gc', GuidingCenterDynamics.coordinates), ('fc', FullCyclotronDynamics.coordinates)):
				values = np.arange(layout.state_dimension * 3 * 2).reshape(-1, 2)
				save_solution(States(np.array([.1, .2]), values, layout=layout), Path(directory) / name, metadata={})
			script = '''
import sys
from unittest.mock import patch
from dynamics import GuidingCenterDynamics, FullCyclotronDynamics
from potential import Potential
from diagnostics.io.persistence import load_solution
from pathlib import Path
with patch.object(GuidingCenterDynamics, '__init__', side_effect=AssertionError('GC construction')), patch.object(FullCyclotronDynamics, '__init__', side_effect=AssertionError('FC construction')), patch.object(Potential, '__init__', side_effect=AssertionError('field construction')), patch.object(Potential, 'load', side_effect=AssertionError('field loading')):
    for name in ('gc', 'fc'):
        stored = load_solution(Path(sys.argv[1]) / name)
        assert stored.potential is None
        assert stored.solution.particle_count == 3
        assert stored.solution.positions()[0].shape == (3, 2)
        if name == 'fc': assert stored.solution.velocities()[0].shape == (3, 2)
'''
			result = subprocess.run([sys.executable, '-c', script, directory], capture_output=True, text=True)
			self.assertEqual(result.returncode, 0, result.stderr)

	def test_shadow_samples_preserve_accepted_nonzero_momentum(self):
		problem = _problem(); request = SimulationRequest((.15, .255), np.array([.15, .153, .181, .255]))
		for cls in (RK4, ABBA2Implicit):
			run = cls(step=.02).new_run(problem.dynamics.vector_field, problem.initial_conditions, request,
				jacobian=problem.dynamics.particle_vector_field_jacobians, jacobian_structure='particle', capture_quadrature=True)
			initial_momentum = np.full(problem.particle_count, .7)
			energy = EnergyObserver(initial_momentum=initial_momentum).new_run(
				problem.dynamics.hamiltonian, problem.dynamics.extended_momentum_derivative,
				particle_count=problem.particle_count, initial_time=.15)
			controller = FixedStepController()
			info, result = next(controller.steps(run, request))
			before = result.state.copy()
			energy.accept(info.index, info.time, info.end_time, result.quadrature)
			momentum_before = energy.momentum
			for time in (.153, .159, .153):
				sample = controller.sample(run, info, result, time)
				expected = run.advance(info.time, info.state_before, time-info.time)
				np.testing.assert_array_equal(sample.values, expected.state)
				observed = energy.sample(time, sample.quadrature)
				np.testing.assert_array_equal(observed, initial_momentum + momentum_increment(
					problem.dynamics.extended_momentum_derivative, expected.quadrature, problem.particle_count))
				np.testing.assert_array_equal(result.state, before)
				np.testing.assert_array_equal(energy.momentum, momentum_before)
				observed[:] = 99
				np.testing.assert_array_equal(energy.momentum, momentum_before)
			with self.assertRaises(ValueError): run.advance(.15, np.tile(run.initial_state, 2), .02)

	def test_snapshots_copy_and_serialize_physical_arrays(self):
		z=np.array([1.,2.]); state=IntegrationState(.3,z)
		z[:]=99
		for value in (state,pickle.loads(pickle.dumps(state))):
			np.testing.assert_array_equal(value.values,[1,2])
			self.assertFalse(value.values.flags.writeable)
			self.assertFalse(hasattr(value, 'momentum'))

	def test_adaptive_step_result_does_not_expose_live_solver_state(self):
		from methods import DOP853, Radau
		problem = _problem()
		request = SimulationRequest((.15, .255), np.array([.15, .255]))
		for cls in (DOP853, Radau):
			run = cls(max_step=.02).new_run(problem.dynamics.vector_field, problem.initial_conditions, request, capture_quadrature=True)
			result = run.advance(.15, run.initial_state, .02)
			physical = run.solver.y.copy()
			self.assertFalse(np.shares_memory(result.state, run.solver.y))
			result.state[:] = 99
			np.testing.assert_array_equal(run.solver.y, physical)
			np.testing.assert_array_equal(run.current_state.values, physical)
