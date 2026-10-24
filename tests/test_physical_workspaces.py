"""Physical coordinate identity and independently owned auxiliary snapshots."""
from dataclasses import FrozenInstanceError
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from dynamics.coordinates import PhysicalCoordinates
from dynamics.gc import GuidingCenterDynamics
from dynamics.fc import FullCyclotronDynamics
from execution.state import IntegrationState
from execution._host import FixedStepController
from simulation import InitialConditions, InitialValueProblem, RK4, ABBA2Implicit, SimulationRequest, simulate
from tests.test_method_integration import _problem


class ExternalThreeComponentSystem:
	state_dimension = 3
	coordinates = PhysicalCoordinates(('q', 'r', 'v'), ('q', 'r'))

	def vector_field(self, t, state):
		return np.ones_like(state)


class PhysicalWorkspaceTests(unittest.TestCase):
	def test_external_coordinates_and_explicit_layout_do_not_infer_a_model(self):
		for count in (1, 3):
			dynamics = ExternalThreeComponentSystem()
			initial = np.arange(3 * count, dtype=float)
			problem = InitialValueProblem(dynamics, initial)
			result = simulate(problem, RK4(step=.02), SimulationRequest((.15, .255), np.array([.15, .153, .255])))
			np.testing.assert_allclose(result.values, initial[:, None] + result.times - .15, rtol=0, atol=4e-15)
			self.assertEqual(result.layout.component_names, ('q', 'r', 'v'))
			self.assertIs(type(result.layout), PhysicalCoordinates)
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
		for layout in (PhysicalCoordinates(('y', 'x'), ('x', 'y')),
		               PhysicalCoordinates(('x', 'y'), (), ('x', 'y'))):
			with self.assertRaises(TypeError):
				InitialValueProblem(problem.dynamics, problem.initial_state, layout=layout)
			with self.assertRaises(TypeError):
				InitialValueProblem(problem.dynamics, InitialConditions(problem.initial_state, layout=layout))
		with self.assertRaises(FrozenInstanceError): problem.layout.component_names = ('y', 'x')

	def test_role_accessors_and_sample_axes_are_generic(self):
		layout = PhysicalCoordinates(('b', 'a', 'w', 'u', 's'), ('a', 'b'), ('u', 'w'))
		physical = np.arange(5 * 3 * 2 * 4).reshape(15, 2, 4)
		components = layout.split(physical)
		np.testing.assert_array_equal(layout.positions(physical)[0], components.a)
		np.testing.assert_array_equal(layout.velocities(physical)[0], components.u)
		np.testing.assert_array_equal(layout.pack_components(*components), physical)
		np.testing.assert_array_equal(pickle.loads(pickle.dumps(components)).a, components.a)

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
			run = cls(step=.02, track_energy=True).new_run(problem, request)
			controller = FixedStepController()
			info, result = next(controller.steps(run, request))
			before = result.state.copy(); momentum_before = result.momentum.copy()
			for time in (.153, .159):
				sample = controller.sample(run, info, result, time)
				expected = run.advance(info.time, info.state_before, time-info.time, momentum=info.momentum_before)
				np.testing.assert_array_equal(sample.values, expected.state)
				np.testing.assert_array_equal(sample.momentum, expected.momentum)
				np.testing.assert_array_equal(result.state, before)
				np.testing.assert_array_equal(result.momentum, momentum_before)
				self.assertFalse(np.shares_memory(sample.momentum, result.momentum))
			shifted = run.advance(.15, run.initial_state, .02, momentum=run.initial_momentum + .7)
			np.testing.assert_allclose(shifted.momentum, result.momentum+.7, rtol=0, atol=1e-16)
			np.testing.assert_array_equal(shifted.state, result.state)
			with self.assertRaises(ValueError): run.advance(.15, np.tile(run.initial_state, 2), .02, momentum=run.initial_momentum)
			with self.assertRaises(ValueError): run.advance(.15, run.initial_state, .02)

	def test_snapshots_copy_and_serialize_both_arrays(self):
		z=np.array([1.,2.]); k=np.array([.4]); state=IntegrationState(.3,z,k)
		z[:]=99; k[:]=88
		for value in (state,pickle.loads(pickle.dumps(state))):
			np.testing.assert_array_equal(value.values,[1,2]); np.testing.assert_array_equal(value.momentum,[.4])
			self.assertFalse(value.values.flags.writeable); self.assertFalse(value.momentum.flags.writeable)

	def test_adaptive_step_result_does_not_expose_live_solver_state(self):
		from methods import DOP853, Radau
		problem = _problem()
		request = SimulationRequest((.15, .255), np.array([.15, .255]))
		for cls in (DOP853, Radau):
			run = cls(max_step=.02, track_energy=True).new_run(problem, request)
			result = run.advance(.15, run.initial_state, .02, momentum=run.initial_momentum)
			physical = run.solver.y.copy()
			self.assertFalse(np.shares_memory(result.state, run.solver.y))
			result.state[:] = 99
			np.testing.assert_array_equal(run.solver.y, physical)
			np.testing.assert_array_equal(run.current_state.values, physical)
