"""Function-only Runge--Kutta preparation and generic derivative structures."""
from dataclasses import replace
import unittest
import numpy as np
from scipy.linalg import expm

from contracts.methods.step import StepInfo
from tests.coordinate_fixtures import ThreeCoordinates
from methods.runge_kutta.euler import ExplicitEuler
from methods.runge_kutta.rk4 import RK4
from methods.runge_kutta.gauss_legendre import GaussLegendre4
from methods.runge_kutta.sdirk import SDIRK4
from methods.hbvm.order4 import HBVM42
from request import SimulationRequest
from solvers.derivatives import packed_particle_jacobian
from solvers.nonlinear import NewtonSolver
from states.initial import InitialConditions


class FunctionMethodTests(unittest.TestCase):
	def setUp(self):
		self.initial = InitialConditions([1., -.2, .4], layout=ThreeCoordinates())
		self.request = SimulationRequest((.15, .17), np.array([.15, .17]))
		self.matrix = np.array([[-.4, .2, .1], [.3, -.5, .2], [-.1, .2, -.3]])

	def test_particle_analytic_functions_on_three_component_coordinates(self):
		for cls in (GaussLegendre4, SDIRK4, HBVM42):
			with self.subTest(method=cls.__name__):
				run = cls(step=.02).new_run(lambda t, z: self.matrix @ z, self.initial, jacobian=lambda t, z: self.matrix[None, :, :])
				actual = run.advance(.15, self.initial.values, .02)
				np.testing.assert_allclose(actual.state, expm(.02 * self.matrix) @ self.initial.values,
					rtol=0, atol=2e-12)
				for name in ('problem', 'dynamics', 'evaluated_dynamics', '_energy_dynamics', 'state_formulation'):
					self.assertFalse(hasattr(run, name))

	def test_particle_blocks_support_three_components_without_coupling_assumptions(self):
		blocks = np.stack((self.matrix, self.matrix * .7))
		dense = packed_particle_jacobian(blocks)
		initial = InitialConditions([1., .7, -.2, .5, .4, -.1], layout=self.initial.layout)
		for cls in (GaussLegendre4, SDIRK4, HBVM42):
			method = cls(step=.02)
			particle = method.new_run(lambda t, z: dense @ z, initial,
				jacobian=lambda t, z: blocks)
			coupled = method.new_run(lambda t, z: dense @ z, initial)
			np.testing.assert_allclose(particle.advance(.15, initial.values, .02).state,
				coupled.advance(.15, initial.values, .02).state, rtol=0, atol=2e-12)

	def test_initial_convergence_does_not_evaluate_analytic_derivatives(self):
		def forbidden(t, z):
			raise AssertionError('A converged initial iterate must not request derivatives.')
		for cls in (GaussLegendre4, SDIRK4, HBVM42):
			method = cls(solver=NewtonSolver(jacobian_method='analytic', backtracking=cls is HBVM42))
			run = method.new_run(lambda t, z: np.zeros_like(z), self.initial,
				jacobian=forbidden)
			result = run.advance(.15, self.initial.values, .02)
			self.assertEqual(result.statistics['nonlinear_iterations'], 0)
			np.testing.assert_array_equal(result.state, self.initial.values)

	def test_hbvm_missing_analytic_function_is_checked_after_first_residual(self):
		calls = []
		def field(t, z):
			calls.append(t)
			return -z
		run = HBVM42(solver=NewtonSolver(jacobian_method='analytic', backtracking=True)).new_run(
			field, self.initial)
		self.assertEqual(calls, [])
		with self.assertRaises(TypeError):
			run.advance(.15, self.initial.values, .02)
		self.assertEqual(len(calls), 9)


	def test_solver_observation_is_silent_during_diagnostic_replay(self):
		for cls in (GaussLegendre4, SDIRK4, HBVM42):
			events = []
			method = cls(solver=NewtonSolver(iteration_observer=events.append, backtracking=cls is HBVM42))
			run = method.new_run(lambda t, z: -z, self.initial,
				jacobian=lambda t, z: -np.eye(z.size)[None, :, :])
			result = run.advance(.15, self.initial.values, .02)
			self.assertGreater(len(events), 0)
			count = len(events)
			event = run.build_observation(StepInfo(0, .15, .02, .17, self.initial.values), result)
			np.testing.assert_array_equal(event.map_state(self.initial.values), result.state)
			self.assertEqual(len(events), count)
			run.advance(.15, self.initial.values, .02, observe=False)
			self.assertEqual(len(events), count)

	def test_repeated_preparation_owns_independent_initial_values(self):
		method = RK4()
		first = method.new_run(lambda t, z: -z, self.initial)
		second = method.new_run(lambda t, z: -2*z, self.initial)
		self.assertFalse(hasattr(first, 'initial_state'))
		self.assertFalse(hasattr(second, 'initial_state'))
		self.assertFalse(hasattr(method, 'f'))
		self.assertFalse(np.array_equal(first.advance(.15, self.initial.values, .02).state,
			second.advance(.15, self.initial.values, .02).state))


if __name__ == '__main__':
	unittest.main()
