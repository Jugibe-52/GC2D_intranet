"""Tableau ownership, inherited recipes and independently solved stage equations."""

from dataclasses import FrozenInstanceError
import copy
import importlib.util
import pickle
import unittest

import numpy as np

from methods.implicit import ImplicitIntegrationMethod
from contracts.methods.step import StepInfo
from methods.runge_kutta import (
	ButcherTableau, RungeKutta, ExplicitRungeKutta,
	DiagonallyImplicitRungeKutta, ImplicitRungeKutta,
	ExplicitEuler, RK4, SDIRK4, GaussLegendre4,
)
from solvers.derivatives import packed_particle_jacobian
from solvers.nonlinear import NewtonSolver
from states.initial import InitialConditions


class RungeKuttaTests(unittest.TestCase):
	def setUp(self):
		self.initial = InitialConditions.gc(x=[.3, -.4], y=[.6, .2])
		self.heun = ButcherTableau([[0., 0.], [1., 0.]], [.5, .5], [0., 1.])
		self.midpoint = ButcherTableau([[0., 0.], [.5, 0.]], [0., 1.], [0., .5])
		self.coupled = ButcherTableau([[.2, .05, -.05], [.1, .25, .05], [.3, .2, .25]],
			[.2, .3, .5], [.2, .4, .75])

	def test_coefficients_are_owned_immutable_hashable_and_serializable(self):
		matrix = self.heun.A.copy()
		tableau = ButcherTableau(matrix, self.heun.B, self.heun.C)
		matrix[1, 0] = 99.
		for value in (tableau, copy.deepcopy(tableau), pickle.loads(pickle.dumps(tableau))):
			self.assertEqual(value, self.heun)
			self.assertEqual(hash(value), hash(self.heun))
			for array in (value.A, value.B, value.C):
				with self.assertRaises(ValueError):
					array.setflags(write=True)
		with self.assertRaises(FrozenInstanceError):
			tableau.A = matrix

	def test_invalid_tables_and_family_mismatches_fail_before_preparation(self):
		for A, B, C in (([], [], []), ([[0.]], [[1.]], [0.]), ([[0.]], [1.], [0., 1.]),
			([[np.nan]], [1.], [0.]), ([[1j]], [1.], [0.]), ([[0.]], [np.inf], [0.]),
			([[0.]], [1.], [1.]), ([[0.]], [.8], [0.]),
			([[0., 0.], [0., 0.]], [1e308, 1e308], [0., 0.])):
			with self.subTest(A=A, B=B, C=C), self.assertRaises(ValueError):
				ButcherTableau(A, B, C)
		with self.assertRaises(TypeError):
			ExplicitRungeKutta(tableau='RK4')
		with self.assertRaisesRegex(ValueError, 'strictly lower'):
			ExplicitRungeKutta(tableau=SDIRK4().tableau)
		with self.assertRaisesRegex(ValueError, 'lower-triangular'):
			DiagonallyImplicitRungeKutta(tableau=self.coupled)

	def test_recipes_inherit_all_stage_calculations(self):
		for recipe, family, order in ((ExplicitEuler, ExplicitRungeKutta, 1),
			(RK4, ExplicitRungeKutta, 4), (SDIRK4, DiagonallyImplicitRungeKutta, 4),
			(GaussLegendre4, ImplicitRungeKutta, 4)):
			self.assertIs(recipe.advance, RungeKutta.advance)
			self.assertIs(recipe._solve_stages, family._solve_stages)
			self.assertEqual(recipe.designed_order, order)
			for name in ('advance', '_solve_stages', 'initialize'):
				self.assertNotIn(name, recipe.__dict__)
			self.assertIsInstance(recipe(), family)
		self.assertIsNone(importlib.util.find_spec('methods.classical'))
		self.assertIsNone(importlib.util.find_spec('contracts.methods.classical'))

	def test_explicit_midpoint_has_correct_times_quadrature_and_second_order(self):
		method = ExplicitRungeKutta(tableau=self.midpoint)
		calls = []
		def field(t, z):
			calls.append(t)
			return t * z
		run = method.new_run(field, self.initial)
		result = run.advance(.3, self.initial.values, .1)
		expected = self.initial.values * (1 + .1 * .35 * (1 + .05 * .3))
		np.testing.assert_allclose(result.state, expected, rtol=0., atol=1e-15)
		np.testing.assert_allclose(calls, [.3, .35], rtol=0., atol=1e-15)
		np.testing.assert_allclose(result.quadrature.times, calls)
		np.testing.assert_allclose(result.quadrature.weights, [0., .1])
		self.assertNotIn('designed_order', run.metadata)
		np.testing.assert_array_equal(run.metadata['tableau_A'], self.midpoint.A)
		errors = []
		for n in (10, 20, 40):
			z = self.initial.values
			for i in range(n):
				z = run.advance(i / n, z, 1 / n).state
			errors.append(np.max(abs(z - self.initial.values * np.exp(.5))))
		self.assertTrue(all(3.5 < a / b < 4.5 for a, b in zip(errors, errors[1:])))

	def test_dirk_zero_diagonal_and_different_diagonals_use_actual_coefficients(self):
		tableau = ButcherTableau([[0., 0., 0.], [.25, .25, 0.], [.2, .3, .5]],
			[.2, .3, .5], [0., .5, 1.])
		events = []
		method = DiagonallyImplicitRungeKutta(tableau=tableau,
			solver=NewtonSolver(iteration_observer=events.append))
		run = method.new_run(lambda t, z: -z, self.initial, jacobian=lambda t, z: np.broadcast_to(-np.eye(2), (z.size // 2, 2, 2)))
		z = self.initial.values; h = .2
		stage2 = (z - h * .25 * z) / (1 + h * .25)
		stage3 = (z - h * (.2 * z + .3 * stage2)) / (1 + h * .5)
		result = run.advance(.3, z, h)
		np.testing.assert_allclose(result.quadrature.states, [z, stage2, stage3], rtol=0., atol=1e-14)
		np.testing.assert_allclose(result.state, stage3, rtol=0., atol=1e-14)
		self.assertEqual(run.metadata['nonlinear_solves_per_step'], 2)
		self.assertEqual(result.statistics['stage_residual_evaluations'][0], 0)
		self.assertEqual({e.stage_index for e in events}, {1, 2})
		observation = run.build_observation(StepInfo(0, .3, h, .5, z), result)
		count = len(events)
		np.testing.assert_allclose(observation.map_state(z), result.state, rtol=0., atol=1e-14)
		self.assertEqual(len(events), count)

	def test_three_coupled_stages_match_independent_linear_block_solution(self):
		blocks = np.array([[[-.7, .2], [-.3, -.2]], [[-.1, .4], [-.2, -.6]]])
		matrix = packed_particle_jacobian(blocks)
		q = np.array([.2, -.3, .4, .5]); z = self.initial.values; h = .17; time = .3
		times = time + h * self.coupled.C
		forcing = (times[:, None] * q).reshape(-1)
		stages = np.linalg.solve(np.eye(12) - h * np.kron(self.coupled.A, matrix),
			np.tile(z, 3) + h * np.kron(self.coupled.A, np.eye(4)) @ forcing).reshape(3, 4)
		expected = z + h * self.coupled.B @ (stages @ matrix.T + times[:, None] * q)
		for jac in (lambda t, z: blocks, None):
			run = ImplicitRungeKutta(tableau=self.coupled).new_run(lambda t, z: matrix @ z + t * q,
				self.initial, jacobian=jac)
			result = run.advance(time, z, h)
			np.testing.assert_allclose(result.quadrature.states, stages, rtol=0., atol=2e-14)
			np.testing.assert_allclose(result.state, expected, rtol=0., atol=2e-14)

	@unittest.skipUnless(importlib.util.find_spec('jax'), 'Optional JAX is not installed')
	def test_custom_families_compile_and_coefficients_separate_cache_signatures(self):
		import jax
		import jax.numpy as jnp
		from execution._jax_step import prepare_step
		previous = jax.config.read('jax_enable_x64')
		jax.config.update('jax_enable_x64', True)
		self.addCleanup(jax.config.update, 'jax_enable_x64', previous)
		methods = (ExplicitRungeKutta(tableau=self.heun), ExplicitRungeKutta(tableau=self.midpoint),
			DiagonallyImplicitRungeKutta(tableau=ButcherTableau([[0., 0.], [.5, .5]], [.5, .5], [0., 1.])),
			ImplicitRungeKutta(tableau=self.coupled))
		signatures = []
		def device_field(t, z):
			return t - z
		def device_jacobian(t, z):
			return jnp.broadcast_to(-jnp.eye(2), (z.size // 2, 2, 2))
		for method in methods:
			if isinstance(method, ImplicitIntegrationMethod):
				cpu = method.new_run(lambda t, z: t - z, self.initial, jacobian=lambda t, z: np.broadcast_to(-np.eye(2), (z.size // 2, 2, 2)))
				device = method.new_run(device_field, self.initial, jacobian=device_jacobian)
			else:
				cpu = method.new_run(lambda t, z: t - z, self.initial)
				device = method.new_run(device_field, self.initial)
			kernel = prepare_step(device, jax.devices('cpu')[0], initial_conditions=self.initial)
			signatures.append(kernel.signature)
			actual = jax.jit(kernel)(.3, jnp.asarray(self.initial.values), .17)
			expected = cpu.advance(.3, self.initial.values, .17)
			np.testing.assert_allclose(actual[0], expected.state, rtol=0., atol=2e-14)
			self.assertTrue(actual[2])
		self.assertNotEqual(signatures[0], signatures[1])
		self.assertEqual(len(set(signatures)), len(methods))
		repeated = methods[0].new_run(device_field, self.initial)
		self.assertEqual(prepare_step(repeated, jax.devices('cpu')[0], initial_conditions=self.initial).signature, signatures[0])

	@unittest.skipUnless(importlib.util.find_spec('jax'), 'Optional JAX is not installed')
	def test_custom_families_complete_host_and_device_jobs_with_energy_and_shadow_samples(self):
		import jax
		from dynamics.gc import GuidingCenterDynamics
		from diagnostics.methods.energy import EnergyObserver
		from execution.options import ExecutionOptions
		from potential import Potential
		from problem import InitialValueProblem
		from request import SimulationRequest
		from simulation.runner import simulate
		previous = jax.config.read('jax_enable_x64')
		jax.config.update('jax_enable_x64', True)
		self.addCleanup(jax.config.update, 'jax_enable_x64', previous)
		potential = Potential.random(A=.01, M=2, nx=12, ny=12, seed=27)
		problem = InitialValueProblem(GuidingCenterDynamics(potential), self.initial,
			request=SimulationRequest((.3, .8), np.array([.3, .34, .47, .68, .8])))
		for method in (ExplicitRungeKutta(tableau=self.heun, step=.17),
			ExplicitRungeKutta(tableau=self.midpoint, step=.17),
			DiagonallyImplicitRungeKutta(tableau=SDIRK4().tableau, step=.17),
			ImplicitRungeKutta(tableau=self.coupled, step=.17)):
			cpu = simulate(problem, method, observers=(EnergyObserver(),))
			device = simulate(problem, method, observers=(EnergyObserver(),), options=ExecutionOptions(backend='jax', device='cpu'))
			np.testing.assert_allclose(device.states, cpu.states, rtol=0., atol=2e-12)
			np.testing.assert_allclose(device.diagnostics['hamiltonian_time_derivative'], cpu.diagnostics['hamiltonian_time_derivative'], rtol=0., atol=2e-12)
			np.testing.assert_array_equal(device.t, problem.request.output_times)
			self.assertEqual(device.diagnostics['step_count'], 3)
			np.testing.assert_array_equal(device.diagnostics['tableau_A'], method.tableau.A)


if __name__ == '__main__':
	unittest.main()
