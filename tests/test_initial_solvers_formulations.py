"""Ownership and numerical interchangeability of initial, solver and equation objects."""

from dataclasses import FrozenInstanceError
import pickle
import unittest

import numpy as np

from contracts.problem import InitialValueProblem
from dynamics import GuidingCenterDynamics, FullCyclotronDynamics
from formulations.layout import GCStateLayout
from formulations.projection import ReducedProjection, SimultaneousProjection
from methods import ABBA4Implicit, BM4Implicit, GaussLegendre4, SDIRK4, HBVM42, RK4
from potential import Potential
from request import SimulationRequest
from simulation.runner import simulate
from solvers import NewtonSolver, BroydenSolver
from states import InitialConditions


class InitialConditionsTests(unittest.TestCase):
	def test_owned_components_and_serialization(self) -> None:
		x = np.array([.2, .4])
		initial = InitialConditions.gc(x=x, y=[.3, .5])
		x[:] = 90
		initial.values[:] = 80
		initial.positions()[0][:] = 70
		self.assertEqual(initial.particle_count, 2)
		np.testing.assert_array_equal(initial.values, [.2, .4, .3, .5])
		restored = pickle.loads(pickle.dumps(initial))
		np.testing.assert_array_equal(restored.values, initial.values)
		self.assertFalse(restored._values.flags.writeable)
		with self.assertRaises(FrozenInstanceError):
			initial.particle_count = 20
		with self.assertRaises(ValueError):
			initial.velocities()

	def test_fc_values_and_problem_compatibility(self) -> None:
		potential = Potential.random(A=.01, M=2, nx=8, ny=8)
		initial = InitialConditions.fc(x=[1.], y=[2.], vx=[3.], vy=[4.])
		np.testing.assert_array_equal(initial.values, [1., 2., 3., 4.])
		np.testing.assert_array_equal(initial.velocities(), [[3.], [4.]])
		problem = InitialValueProblem(FullCyclotronDynamics(potential, rho=.2, eta=.1), initial)
		self.assertIs(problem.initial_conditions, initial)
		np.testing.assert_array_equal(problem.initial_state, initial.values)
		with self.assertRaisesRegex(TypeError, 'incompatible'):
			InitialValueProblem(GuidingCenterDynamics(potential), initial)
		with self.assertRaisesRegex(ValueError, 'already owns'):
			InitialValueProblem(problem.dynamics, initial, layout=GCStateLayout())

	def test_invalid_initial_values(self) -> None:
		for x,y in [([], []), ([1.], [2., 3.]), ([np.nan], [1.]), ([[1.]], [[2.]])]:
			with self.subTest(x=x), self.assertRaises(ValueError):
				InitialConditions.gc(x=x, y=y)


class AlgebraicSolverTests(unittest.TestCase):
	def test_newton_reuse_cached_evaluation_and_failure(self) -> None:
		solver = NewtonSolver(1e-13, 1e-12, 20)
		calls = []
		def evaluate(value: np.ndarray) -> tuple[np.ndarray, object]:
			calls.append(value.copy())
			return value * value - 2., value.copy()
		initial = np.array([1.])
		cached = evaluate(initial)
		result = solver.solve(evaluate, initial, scale=initial,
			jacobian=lambda u: np.diag(2 * u), initial_evaluation=cached)
		np.testing.assert_allclose(result.unknown, np.sqrt(2), rtol=1e-12)
		self.assertEqual(result.residual_evaluations, len(calls))
		second = solver.solve(evaluate, initial, scale=initial, jacobian=lambda u: np.diag(2 * u))
		np.testing.assert_array_equal(second.unknown, result.unknown)
		with self.assertRaises(RuntimeError):
			NewtonSolver(1e-14, 1e-14, 1).solve(evaluate, initial, scale=initial, jacobian=lambda u: np.diag(2*u))
		with self.assertRaises(ValueError):
			solver.solve(lambda u: (np.array([np.nan]), None), initial, scale=initial, jacobian=lambda u: np.eye(1))

	def test_broyden_owns_jacobian_work(self) -> None:
		matrix = np.array([[2., .1], [.2, 3.]])
		initial_matrix = np.eye(2)
		solver = BroydenSolver(1e-12, 1e-12, 40)
		for unused in range(2):
			result = solver.solve(lambda u: (matrix @ u - np.ones(2), None),
				np.zeros(2), initial_matrix, scale=np.ones(2))
			np.testing.assert_allclose(result.unknown, np.linalg.solve(matrix, np.ones(2)), atol=1e-11)
		np.testing.assert_array_equal(initial_matrix, np.eye(2))

	def test_damped_newton_uses_caller_norm_and_counts_all_trials(self) -> None:
		calls = []
		def evaluate(value: np.ndarray) -> tuple[np.ndarray, object]:
			calls.append(value.copy())
			return value ** 3 - 1, None
		result = NewtonSolver(1e-12, 1e-12, 30).solve(evaluate, np.array([.1]), scale=np.ones(1),
			jacobian=lambda u: np.diag(3 * u ** 2), backtracking=True)
		np.testing.assert_allclose(result.unknown, 1., atol=1e-11)
		self.assertEqual(result.residual_evaluations, len(calls))
		self.assertGreater(result.residual_evaluations, result.iterations + 1)


class ProjectionEquationsTests(unittest.TestCase):
	def test_jacobians_against_independent_residual_differences(self) -> None:
		base = np.eye(4) + .03 * np.random.default_rng(41).normal(size=(4, 4))
		state = np.array([.2, .4])
		for equation in (ReducedProjection(), SimultaneousProjection()):
			with self.subTest(equation=equation.name):
				initial, cached = equation.prepare(state, lambda u: (base @ u, None))
				def residual(u: np.ndarray) -> np.ndarray:
					mu = equation.multiplier(u, 2)
					mapped = base @ np.concatenate((state + mu, state - mu))
					return equation.residual(u, mapped, 2)
				epsilon = 1e-6
				jacobian = np.column_stack([(residual(initial + epsilon * e) - residual(initial - epsilon * e))/(2*epsilon)
					for e in np.eye(initial.size)])
				np.testing.assert_allclose(equation.dense_jacobian(base, 2), jacobian, rtol=1e-8, atol=1e-9)
				np.testing.assert_allclose(equation.particle_jacobian(base[None])[0], jacobian, rtol=1e-8, atol=1e-9)


class ConfiguredIntegrationTests(unittest.TestCase):
	def setUp(self) -> None:
		self.dynamics = GuidingCenterDynamics(Potential.random(A=.04, M=2, nx=12, ny=12, seed=41))
		self.initial = InitialConditions.gc(x=[.2, .7], y=[.3, .6])
		self.problem = InitialValueProblem(self.dynamics, self.initial)
		self.request = SimulationRequest.uniform(t_span=(.2, .28), max_step=.02, sample_count=7)

	def test_solver_objects_match_existing_controls_and_can_be_reused(self) -> None:
		solver = NewtonSolver(1e-12, 1e-11, 30)
		controls = dict(newton_absolute_tolerance=1e-12, newton_relative_tolerance=1e-11, newton_max_iterations=30)
		cases = [(kind(**controls), kind(solver=solver)) for kind in (GaussLegendre4, SDIRK4, BM4Implicit)]
		cases += [(ABBA4Implicit(projection_formulation=name, **controls),
			ABBA4Implicit(projection_formulation=equation, solver=solver)) for name,equation in
			[('reduced_multiplier', ReducedProjection()), ('simultaneous_state_multiplier', SimultaneousProjection())]]
		cases += [(HBVM42(absolute_tolerance=1e-12, relative_tolerance=1e-11, max_iterations=30), HBVM42(solver=solver))]
		cases += [(ABBA4Implicit(nonlinear_solver='broyden', **controls), ABBA4Implicit(solver=BroydenSolver(1e-12, 1e-11, 30)))]
		for legacy, configured in cases:
			with self.subTest(method=type(configured).__name__, projection=getattr(configured, 'projection_formulation', None)):
				expected = simulate(self.problem, legacy, self.request)
				for unused in range(2):
					actual = simulate(self.problem, configured, self.request)
					np.testing.assert_array_equal(actual.states, expected.states)
					for key in ('nonlinear_iterations', 'residual_evaluations'):
						np.testing.assert_array_equal(actual.diagnostics[key], expected.diagnostics[key])

	def test_explicit_solver_rejects_unsupported_method_and_override(self) -> None:
		for method in (GaussLegendre4, SDIRK4, HBVM42):
			with self.assertRaises(ValueError):
				method(solver=BroydenSolver())
		class CustomNewton(NewtonSolver):
			pass
		with self.assertRaises(TypeError):
			ABBA4Implicit(solver=CustomNewton())
		with self.assertRaises(TypeError):
			RK4(solver=NewtonSolver())

	def test_explicit_objects_match_cpu_on_jax(self) -> None:
		try:
			import jax
		except ImportError:
			self.skipTest('Optional JAX is unavailable')
		from contracts.execution_options import ExecutionOptions
		with jax.experimental.enable_x64():
			for method in (GaussLegendre4(solver=NewtonSolver()),
				ABBA4Implicit(projection_formulation=ReducedProjection(), solver=NewtonSolver()),
				ABBA4Implicit(projection_formulation=SimultaneousProjection(), solver=NewtonSolver())):
				expected = simulate(self.problem, method, self.request)
				actual = simulate(self.problem, method, self.request, options=ExecutionOptions(backend='jax'))
				np.testing.assert_allclose(actual.states, expected.states, atol=3e-12, rtol=0)
				np.testing.assert_array_equal(actual.diagnostics['nonlinear_iterations'], expected.diagnostics['nonlinear_iterations'])

	def test_configured_job_serialization_and_cpu_execution(self) -> None:
		from execution._modal_worker import encode_job, decode_job, execute_payload
		equation = SimultaneousProjection()
		method = ABBA4Implicit(projection_formulation=equation, solver=NewtonSolver())
		self.assertIs(method.projection_formulation, equation)
		payload = encode_job(self.problem, method, self.request, None)
		problem, restored, request, options = decode_job(payload)
		self.assertIsInstance(restored.projection_formulation, SimultaneousProjection)
		self.assertIsInstance(problem.initial_conditions, InitialConditions)
		expected = simulate(self.problem, method, self.request)
		actual = execute_payload(payload)[2]
		np.testing.assert_array_equal(actual.states, expected.states)

	def test_projection_forms_recover_same_physical_map(self) -> None:
		results = [simulate(self.problem, ABBA4Implicit(projection_formulation=form,
			solver=NewtonSolver(1e-13, 1e-12, 30)), self.request) for form in (ReducedProjection(), SimultaneousProjection())]
		np.testing.assert_allclose(results[0].states, results[1].states, rtol=0, atol=3e-12)
		self.assertEqual(results[0].diagnostics['nonlinear_unknown_dimension'], 4)
		self.assertEqual(results[1].diagnostics['nonlinear_unknown_dimension'], 12)


if __name__ == '__main__':
	unittest.main()
