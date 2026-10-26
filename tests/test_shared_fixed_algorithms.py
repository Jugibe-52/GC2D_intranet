"""Shared numerical functions, prepared solver policy and optional imports."""
from dataclasses import fields
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from solvers.nonlinear import NewtonSolver, BroydenSolver


class OptionalImportTests(unittest.TestCase):
	def test_ordinary_configuration_does_not_import_jax(self):
		code = '''
import sys
from methods import RK4, ABBA4Implicit, HBVM42
from dynamics import GuidingCenterDynamics
from potential import Potential
from solvers import NewtonSolver
assert 'jax' not in sys.modules
'''
		subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)

	def test_fixed_methods_do_not_select_an_execution_backend(self):
		root = Path(__file__).resolve().parents[1] / 'src' / 'methods'
		for file in root.rglob('*.py'):
			if 'adaptive' in file.parts:
				continue
			with self.subTest(file=file):
				text = file.read_text()
				self.assertNotIn('import jax', text)
				self.assertNotIn('from jax', text)
				self.assertNotIn('jax_mode', text)
				self.assertNotIn('uses_jax', text)
		for name in ('_jax_dispatch.py', '_jax_options.py', 'classical/_jax_steps.py', 'extended/core/_jax_projection.py'):
			self.assertFalse((root / name).exists())


@unittest.skipUnless(importlib.util.find_spec('jax'), 'Optional JAX is not installed')
class SharedSolverTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		import jax
		import jax.numpy as jnp
		cls.jax, cls.xp = jax, jnp
		cls.previous = jax.config.read('jax_enable_x64')
		jax.config.update('jax_enable_x64', True)

	@classmethod
	def tearDownClass(cls):
		cls.jax.config.update('jax_enable_x64', cls.previous)

	def test_preparation_retains_every_control_and_is_idempotent(self):
		for solver in (NewtonSolver(2e-13, 3e-12, 19, jacobian_method='finite_difference',
			jacobian_relative_step=2e-6, backtracking=True), BroydenSolver(2e-13, 3e-12, 19)):
			prepared = solver._as_jax()
			self.assertIs(prepared._as_jax(), prepared)
			for item in fields(solver):
				self.assertEqual(getattr(prepared, item.name), getattr(solver, item.name))

	def test_dense_nonlinear_solves_and_full_broyden_secant_counts(self):
		matrix = np.array([[2., .3], [-.2, 1.5]])
		target = np.array([.8, -.4])
		initial = np.array([.2, -.1])
		for solver in (NewtonSolver(jacobian_method='finite_difference'), BroydenSolver()):
			def solve(config, xp, value):
				def evaluate(u):
					r = xp.asarray(matrix) @ u + .1 * u**3 - xp.asarray(target)
					return r, (u**2,)
				args = () if isinstance(config, NewtonSolver) else (xp.asarray(matrix),)
				return config.solve(evaluate, value, *args, scale=value)
			reference = solve(solver, np, initial)
			def device(value):
				r = solve(solver._as_jax(), self.xp, value)
				return r.unknown, r.residual, r.payload, r.iterations, r.residual_evaluations, r.converged
			actual = self.jax.jit(device)(self.xp.asarray(initial))
			np.testing.assert_allclose(actual[0], reference.unknown, rtol=0, atol=2e-13)
			np.testing.assert_allclose(actual[1], reference.residual, rtol=0, atol=2e-13)
			self.assertEqual(int(actual[3]), reference.iterations)
			self.assertEqual(int(actual[4]), reference.residual_evaluations)
			self.assertTrue(actual[5])

	def test_cached_initial_residual_zero_corrections_and_matrix_norm(self):
		for solver in (NewtonSolver(), BroydenSolver()):
			def solve(value):
				def evaluate(u):
					return u - 1., u * 2.
				args = () if isinstance(solver, NewtonSolver) else (self.xp.eye(2),)
				r = solver._as_jax().solve(evaluate, value, *args, scale=value,
					initial_evaluation=(self.xp.zeros_like(value), value * 2.))
				return r.iterations, r.residual_evaluations, r.payload, r.converged
			iterations, evaluations, payload, success = self.jax.jit(solve)(self.xp.ones(2))
			self.assertEqual(int(iterations), 0)
			self.assertEqual(int(evaluations), 1)
			np.testing.assert_array_equal(payload, [2., 2.])
			self.assertTrue(success)
		# Four entries each below atol, but a matrix row sum above atol: one
		# correction is necessary under HBVM's supplied matrix infinity norm.
		solver = NewtonSolver(.1, 1e-15, 2)
		def solve_matrix(u):
			r = solver._as_jax().solve(lambda z: (z, None), u, lambda z, r, p: self.xp.zeros_like(z),
				scale=u, norm=lambda r: self.xp.linalg.norm(r.reshape(2, 2), ord=np.inf))
			return r.iterations, r.converged
		count, success = self.jax.jit(solve_matrix)(self.xp.full(4, .08))
		self.assertEqual(int(count), 1)
		self.assertTrue(success)

	def test_damping_counts_and_failure_are_returned_from_compiled_iteration(self):
		initial = np.array([.1])
		def run(solver, xp, u):
			return solver.solve(lambda z: (z*z-2., None), u, lambda z, r, p: -r/(2*z),
				scale=u, update_returns_correction=True)
		for limit in (1, 20):
			solver = NewtonSolver(max_iterations=limit, backtracking=True)
			def compiled(u):
				r = run(solver._as_jax(), self.xp, u)
				return r.unknown, r.iterations, r.residual_evaluations, r.converged
			actual = self.jax.jit(compiled)(self.xp.asarray(initial))
			if limit == 1:
				self.assertFalse(actual[3])
			else:
				reference = run(solver, np, initial)
				np.testing.assert_allclose(actual[0], reference.unknown, rtol=0, atol=1e-15)
				self.assertEqual(int(actual[1]), reference.iterations)
				self.assertEqual(int(actual[2]), reference.residual_evaluations)

	def test_compilation_calls_the_same_advance_function(self):
		from methods import RK4, GaussLegendre4, ABBA2Implicit
		from dynamics.gc import GuidingCenterDynamics
		from potential import Potential
		from simulation import InitialValueProblem, SimulationRequest, simulate
		from execution.options import ExecutionOptions
		problem = InitialValueProblem(GuidingCenterDynamics(Potential.random(A=.1, M=1, nx=8, ny=8, seed=4)),
			np.array([.2, .3]), layout=GuidingCenterDynamics.coordinates)
		request = SimulationRequest((0., .01), np.array([0., .01]))
		for cls in (RK4, GaussLegendre4, ABBA2Implicit):
			calls = []
			advance = cls.advance
			def observed(method, *args, **kwargs):
				calls.append(type(args[1]))
				return advance(method, *args, **kwargs)
			with patch.object(cls, 'advance', observed):
				reference = simulate(problem.with_request(request), cls(step=.01))
				host_calls = len(calls)
				actual = simulate(problem.with_request(request), cls(step=.01), options=ExecutionOptions(backend='jax'))
			self.assertGreater(host_calls, 0)
			self.assertGreater(len(calls), host_calls)
			np.testing.assert_allclose(actual.states, reference.states, rtol=0, atol=2e-14)

	def test_zero_residual_cannot_hide_invalid_compiled_initial_data(self):
		for solver in (NewtonSolver(), BroydenSolver()):
			def solve(value, matrix):
				args = () if isinstance(solver, NewtonSolver) else (matrix,)
				result = solver._as_jax().solve(lambda z: (self.xp.zeros_like(z), None),
					value, *args, scale=self.xp.ones(2))
				return result.converged
			compiled = self.jax.jit(solve)
			self.assertTrue(compiled(self.xp.ones(2), self.xp.eye(2)))
			for invalid in (np.nan, np.inf):
				self.assertFalse(compiled(self.xp.array([1., invalid]), self.xp.eye(2)))
				if isinstance(solver, BroydenSolver):
					self.assertFalse(compiled(self.xp.ones(2), self.xp.full((2, 2), invalid)))


if __name__ == '__main__':
	unittest.main()
