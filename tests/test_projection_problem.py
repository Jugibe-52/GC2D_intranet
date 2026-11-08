"""Independent checks of candidate-only projection residuals and derivatives."""

from dataclasses import replace
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from contracts.arrays import array_namespace
from dynamics.coordinates import GCCoordinates
from methods import ABBA2Implicit, ABBA4Implicit, BM4Implicit
from methods.palindromic.maps import GCDoubledMaps
from methods.palindromic.core.composition import compose
from methods.palindromic.core.projection import ProjectionProblem, solve_projection
from methods.projection import ReducedProjection, SimultaneousProjection
from solvers import NewtonSolver, BroydenSolver
from solvers.derivatives import packed_particle_jacobian


def make_problem(form, *, structure='particle', coupling=0., derivative='analytic', recipe=ABBA2Implicit.recipe):
	"""A bounded nonautonomous Hamiltonian field with optional particle coupling."""
	def field(t, z):
		xp = array_namespace(z)
		x, y = xp.split(z, 2)
		return (1 + .1*xp.sin(t)) * xp.concatenate((
			xp.sin(y) + coupling*xp.sin(xp.sum(y)),
			-xp.sin(x) - coupling*xp.sin(xp.sum(x))))
	def derivative_field(t, z):
		xp = array_namespace(z)
		x, y = xp.split(z, 2)
		if structure == 'particle':
			zero = xp.zeros_like(x)
			return (1+.1*xp.sin(t))*xp.stack((xp.stack((zero, xp.cos(y)), axis=-1),
				xp.stack((-xp.cos(x), zero), axis=-1)), axis=-2)
		upper = xp.diag(xp.cos(y)) + coupling*xp.cos(xp.sum(y))*xp.ones((y.size, y.size))
		lower = -xp.diag(xp.cos(x)) - coupling*xp.cos(xp.sum(x))*xp.ones((x.size, x.size))
		zero = xp.zeros_like(upper)
		return (1+.1*xp.sin(t))*xp.concatenate((xp.concatenate((zero, upper), axis=1),
			xp.concatenate((lower, zero), axis=1)), axis=0)
	state = np.array([.2, .6, .4, .8])
	maps = GCDoubledMaps(field, state.size, coordinates=GCCoordinates(), coupling_frequency=None,
		jacobian=derivative_field if structure == 'particle' else None)
	return ProjectionProblem(maps, recipe, form, .3, state, .15, derivative if structure == 'particle' else 'finite_difference', 1e-6)


def candidate(problem):
	mu = np.array([.01, -.02, .03, -.01])
	return mu if isinstance(problem.equations, ReducedProjection) else np.concatenate((
		problem.state + .04, problem.state - .03, mu))


def dense(matrix):
	return packed_particle_jacobian(matrix) if matrix.ndim == 3 else matrix


class ProjectionProblemTests(unittest.TestCase):
	def test_residual_recomputes_copies_from_each_candidate(self):
		for form in (ReducedProjection(), SimultaneousProjection()):
			problem = make_problem(form)
			first = candidate(problem)
			for x in (first, first + .02, first.copy()):
				mu = x if isinstance(form, ReducedProjection) else x[-4:]
				mapped = compose(problem.maps, problem.recipe, problem.time,
					np.concatenate((problem.state+mu, problem.state-mu)), problem.duration).state
				u, v = np.split(mapped, 2)
				expected = u-v+2*mu if isinstance(form, ReducedProjection) else np.concatenate((
					x[:4]-mu-u, x[4:8]+mu-v, x[:4]-x[4:8]))
				np.testing.assert_array_equal(problem.residual(x), expected)
				self.assertEqual(problem.residual(x).shape, x.shape)
			# A public evaluation must not retain a stale reference after mutation.
			before = problem.residual(first)
			first[-4:] += .03
			self.assertFalse(np.array_equal(before, problem.residual(first)))
			with self.assertRaises(ValueError):
				problem.residual(np.zeros(first.size+1))
			with self.assertRaises(ValueError):
				problem.jacobian(np.zeros((first.size, 1)))

	def test_complete_jacobian_matches_independent_residual_differences(self):
		maximum = 0.
		for form in (ReducedProjection(), SimultaneousProjection()):
			for recipe in (ABBA2Implicit.recipe, ABBA4Implicit.recipe, BM4Implicit.recipe):
				for structure, coupling in (('particle', 0.), ('dense', .15)):
					problem = make_problem(form, structure=structure, coupling=coupling, recipe=recipe)
					x = candidate(problem)
					eps = 1e-6
					numeric = np.column_stack([(problem.residual(x+eps*e)-problem.residual(x-eps*e))/(2*eps)
						for e in np.eye(x.size)])
					analytic = dense(problem.jacobian(x))
					maximum = max(maximum, float(np.max(np.abs(analytic-numeric))))
					np.testing.assert_allclose(analytic, numeric, atol=2e-9, rtol=2e-8)
					np.testing.assert_allclose(replace(problem, derivative='finite_difference').jacobian(x),
						numeric, atol=2e-9, rtol=2e-8)
		print(f'Complete projection Jacobian maximum discrepancy: {maximum:.3e}')

	def test_zero_step_jacobian_and_independent_step_context(self):
		for form in (ReducedProjection(), SimultaneousProjection()):
			problem = make_problem(form)
			x = candidate(problem)
			zero = replace(problem, duration=0.)
			np.testing.assert_array_equal(dense(zero.jacobian(x)), form.initial_jacobian(problem.state))
			original = problem.residual(x)
			other = replace(problem, time=1.2, state=problem.state+.1, duration=-.2)
			self.assertFalse(np.array_equal(original, other.residual(x)))
			np.testing.assert_array_equal(original, problem.residual(x))

	def test_newton_reuses_each_residual_composition_for_its_jacobian(self):
		for form in (ReducedProjection(), SimultaneousProjection()):
			for solver in (NewtonSolver(1e-14, 1e-13, 30), BroydenSolver(1e-14, 1e-13, 30)):
				problem = make_problem(form)
				with patch('methods.palindromic.core.projection.compose', wraps=compose) as calls:
					result = solve_projection(problem.maps, problem.recipe, solver, form,
						problem.time, problem.state, problem.duration)
				self.assertTrue(result.stats.converged)
				self.assertEqual(calls.call_count, result.stats.residual_evaluations)
				self.assertLessEqual(result.stats.residual_norm, result.stats.tolerance)

	def test_solved_initial_residual_needs_no_jacobian(self):
		for form in (ReducedProjection(), SimultaneousProjection()):
			problem = make_problem(form)
			with patch.object(ProjectionProblem, 'jacobian', side_effect=AssertionError('Unneeded derivative')):
				result = solve_projection(problem.maps, problem.recipe, NewtonSolver(), form,
					problem.time, problem.state, 0.)
			self.assertEqual(result.stats.iterations, 0)
			np.testing.assert_array_equal(result.state, problem.state)

	@unittest.skipUnless(importlib.util.find_spec('jax'), 'Optional JAX is unavailable')
	def test_candidate_only_functions_are_jittable_and_match_autodiff(self):
		import jax
		import jax.numpy as jnp
		with jax.experimental.enable_x64():
			for form in (ReducedProjection(), SimultaneousProjection()):
				problem = make_problem(form)
				x = jnp.asarray(candidate(problem))
				for current in (x, x+.02, x):
					actual = jax.jit(problem.residual)(current)
					jacobian = dense(jax.jit(problem.jacobian)(current))
					np.testing.assert_allclose(actual, problem.residual(np.asarray(current)), atol=1e-15, rtol=1e-14)
					np.testing.assert_allclose(jacobian, jax.jacfwd(problem.residual)(current), atol=2e-14, rtol=2e-14)


if __name__ == '__main__':
	unittest.main()
