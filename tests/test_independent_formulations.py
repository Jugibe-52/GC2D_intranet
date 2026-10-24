"""Generic copy workspaces and method-owned physical/auxiliary snapshots."""
from dataclasses import replace
import importlib.util
import subprocess
import sys
import unittest
import numpy as np
from dynamics.coordinates import PhysicalCoordinates
from formulations.state import SingleFormulation, DoubleFormulation
from simulation import ABBA2Implicit, ABBA2Midpoint, BroydenSolver, NewtonSolver, InitialValueProblem, SimulationRequest
from states.energy import energy_diagnostics
from tests.test_method_integration import _problem


class IndependentFormulationTests(unittest.TestCase):
	def test_complete_independent_copies_for_arbitrary_sizes_and_sample_axes(self):
		for dimension in (1, 2, 3, 4, 7):
			for count in (1, 3):
				for samples in ((), (2, 3)):
					shape = (dimension * count, *samples)
					physical = np.arange(np.prod(shape), dtype=float).reshape(shape)
					before = physical.copy()
					for cls in (SingleFormulation, DoubleFormulation):
						form = cls(shape[0])
						spatial = form.lift(physical)
						self.assertEqual(spatial.shape, (form.copies * shape[0], *samples))
						for value in form.split_copies(spatial):
							np.testing.assert_array_equal(value, physical)
						spatial[0] = -99
						np.testing.assert_array_equal(physical, before)
						if form.copies == 2:
							np.testing.assert_array_equal(form.split_copies(spatial)[1], before)

	def test_shifted_copies_are_not_projected_or_repeatedly_duplicated(self):
		form = DoubleFormulation(4)
		physical = np.array([1., 2., 3., 4.]); mu = physical / 10
		first, second = form.split_copies(form.join_copies(physical + mu, physical - mu))
		np.testing.assert_array_equal(first, physical + mu)
		np.testing.assert_array_equal(second, physical - mu)
		for size in (0, -1, True, 1.5):
			with self.assertRaises(ValueError): DoubleFormulation(size)
		for callback in (lambda: form.lift(form.lift(physical)),
		                 lambda: form.join_copies(np.zeros((4, 2)), np.zeros((4, 3))),
		                 lambda: form.join_copies(physical),
		                 lambda: form.split_copies(physical)):
			with self.assertRaises(ValueError): callback()

	def test_workspaces_retain_only_size_and_import_no_numerical_or_model_packages(self):
		form = DoubleFormulation(6)
		for name in ('layout', 'particle_count', 'track_energy', 'problem', 'initial_state', 'initial_time', 'solver', 'projection', 'momentum', 'pack', 'physical'):
			self.assertFalse(hasattr(form, name), name)
		result = subprocess.run([sys.executable, '-B', '-c',
			'from formulations import SingleFormulation, DoubleFormulation; import sys; '
			'assert not any(x in sys.modules for x in ("dynamics", "methods", "execution", "solvers", "jax"))'], capture_output=True, text=True)
		self.assertEqual(result.returncode, 0, result.stderr)

	def test_energy_output_only_combines_supplied_values(self):
		times = np.array([.3, .4]); physical = np.ones((4, 2))
		auxiliary = {'extended_momentum': np.array([[2., 3.]])}
		with self.assertRaisesRegex(ValueError, 'caller-supplied'):
			energy_diagnostics(times, physical, auxiliary)
		data = energy_diagnostics(times, physical, auxiliary, energy=np.array([[5., 4.]]))
		np.testing.assert_array_equal(data['generalized_energy'], [[7., 7.]])
		np.testing.assert_array_equal(data['generalized_energy_error'], [[0., 0.]])

	@unittest.skipUnless(importlib.util.find_spec('jax'), 'Optional JAX is not installed')
	def test_jax_copies_and_derivatives_preserve_backend(self):
		import jax
		import jax.numpy as jnp
		for size in (6, 9, 12):
			for cls in (SingleFormulation, DoubleFormulation):
				form = cls(size)
				physical = jnp.arange(size * 2, dtype=float).reshape(size, 2)
				packed = jax.jit(form.lift)(physical)
				self.assertIsInstance(packed, jax.Array)
				for value in form.split_copies(packed): np.testing.assert_array_equal(value, physical)
				np.testing.assert_array_equal(jax.grad(lambda z: jnp.sum(form.lift(z)))(physical), np.full(physical.shape, form.copies))
				if cls is DoubleFormulation:
					self.assertIsInstance(form.join_copies(np.asarray(physical), physical), jax.Array)

	def test_methods_prepare_supplied_states_and_preserve_nonzero_momentum(self):
		problem = _problem(); request = SimulationRequest.uniform(t_span=(.3, .36))
		for method in (ABBA2Midpoint(), ABBA2Implicit(solver=NewtonSolver()), ABBA2Implicit(solver=BroydenSolver())):
			for tracking in (False, True):
				run = replace(method, step=.02, track_energy=tracking).new_run(problem, request)
				momentum = np.full(problem.particle_count, .17) if tracking else None
				first = run.advance(.3, run.initial_state, .02, momentum=momentum)
				self.assertEqual(first.state.shape, problem.initial_state.shape)
				for offset, duration in ((0., .02), (.03, .007)):
					candidate = first.state + offset; before = candidate.copy()
					actual = run.advance(.32, candidate, duration, momentum=first.momentum)
					np.testing.assert_array_equal(candidate, before)
					fresh = replace(method, track_energy=tracking).new_run(InitialValueProblem(problem.dynamics, candidate), SimulationRequest.uniform(t_span=(.32, .36)))
					expected = fresh.advance(.32, fresh.initial_state, duration, momentum=first.momentum)
					np.testing.assert_array_equal(actual.state, expected.state)
					if tracking: np.testing.assert_array_equal(actual.momentum, expected.momentum)
					mu = actual.details.multiplier if isinstance(method, ABBA2Implicit) else np.zeros_like(candidate)
					np.testing.assert_array_equal(actual.details.trace.state_before, np.concatenate((candidate + mu, candidate - mu)))
					self.assertFalse(np.shares_memory(actual.state, actual.details.trace.state))
