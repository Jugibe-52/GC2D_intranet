"""Model-independent representations and method-owned state preparation."""

from __future__ import annotations

from dataclasses import replace

import importlib.util
import subprocess
import sys
import unittest

import numpy as np

from formulations.layout import GCStateLayout, FCStateLayout, PackedStateLayout
from formulations.state import SingleFormulation, DoubleFormulation
from simulation import (
	ABBA2Implicit, ABBA2Midpoint, BroydenSolver, InitialValueProblem,
	NewtonSolver, SimulationRequest,
)
from tests.test_method_integration import _problem


class ExternalLayout(PackedStateLayout):
	"""Three components prove that representation does not infer GC or FC."""

	state_dimension = 3

	def positions(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		parts = self.split(state)
		return parts[0], parts[1]


class IndependentFormulationTests(unittest.TestCase):
	def test_complete_copies_and_auxiliaries_for_every_layout(self) -> None:
		for layout in (GCStateLayout(), FCStateLayout(), ExternalLayout()):
			for count in (1, 3):
				for samples in ((), (2, 3)):
					shape = (layout.state_dimension * count, *samples)
					physical = np.arange(np.prod(shape), dtype=float).reshape(shape)
					before = physical.copy()
					for cls in (SingleFormulation, DoubleFormulation):
						for tracking in (False, True):
							with self.subTest(layout=type(layout).__name__, count=count, samples=samples, cls=cls.__name__, energy=tracking):
								form = cls(layout, count, tracking)
								spatial = form.lift(physical)
								self.assertEqual(spatial.shape, (form.copies * shape[0], *samples))
								for copy in form.split_copies(spatial):
									np.testing.assert_array_equal(copy, physical)
								momentum = np.full((count, *samples), 7.)
								state = form.pack(spatial, .3, momentum)
								self.assertEqual(state.shape, ((form.copies * layout.state_dimension + 2 * tracking) * count, *samples))
								np.testing.assert_array_equal(form.physical(state), physical)
								if tracking:
									np.testing.assert_array_equal(form.momentum(state), momentum)
									np.testing.assert_array_equal(form.time(state), np.full(momentum.shape, .3))
								else:
									self.assertIsNone(form.momentum(state))
								spatial[0] = -99
								np.testing.assert_array_equal(physical, before)
								np.testing.assert_array_equal(form.physical(state), before)
								if form.copies == 2:
									np.testing.assert_array_equal(form.split_copies(spatial)[1], before)

	def test_shifted_copies_are_preserved_without_hidden_projection(self) -> None:
		form = DoubleFormulation(FCStateLayout(), 1, True)
		physical = np.array([1., 2., 3., 4.])
		mu = np.array([.1, .2, .3, .4])
		spatial = form.join_copies(physical + mu, physical - mu)
		first, second = form.split_copies(spatial)
		np.testing.assert_array_equal(first, physical + mu)
		np.testing.assert_array_equal(second, physical - mu)
		state = form.pack(spatial, .2, np.zeros(1))
		# Extraction has an accepted-diagonal precondition; it never averages.
		np.testing.assert_array_equal(form.physical(state), physical + mu)

	def test_representation_rejects_invalid_and_repeated_duplication(self) -> None:
		for count in (0, -1, True, 1.5):
			with self.assertRaises(ValueError):
				DoubleFormulation(GCStateLayout(), count)
		form = DoubleFormulation(GCStateLayout(), 3, True)
		with self.assertRaises(ValueError):
			form.lift(form.lift(np.zeros(6)))
		with self.assertRaises(ValueError):
			form.pack(np.zeros(6), .3, np.zeros(3))
		with self.assertRaises(ValueError):
			form.join_copies(np.zeros((6, 2)), np.zeros((6, 3)))
		with self.assertRaises(ValueError):
			form.join_copies(np.zeros(6))
		with self.assertRaises(ValueError):
			form.pack(np.zeros(12), .3, np.zeros(2))
		with self.assertRaises(ValueError):
			form.pack(np.zeros(12), .3)
		with self.assertRaises(ValueError):
			form.physical(np.zeros(6))

	def test_formulations_own_structure_but_no_initial_values_or_algorithms(self) -> None:
		layout = ExternalLayout()
		layout.marker = [1]
		form = DoubleFormulation(layout, 2)
		layout.marker.append(2)
		self.assertEqual(form.layout.marker, [1])
		for name in ('problem', 'initial_state', 'initial_time', 'solver', 'projection', 'momentum_rate', 'finish'):
			self.assertFalse(hasattr(form, name), name)
		import formulations
		import simulation
		for module in (formulations, simulation):
			self.assertIs(module.SingleFormulation, SingleFormulation)
			self.assertIs(module.DoubleFormulation, DoubleFormulation)
			self.assertFalse(hasattr(module, 'PhysicalFormulation'))
			self.assertFalse(hasattr(module, 'DoubledFormulation'))
		result = subprocess.run([sys.executable, '-B', '-c',
			'from formulations.state import SingleFormulation; import sys; '
			'assert not any(x in sys.modules for x in ("methods", "execution", "jax"))'],
			capture_output=True, text=True)
		self.assertEqual(result.returncode, 0, result.stderr)

	def test_energy_output_uses_only_caller_supplied_values(self) -> None:
		form = DoubleFormulation(FCStateLayout(), 1, True)
		times = np.array([.3, .4])
		physical = np.ones((4, 2))
		momentum = np.array([[2., 3.]])
		energy = np.array([[5., 4.]])
		history = form.pack(form.lift(physical), times, momentum)
		with self.assertRaisesRegex(ValueError, 'caller-supplied'):
			form.extract_history(times, history)
		values, data = form.extract_history(times, history, energy=energy)
		np.testing.assert_array_equal(values, physical)
		np.testing.assert_array_equal(data['generalized_energy'], [[7., 7.]])
		np.testing.assert_array_equal(data['generalized_energy_error'], [[0., 0.]])

	@unittest.skipUnless(importlib.util.find_spec('jax'), 'Optional JAX is not installed')
	def test_jax_lift_pack_views_and_derivatives(self) -> None:
		import jax
		import jax.numpy as jnp
		for layout in (GCStateLayout(), FCStateLayout(), ExternalLayout()):
			for cls in (SingleFormulation, DoubleFormulation):
				form = cls(layout, 3, True)
				physical = jnp.arange(form.physical_size * 2, dtype=float).reshape(form.physical_size, 2)
				momentum = jnp.ones((3, 2))
				packed = jax.jit(lambda z, k: form.pack(form.lift(z), jnp.array([.3, .4]), k))(physical, momentum)
				self.assertIsInstance(packed, jax.Array)
				np.testing.assert_array_equal(form.physical(packed), physical)
				self.assertEqual(form.components(packed).shape, (form.dimension // 3, 3, 2))
				np.testing.assert_array_equal(jax.grad(lambda z: jnp.sum(form.lift(z)))(physical), np.full(physical.shape, form.copies))
				mixed = form.pack(form.lift(np.asarray(physical)), .3, momentum)
				self.assertIsInstance(mixed, jax.Array)

	def test_methods_prepare_each_supplied_state_and_shifted_candidate(self) -> None:
		problem = _problem()
		request = SimulationRequest.uniform(t_span=(.3, .36))
		methods = (ABBA2Midpoint(), ABBA2Implicit(solver=NewtonSolver()),
			ABBA2Implicit(solver=BroydenSolver()))
		for method in methods:
			with self.subTest(method=type(method).__name__, solver=getattr(method, 'solver', None)):
				run = replace(method, step=0.02).new_run(problem, request)
				form = run.state_formulation
				first = run.advance(.3, run.initial_state, .02)
				physical = form.physical(first.state)
				for offset, duration in ((0., .02), (.03, .007)):
					candidate = physical + offset
					internal = form.pack(form.lift(candidate), .32)
					before = internal.copy()
					actual = run.advance(.32, internal, duration)
					np.testing.assert_array_equal(internal, before)
					fresh_problem = InitialValueProblem(problem.dynamics, candidate, layout=problem.layout)
					fresh_request = SimulationRequest.uniform(t_span=(.32, .36))
					fresh = replace(method, step=0.02).new_run(fresh_problem, fresh_request)
					expected = fresh.advance(.32, fresh.initial_state, duration)
					np.testing.assert_array_equal(actual.state, expected.state)
					self.assertEqual(actual.state.shape, run.initial_state.shape)
					first_copy, second_copy = form.split_copies(actual.state)
					np.testing.assert_array_equal(first_copy, second_copy)
					if isinstance(method, ABBA2Implicit):
						mu = actual.details.multiplier
						np.testing.assert_array_equal(actual.details.trace.state_before,
							np.concatenate((candidate + mu, candidate - mu)))
					else:
						np.testing.assert_array_equal(actual.details.trace.state_before,
							np.concatenate((candidate, candidate)))
