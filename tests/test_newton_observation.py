"""Run-local Newton histories and independent particle-system contracts."""

from __future__ import annotations


from dynamics import GuidingCenterDynamics
from solvers import BroydenSolver, NewtonSolver
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from contracts.solvers.nonlinear import NewtonIteration
from problem import InitialValueProblem
from request import SimulationRequest

from solvers.linear import _solve_particle_systems
from solvers.nonlinear import _bind_newton_observer, _solve_newton
from methods.classical.gauss_legendre import GaussLegendre4
from methods.classical.sdirk import SDIRK4
from methods.extended.abba import ABBA2Implicit, ABBA4Implicit
from methods.extended.bm4 import BM4Implicit
from potential.potential import Potential
from simulation.runner import simulate


def _problem(count: int = 3) -> InitialValueProblem:
	"""Three independent nonlinear particles expose packed/batched shape errors."""
	return InitialValueProblem(GuidingCenterDynamics(Potential.random(A=.08, M=3, nx=16, ny=16, seed=27), rho=.05), GuidingCenterDynamics.coordinates.pack_components(np.linspace(1., 1.2, count), np.linspace(1.2, 1.4, count)), layout=GuidingCenterDynamics.coordinates)


def _request(dense: bool = False) -> SimulationRequest:
	return SimulationRequest((.3, .34),  np.asarray([.3, .31, .32, .34] if dense else [.3, .32, .34]))


class NewtonObservationTests(unittest.TestCase):
	def test_observation_preserves_work_results_and_owned_iterates(self) -> None:
		problem = _problem()
		for method in (
			BM4Implicit(), ABBA2Implicit(), ABBA4Implicit(),
			ABBA2Implicit(projection_formulation='simultaneous_state_multiplier'),
			GaussLegendre4(), SDIRK4(),
		):
			with self.subTest(method=method):
				events: list[NewtonIteration] = []
				with patch.object(type(problem.dynamics), 'vector_field', autospec=True,
					side_effect=type(problem.dynamics).vector_field) as field:
					plain = simulate(problem.with_request(_request()), replace(method, step=(0.34 - 0.3) / 2))
					plain_calls = field.call_count
					field.reset_mock()
					observed = simulate(problem.with_request(_request()), replace(replace(method, solver=replace(method.solver, iteration_observer=events.append)), step=(0.34 - 0.3) / 2))
					self.assertEqual(field.call_count, plain_calls)
				np.testing.assert_array_equal(plain.states, observed.states)
				for name in plain.diagnostics:
					np.testing.assert_equal(plain.diagnostics[name], observed.diagnostics[name])
				self.assertEqual(len(events), np.sum(observed.diagnostics['residual_evaluations']))
				for event in events:
					self.assertEqual(event.residual_norm, np.linalg.norm(event.residual, ord=np.inf))
					self.assertGreater(event.tolerance, 0.)
					for vector in (event.unknown, event.residual, event.multiplier):
						if vector is not None:
							self.assertFalse(vector.flags.writeable)
							with self.assertRaises(ValueError):
								vector[:] = 99
					if isinstance(method, SDIRK4):
						self.assertIn(event.stage_index, range(5))
					elif isinstance(method, (BM4Implicit, ABBA2Implicit, ABBA4Implicit,)):
						self.assertEqual(event.multiplier.shape, problem.initial_state.shape)

	def test_reentrant_runs_and_diagnostic_maps_do_not_share_callbacks(self) -> None:
		problem = _problem(1)
		for method in (BM4Implicit(), ABBA2Implicit(), GaussLegendre4(), SDIRK4()):
			with self.subTest(method=type(method).__name__):
				outer: list[NewtonIteration] = []
				inner: list[NewtonIteration] = []
				steps = []
				def observe(event: NewtonIteration) -> None:
					outer.append(event)
					if len(outer) == 1:
						simulate(problem.with_request(_request()), replace(replace(method, solver=replace(method.solver, iteration_observer=inner.append)), step=(0.34 - 0.3) / 2))
				result = simulate(problem.with_request(_request()), replace(replace(method, solver=replace(method.solver, iteration_observer=observe)), step=(0.34 - 0.3) / 2), observers=(steps.append,))
				self.assertEqual(len(outer), len(inner))
				self.assertEqual(len(outer), np.sum(result.diagnostics['residual_evaluations']))
				before = len(outer)
				for step in steps:
					np.testing.assert_array_equal(step.map_state(step.state_before), step.state_after)
				self.assertEqual(len(outer), before)

	def test_cached_and_failed_iterations_are_reported_once(self) -> None:
		events: list[NewtonIteration] = []
		callback = _bind_newton_observer(events.append, time=1., duration=.1, tolerance=1e-12)
		def unexpected(*args):
			raise AssertionError('A cached root requires no numerical work.')
		_solve_newton(unexpected, np.ones(2), unexpected, tolerance=1e-12,
			max_iterations=2, context='cached test', initial_evaluation=(np.zeros(2), None),
			iteration_observer=callback)
		self.assertEqual([event.iteration for event in events], [0])
		events.clear()
		with self.assertRaisesRegex(RuntimeError, 'did not converge'):
			_solve_newton(lambda x: (x * x + 1, None), np.ones(1), lambda x, r, p: x - r / (2*x),
				tolerance=1e-12, max_iterations=1, context='failed test', iteration_observer=callback)
		self.assertEqual([event.iteration for event in events], [0, 1])

	def test_history_is_explicit_and_matches_projection_statistics(self) -> None:
		from studies.poincare_batch_runtime.newton_diagnostics import NewtonHistory
		history = NewtonHistory(expected_steps=2, progress_every=0)
		result = simulate(_problem().with_request(_request()), BM4Implicit(step=(0.34 - 0.3) / 2, solver=NewtonSolver(jacobian_method='analytic', iteration_observer=history)))
		arrays = history.arrays()
		np.testing.assert_array_equal(np.diff(arrays['offsets']), result.diagnostics['nonlinear_iterations'] + 1)
		final = arrays['offsets'][1:] - 1
		np.testing.assert_array_equal(arrays['residuals'][final], result.diagnostics['nonlinear_residual_norms'])
		np.testing.assert_array_equal(arrays['mu_norms'][final], result.diagnostics['projection_multiplier_norms'])

	def test_shadow_solves_are_visible_and_broyden_callbacks_are_rejected(self) -> None:
		events: list[NewtonIteration] = []
		simulate(_problem().with_request(_request(dense=True)), BM4Implicit(step=(0.34 - 0.3) / 2, solver=NewtonSolver(jacobian_method='analytic', iteration_observer=events.append)))
		initial_guesses = [event for event in events if event.iteration == 0]
		self.assertEqual(len(initial_guesses), 3)
		self.assertTrue(any(abs(event.duration - .01) < 1e-15 for event in initial_guesses))
		for method in (BM4Implicit, ABBA2Implicit, ABBA4Implicit,):
			with self.assertRaisesRegex(ValueError, 'BroydenSolver does not support'):
				method(solver=BroydenSolver(iteration_observer=events.append))

	def test_remote_callbacks_are_rejected_before_serialization(self) -> None:
		from execution._modal_worker import encode_job
		problem = _problem(1)
		with patch('execution._modal_worker.pickle.dumps') as serialize:
			with self.assertRaisesRegex(NotImplementedError, 'iteration callbacks'):
				encode_job(problem.with_request(_request()), BM4Implicit(solver=NewtonSolver(jacobian_method='analytic', iteration_observer=lambda event: None)), None)
			serialize.assert_not_called()

	def test_snapshot_mutation_and_callback_failure_leave_other_runs_unchanged(self) -> None:
		problem = _problem(1)
		for method in (BM4Implicit(), ABBA2Implicit(), GaussLegendre4(), SDIRK4()):
			with self.subTest(method=type(method).__name__):
				plain = simulate(problem.with_request(_request()), replace(method, step=(0.34 - 0.3) / 2))
				def mutate(event: NewtonIteration) -> None:
					# Even deliberately re-enabling writes can only change owned copies.
					for vector in (event.unknown, event.residual, event.multiplier):
						if vector is not None:
							vector.setflags(write=True)
							vector[:] = 999
				observed = simulate(problem.with_request(_request()), replace(replace(method, solver=replace(method.solver, iteration_observer=mutate)), step=(0.34 - 0.3) / 2))
				np.testing.assert_array_equal(observed.states, plain.states)
				def fail(event: NewtonIteration) -> None:
					raise RuntimeError('Observer failed deliberately.')
				with self.assertRaisesRegex(RuntimeError, 'Observer failed deliberately'):
					simulate(problem.with_request(_request()), replace(replace(method, solver=replace(method.solver, iteration_observer=fail)), step=(0.34 - 0.3) / 2))
				after_failure = simulate(problem.with_request(_request()), replace(method, step=(0.34 - 0.3) / 2))
				np.testing.assert_array_equal(after_failure.states, plain.states)


class ParticleLinearSystemTests(unittest.TestCase):
	def test_batched_vectors_match_independent_systems(self) -> None:
		random = np.random.default_rng(19)
		for count in (1, 2, 3, 4):
			for dimension in (2, 4, 6):
				with self.subTest(count=count, dimension=dimension):
					matrix = random.normal(size=(count, dimension, dimension)) + dimension * np.eye(dimension)
					rhs = random.normal(size=(count, dimension))
					actual = _solve_particle_systems(matrix, rhs, singular_message='singular')
					expected = np.stack([np.linalg.solve(a, b) for a, b in zip(matrix, rhs)])
					np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-15)
					self.assertEqual(actual.shape, rhs.shape)

	def test_all_analytic_particle_solves_use_explicit_rhs_axis(self) -> None:
		original = np.linalg.solve
		def require_explicit_rhs(matrix, rhs):
			if matrix.ndim == 3:
				self.assertEqual(rhs.shape, (*matrix.shape[:2], 1))
			return original(matrix, rhs)
		for count in (1, 2, 3, 4):
			for method in (GaussLegendre4(), SDIRK4(), BM4Implicit(), ABBA2Implicit(projection_formulation='simultaneous_state_multiplier')):
				with self.subTest(count=count, method=method), patch('numpy.linalg.solve', side_effect=require_explicit_rhs):
					simulate(_problem(count).with_request(_request()), replace(method, step=(0.34 - 0.3) / 2))


if __name__ == '__main__':
	unittest.main()
