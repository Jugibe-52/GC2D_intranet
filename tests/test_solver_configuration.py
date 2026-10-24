"""Solver-owned policies, event lifecycle and reusable derivative operations."""

from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from solvers import BroydenSolver, NewtonSolver
from solvers.derivatives import centered_difference, packed_particle_jacobian


class SolverConfigurationTests(unittest.TestCase):
	def test_configuration_is_immutable_and_additional_controls_are_keyword_only(self) -> None:
		solver = NewtonSolver(1e-12, 1e-11, 30, jacobian_method='finite_difference', backtracking=True)
		self.assertEqual(solver.max_iterations, 30)
		with self.assertRaises(FrozenInstanceError):
			solver.max_iterations = 1
		with self.assertRaises(TypeError):
			NewtonSolver(1e-12, 1e-11, 30, 'analytic')
		for controls in ({'jacobian_method': 'other'}, {'jacobian_relative_step': 0.},
			{'backtracking': 1}, {'iteration_observer': 3}):
			with self.subTest(controls=controls), self.assertRaises((ValueError, TypeError)):
				NewtonSolver(**controls)
		for controls in ({'backtracking': True}, {'iteration_observer': lambda event: None}):
			with self.subTest(controls=controls), self.assertRaises(ValueError):
				BroydenSolver(**controls)

	def test_observer_has_context_and_independent_snapshots_without_extra_evaluations(self) -> None:
		events = []
		calls = []
		def evaluate(value: np.ndarray) -> tuple[np.ndarray, None]:
			calls.append(value.copy())
			return value * value - 2., None
		solver = NewtonSolver(iteration_observer=events.append)
		initial = np.array([1., 2.])
		result = solver.solve(evaluate, initial, scale=initial, jacobian=lambda u: np.diag(2*u),
			time=.3, duration=-.02, stage_index=2, multiplier_slice=slice(1, None))
		self.assertEqual(len(calls), result.residual_evaluations)
		self.assertEqual(len(events), result.iterations + 1)
		self.assertEqual(events[0].time, .3)
		self.assertEqual(events[0].duration, -.02)
		self.assertEqual(events[0].stage_index, 2)
		np.testing.assert_array_equal(events[0].multiplier, [2.])
		result.unknown[:] = 10.
		np.testing.assert_allclose(events[-1].unknown, np.sqrt(2), atol=1e-11)
		self.assertFalse(events[-1].unknown.flags.writeable)
		count = len(events)
		muted = solver.solve(evaluate, initial, scale=initial, jacobian=lambda u: np.diag(2*u), observe=False)
		self.assertEqual(len(events), count)
		np.testing.assert_array_equal(muted.unknown, events[-1].unknown)

	def test_observer_failure_propagates_without_retry(self) -> None:
		calls = []
		def fail(event: object) -> None:
			calls.append(event)
			raise RuntimeError('observer failed')
		with self.assertRaisesRegex(RuntimeError, 'observer failed'):
			NewtonSolver(iteration_observer=fail).solve(lambda u: (u.copy(), None),
				np.zeros(1), scale=np.ones(1), jacobian=lambda u: np.eye(1))
		self.assertEqual(len(calls), 1)

	def test_zero_correction_convergence_does_not_require_or_evaluate_jacobian(self) -> None:
		solver = NewtonSolver(jacobian_method='analytic')
		result = solver.solve(lambda u: (u.copy(), None), np.zeros(2), scale=np.ones(2))
		self.assertEqual(result.iterations, 0)
		self.assertEqual(result.residual_evaluations, 1)
		with self.assertRaisesRegex(ValueError, 'requires a Jacobian'):
			solver.solve(lambda u: (u.copy(), None), np.ones(2), scale=np.ones(2))

	def test_finite_difference_strategy_uses_residual_and_supports_coupled_unknowns(self) -> None:
		matrix = np.array([[2., .4], [-.2, 3.]])
		calls = []
		def evaluate(value: np.ndarray) -> tuple[np.ndarray, None]:
			calls.append(value.copy())
			return matrix @ value - 1., None
		def forbidden(value: np.ndarray) -> np.ndarray:
			raise AssertionError('An explicitly disabled analytic Jacobian was evaluated.')
		result = NewtonSolver(jacobian_method='finite_difference').solve(
			evaluate, np.zeros(2), scale=np.ones(2), jacobian=forbidden)
		np.testing.assert_allclose(result.unknown, np.linalg.solve(matrix, np.ones(2)), atol=1e-12)
		self.assertEqual(result.residual_evaluations, len(calls))

	def test_backtracking_configuration_preserves_corrections_and_trial_accounting(self) -> None:
		calls = []
		def evaluate(value: np.ndarray) -> tuple[np.ndarray, None]:
			calls.append(value.copy())
			return value ** 3 - 1., None
		solver = NewtonSolver(1e-12, 1e-12, 30, backtracking=True)
		result = solver.solve(evaluate, np.array([.1]), scale=np.ones(1),
			jacobian=lambda u: np.diag(3*u*u))
		self.assertEqual(len(calls), result.residual_evaluations)
		self.assertGreater(result.residual_evaluations, result.iterations + 1)
		updated = solver.solve(evaluate, np.array([.1]),
			lambda u, r, payload: u - r / (3*u*u), scale=np.ones(1))
		np.testing.assert_allclose(updated.unknown, result.unknown, rtol=0, atol=1e-15)

	def test_equation_norm_and_cached_initial_residual_are_retained(self) -> None:
		calls = []
		def evaluate(value: np.ndarray) -> tuple[np.ndarray, None]:
			calls.append(value.copy())
			return value.copy(), None
		initial = np.full(4, .06)
		cached = evaluate(initial)
		result = NewtonSolver(.1, 1e-14).solve(evaluate, initial, scale=np.ones(4),
			jacobian=lambda u: np.eye(4), initial_evaluation=cached,
			norm=lambda r: float(np.linalg.norm(r.reshape(2, 2), ord=np.inf)))
		self.assertEqual(result.iterations, 1)
		self.assertEqual(len(calls), result.residual_evaluations)
		np.testing.assert_array_equal(result.unknown, np.zeros(4))


class SharedDerivativeTests(unittest.TestCase):
	def test_centered_difference_keeps_cross_particle_terms_and_counts_probes(self) -> None:
		matrix = np.array([[2., .4, 0.], [-.2, 3., .1], [.3, 0., 4.]])
		calls = []
		def evaluate(value: np.ndarray) -> np.ndarray:
			calls.append(value.copy())
			return matrix @ value
		state = np.array([.2, 30., -.5])
		actual = centered_difference(evaluate, state, relative_step=np.cbrt(np.finfo(float).eps))
		np.testing.assert_allclose(actual, matrix, atol=1e-9, rtol=0)
		self.assertEqual(len(calls), 2 * state.size)
		np.testing.assert_array_equal(state, [.2, 30., -.5])
		with self.assertRaisesRegex(ValueError, 'state shape'):
			centered_difference(lambda u: np.ones(2), state, relative_step=1e-5)
		with self.assertRaisesRegex(ValueError, 'finite'):
			centered_difference(lambda u: np.full_like(u, np.nan), state, relative_step=1e-5)

	def test_block_packing_uses_component_major_coordinates(self) -> None:
		blocks = np.arange(18., dtype=float).reshape(2, 3, 3)
		packed = packed_particle_jacobian(blocks)
		state = np.array([.1, .2, .3, .4, .5, .6])
		expected = np.stack([blocks[i] @ state[i::2] for i in range(2)]).T.reshape(-1)
		np.testing.assert_allclose(packed @ state, expected, atol=2e-15, rtol=0)


if __name__ == '__main__':
	unittest.main()
