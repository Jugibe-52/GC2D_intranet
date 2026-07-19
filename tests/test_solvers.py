from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import ArrayLike

from classes import Solution, solve_extended, solve_symplectic
from classes.system import _step_count


def identity_flow(_h: float, _t: float, state: np.ndarray) -> np.ndarray:
	return state


class HarmonicOscillator:
	"""Small protocol implementation used to exercise the extended solver."""

	degrees_of_freedom = 1
	time_dependent = False

	@staticmethod
	def vector_field(_t: float, state: np.ndarray) -> np.ndarray:
		q, p = np.split(state, 2)
		return np.concatenate((p, -q))

	@staticmethod
	def extended_momentum_derivative(_t: float, state: np.ndarray) -> np.ndarray:
		return np.zeros(state.shape[0] // 2)

	@staticmethod
	def compute_energy(_solution: Solution) -> float:
		return 0.0


class SolveSymplecticTests(unittest.TestCase):
	def test_accepts_array_like_initial_state(self) -> None:
		solution = solve_symplectic(
			identity_flow,
			identity_flow,
			(0.0, 0.1),
			[1.0, 0.0],
			step=0.01,
		)

		self.assertIsInstance(solution, Solution)
		self.assertEqual(solution.y.shape, (2, 11))
		self.assertTrue(np.issubdtype(solution.y.dtype, np.floating))
		np.testing.assert_array_equal(solution.y[:, -1], [1.0, 0.0])

	def test_samples_nonuniform_times_without_including_t0(self) -> None:
		requested = np.array([0.07, 0.23, 0.4])
		solution = solve_symplectic(
			identity_flow,
			identity_flow,
			(0.0, 0.4),
			[1.0, 0.0],
			step=0.03,
			t_eval=requested,
		)

		np.testing.assert_array_equal(solution.t, requested)
		self.assertEqual(solution.y.shape, (2, 3))
		self.assertLessEqual(solution.step, 0.03)

	def test_accepts_one_requested_time(self) -> None:
		solution = solve_symplectic(
			identity_flow,
			identity_flow,
			(0.0, 1.0),
			[1.0],
			step=0.1,
			t_eval=[0.5],
		)

		np.testing.assert_array_equal(solution.t, [0.5])
		self.assertEqual(solution.y.shape, (1, 1))

	def test_preserves_positional_t_eval_compatibility(self) -> None:
		solution = solve_symplectic(
			identity_flow,
			identity_flow,
			(0.0, 1.0),
			[1.0],
			0.1,
			[0.0, 0.5, 1.0],
		)

		np.testing.assert_array_equal(solution.t, [0.0, 0.5, 1.0])

	def test_zero_length_span_returns_one_initial_state(self) -> None:
		solution = solve_symplectic(
			identity_flow,
			identity_flow,
			(2.0, 2.0),
			[3.0, 4.0],
			step=0.1,
		)

		np.testing.assert_array_equal(solution.t, [2.0])
		np.testing.assert_array_equal(solution.y[:, 0], [3.0, 4.0])
		self.assertEqual(solution.step, 0.1)

	def test_callback_runs_after_each_internal_step(self) -> None:
		callback_times: list[float] = []
		solve_symplectic(
			identity_flow,
			identity_flow,
			(0.0, 0.1),
			[1.0],
			step=0.03,
			command=lambda t, _state: callback_times.append(t),
		)

		np.testing.assert_allclose(callback_times, [0.03, 0.06, 0.09, 0.1])

	def test_save_step_controls_storage_independently(self) -> None:
		solution = solve_symplectic(
			identity_flow,
			identity_flow,
			(0.0, 0.25),
			[1.0],
			step=0.03,
			save_step=0.1,
		)

		np.testing.assert_allclose(solution.t, [0.0, 0.1, 0.2, 0.25])
		self.assertEqual(solution.y.shape, (1, 4))
		self.assertEqual(solution.n_steps, 10)
		self.assertEqual(solution.requested_step, 0.03)
		self.assertLessEqual(solution.max_step, solution.requested_step)
		self.assertLessEqual(solution.min_step, solution.max_step)

	def test_save_step_matches_an_equivalent_regular_t_eval(self) -> None:
		def drift(h: float, t: float, state: np.ndarray) -> np.ndarray:
			q, p = state
			return np.array([q + h * (p + 0.1 * t), p])

		def kick(h: float, t: float, state: np.ndarray) -> np.ndarray:
			q, p = state
			return np.array([q, p - h * (q - 0.05 * t)])

		requested = 2 * np.pi * np.arange(4)
		by_save_step = solve_symplectic(
			drift,
			kick,
			(0.0, requested[-1]),
			[1.0, 0.25],
			step=0.1,
			save_step=2 * np.pi,
		)
		by_t_eval = solve_symplectic(
			drift,
			kick,
			(0.0, requested[-1]),
			[1.0, 0.25],
			step=0.1,
			t_eval=requested,
		)

		np.testing.assert_array_equal(by_save_step.t, requested)
		np.testing.assert_array_equal(by_save_step.y, by_t_eval.y)
		self.assertEqual(by_save_step.n_steps, by_t_eval.n_steps)

	def test_n_save_step_builds_uniform_output(self) -> None:
		solution = solve_symplectic(
			identity_flow,
			identity_flow,
			(1.0, 3.0),
			[2.0],
			step=0.3,
			n_save_step=5,
		)

		np.testing.assert_array_equal(solution.t, np.linspace(1.0, 3.0, 5))
		self.assertEqual(solution.y.shape, (1, 5))

	def test_output_controls_are_mutually_exclusive(self) -> None:
		combinations = (
			{"save_step": 0.2, "t_eval": [0.0, 1.0]},
			{"save_step": 0.2, "n_save_step": 3},
			{"t_eval": [0.0, 1.0], "n_save_step": 3},
		)
		for options in combinations:
			with self.subTest(options=options):
				with self.assertRaisesRegex(ValueError, "mutually exclusive"):
					solve_symplectic(
						identity_flow,
						identity_flow,
						(0.0, 1.0),
						[1.0],
						step=0.1,
						**options,
					)

	def test_integration_reaches_tf_after_last_requested_sample(self) -> None:
		callback_times: list[float] = []
		solution = solve_symplectic(
			identity_flow,
			identity_flow,
			(0.0, 1.0),
			[1.0],
			step=0.25,
			t_eval=[0.0, 0.2],
			command=lambda t, _state: callback_times.append(t),
		)

		np.testing.assert_array_equal(solution.t, [0.0, 0.2])
		self.assertAlmostEqual(callback_times[-1], 1.0)
		self.assertEqual(solution.n_steps, 5)

	def test_step_count_uses_float_precision_not_relative_tolerance(self) -> None:
		self.assertEqual(_step_count(1_000_000_000_000.4, 1.0), 1_000_000_000_001)

	def test_rejects_invalid_steps(self) -> None:
		for step in (0.0, -1.0, np.nan, np.inf):
			with self.subTest(step=step):
				with self.assertRaisesRegex(ValueError, "positive finite"):
					solve_symplectic(identity_flow, identity_flow, (0.0, 1.0), [1.0], step=step)

	def test_rejects_invalid_save_steps(self) -> None:
		for save_step in (0.0, -1.0, np.nan, np.inf):
			with self.subTest(save_step=save_step):
				with self.assertRaisesRegex(ValueError, "positive finite"):
					solve_symplectic(
						identity_flow,
						identity_flow,
						(0.0, 1.0),
						[1.0],
						step=0.1,
						save_step=save_step,
					)

	def test_rejects_invalid_n_save_step(self) -> None:
		for n_save_step in (0, 1, -1, 2.5, True):
			with self.subTest(n_save_step=n_save_step):
				with self.assertRaises(ValueError):
					solve_symplectic(
						identity_flow,
						identity_flow,
						(0.0, 1.0),
						[1.0],
						step=0.1,
						n_save_step=n_save_step,  # type: ignore[arg-type]
					)

	def test_rejects_nonfinite_or_nonnumeric_initial_state(self) -> None:
		for initial in ([np.nan], [np.inf], ["not-a-number"]):
			with self.subTest(initial=initial):
				with self.assertRaises(ValueError):
					solve_symplectic(
						identity_flow,
						identity_flow,
						(0.0, 1.0),
						initial,
						step=0.1,
					)

	def test_rejects_invalid_evaluation_times(self) -> None:
		invalid_values: tuple[ArrayLike, ...] = (
			[],
			0.5,
			[0.2, 0.1],
			[-0.1, 0.2],
			[0.1, np.nan],
		)
		for t_eval in invalid_values:
			with self.subTest(t_eval=t_eval):
				with self.assertRaises(ValueError):
					solve_symplectic(
						identity_flow,
						identity_flow,
						(0.0, 1.0),
						[1.0],
						step=0.1,
						t_eval=t_eval,
					)

	def test_rejects_a_flow_that_changes_state_shape(self) -> None:
		def wrong_shape(_h: float, _t: float, state: np.ndarray) -> np.ndarray:
			return np.append(state, 0.0)

		with self.assertRaisesRegex(ValueError, "preserve the shape"):
			solve_symplectic(
				wrong_shape,
				wrong_shape,
				(0.0, 0.1),
				[1.0],
				step=0.1,
			)

	def test_extended_solver_uses_the_same_output_contract(self) -> None:
		solution = solve_extended(
			HarmonicOscillator(),
			(0.0, 0.1),
			[1.0, 0.0],
			step=0.01,
			t_eval=[0.05, 0.1],
		)

		self.assertIsInstance(solution, Solution)
		np.testing.assert_array_equal(solution.t, [0.05, 0.1])
		self.assertEqual(solution.y.shape, (2, 2))


if __name__ == "__main__":
	unittest.main()
