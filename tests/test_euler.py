"""Contracts for the classical explicit Euler method."""

from __future__ import annotations

import unittest

import numpy as np

from simulation import ExplicitEuler, SimulationRequest, simulate
from tests.test_abba4_implicit import _rotation_problem


class ExplicitEulerTests(unittest.TestCase):
	"""Verify the public method's one-step forward map."""

	def test_one_step_is_the_classical_forward_euler_map(self) -> None:
		problem = _rotation_problem()
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.1),
			max_step=0.1,
			sample_count=2,
		)
		initial = problem.initial_state
		expected = initial + 0.1 * problem.dynamics.vector_field(0.0, initial)
		solution = simulate(problem, ExplicitEuler(), request)
		np.testing.assert_allclose(
			solution.states[:, -1],
			expected,
			atol=0.0,
			rtol=0.0,
		)
		self.assertEqual(solution.n_steps, 1)


if __name__ == "__main__":
	unittest.main()
