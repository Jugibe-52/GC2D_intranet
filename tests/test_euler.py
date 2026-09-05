"""Contracts for the classical explicit Euler method."""

from __future__ import annotations

import unittest

import numpy as np

from simulation import (
	DEFAULT_INTEGRATION_STEP,
	DEFAULT_INTEGRATION_STEPS_PER_CYCLE,
	DEFAULT_SAVED_STEPS_PER_CYCLE,
	DEFAULT_SAVE_INTERVAL,
	ExplicitEuler,
	SimulationRequest,
	simulate,
)
from tests.test_abba4_implicit import _rotation_problem


class ExplicitEulerTests(unittest.TestCase):
	"""Verify the public method's one-step forward map."""

	def test_default_request_uses_normalized_cycle_cadences(self) -> None:
		"""Use 10 integration steps and 10 saved intervals per unit cycle."""
		self.assertEqual(DEFAULT_INTEGRATION_STEPS_PER_CYCLE, 10)
		self.assertEqual(DEFAULT_SAVED_STEPS_PER_CYCLE, 10)
		self.assertEqual(DEFAULT_INTEGRATION_STEP, 0.1)
		self.assertEqual(DEFAULT_SAVE_INTERVAL, 0.1)
		request = SimulationRequest.uniform()
		self.assertEqual(request.t_span, (0.0, 1.0))
		self.assertEqual(request.max_step, DEFAULT_INTEGRATION_STEP)
		self.assertEqual(request.output_times.size, DEFAULT_SAVED_STEPS_PER_CYCLE + 1)
		np.testing.assert_allclose(np.diff(request.output_times), DEFAULT_SAVE_INTERVAL)

		solution = simulate(_rotation_problem(), ExplicitEuler(), request)
		self.assertEqual(solution.n_steps, DEFAULT_INTEGRATION_STEPS_PER_CYCLE)

		ten_cycles = SimulationRequest.uniform(t_span=(0.0, 10.0))
		self.assertEqual(
			ten_cycles.output_times.size,
			10 * DEFAULT_SAVED_STEPS_PER_CYCLE + 1,
		)
		np.testing.assert_allclose(np.diff(ten_cycles.output_times), DEFAULT_SAVE_INTERVAL)

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
