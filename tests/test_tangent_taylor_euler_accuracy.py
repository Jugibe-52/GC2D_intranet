"""Classical Euler and tangent-Taylor accuracy comparison tests."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from initial_conditions import GCInitialConfiguration
from simulation import ExplicitEuler, SimulationRequest, simulate
from studies import (
	RandomPotentialConfig,
	TANGENT_TAYLOR_EULER_METHOD_NAMES,
	TangentTaylorEulerAccuracyConfig,
	run_tangent_taylor_euler_accuracy_study,
)
from tests.test_abba4_implicit import _rotation_problem
from visualization import (
	plot_tangent_taylor_euler_accuracy,
	plot_tangent_taylor_h_error,
)


class ExplicitEulerTests(unittest.TestCase):
	"""Verify the public method and the certified three-method study."""

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
		np.testing.assert_allclose(solution.states[:, -1], expected, atol=0.0, rtol=0.0)
		self.assertEqual(solution.n_steps, 1)

	def test_accuracy_study_runs_nested_grids_and_plots(self) -> None:
		potential = RandomPotentialConfig(
			amplitude=0.04,
			max_wave_number=2,
			nx=12,
			ny=12,
			seed=27,
			interpolation_order=3,
		).build()
		configuration = GCInitialConfiguration.from_components(
			x=np.asarray([1.0, 1.1]),
			y=np.asarray([1.2, 1.3]),
		)
		config = TangentTaylorEulerAccuracyConfig(
			rho=0.05,
			t_span=(0.0, 0.04),
			step_counts=(2, 4),
			newton_absolute_tolerance=1e-14,
			newton_relative_tolerance=1e-14,
			reference_maximum_step=0.01,
			audit_maximum_step=0.005,
		)
		result = run_tangent_taylor_euler_accuracy_study(
			potential,
			configuration,
			config=config,
		)
		self.assertEqual(tuple(result.runs), (2, 4))
		self.assertEqual(tuple(result.finest_runs), TANGENT_TAYLOR_EULER_METHOD_NAMES)
		self.assertEqual(len(result.summaries()), 6)
		self.assertEqual(len(result.convergence_orders()), 3)
		self.assertEqual(result.reference.states.shape, (4, 3))
		figure, axes = plot_tangent_taylor_euler_accuracy(result)
		self.assertEqual(axes.shape, (2,))
		figure.canvas.draw()
		plt.close(figure)
		figure, axes = plot_tangent_taylor_h_error(result)
		self.assertEqual(axes.shape, (2,))
		figure.canvas.draw()
		plt.close(figure)


if __name__ == "__main__":
	unittest.main()
