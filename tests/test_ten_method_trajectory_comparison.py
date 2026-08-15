"""Tests for the aligned ten-method trajectory comparison."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from studies import (
	TEN_METHOD_LABELS,
	RandomPotentialConfig,
	TenMethodTrajectoryComparisonConfig,
	random_gc_configuration,
	run_ten_method_trajectory_comparison,
)
from visualization import (
	animate_ten_method_trajectory_points,
	plot_ten_method_nonlinear_work,
	plot_ten_method_runtimes,
	plot_ten_method_trajectory_differences,
)


class TenMethodTrajectoryComparisonTests(unittest.TestCase):
	"""Verify aligned execution, all-pairs metrics, and public views."""

	def test_all_ten_variants_share_the_problem_and_time_grid(self) -> None:
		potential = RandomPotentialConfig(
			amplitude=0.02,
			max_wave_number=2,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=3,
		).build()
		configuration = random_gc_configuration(
			potential,
			particle_count=3,
			seed=41,
		)
		config = TenMethodTrajectoryComparisonConfig(
			rho=0.05,
			t_span=(0.0, 0.05),
			integration_step=0.05,
		)
		result = run_ten_method_trajectory_comparison(
			potential,
			configuration,
			config=config,
		)

		self.assertEqual(tuple(result.solutions), TEN_METHOD_LABELS)
		self.assertEqual(len(result.solutions), 10)
		reference_times = next(iter(result.solutions.values())).t
		for solution in result.solutions.values():
			self.assertIs(solution.source, configuration)
			np.testing.assert_array_equal(solution.t, reference_times)
			self.assertEqual(solution.states.shape, (6, 2))
		self.assertEqual(len(result.trajectory_difference_summaries()), 45)
		self.assertEqual(len(result.runtime_summaries()), 10)
		self.assertEqual(len(result.nonlinear_work_summaries()), 8)
		self.assertEqual(len(result.implicit_solutions), 8)
		self.assertEqual(
			{row.nonlinear_solver for row in result.nonlinear_work_summaries()},
			{"newton", "broyden"},
		)

		difference_figure, difference_axis = plot_ten_method_trajectory_differences(
			result.effective_potential,
			result.solutions,
		)
		self.assertEqual(
			np.asarray(difference_axis.images[0].get_array()).shape,
			(10, 10),
		)
		self.assertEqual(len(difference_axis.texts), 100)
		difference_figure.canvas.draw()
		plt.close(difference_figure)

		work_figure, work_axes = plot_ten_method_nonlinear_work(
			result.implicit_solutions
		)
		self.assertEqual(work_axes.shape, (3,))
		work_figure.canvas.draw()
		plt.close(work_figure)

		runtime_figure, runtime_axis = plot_ten_method_runtimes(result.runtimes)
		self.assertEqual(len(runtime_axis.patches), 10)
		runtime_figure.canvas.draw()
		plt.close(runtime_figure)

		animation = animate_ten_method_trajectory_points(
			result.effective_potential,
			result.solutions,
			frames=2,
			interval=10,
		)
		self.assertEqual(len(animation._func(1)), 21)
		animation._draw_was_started = True
		plt.close(animation._fig)

	def test_config_requires_an_integral_common_grid(self) -> None:
		with self.assertRaisesRegex(ValueError, "duration / integration_step"):
			TenMethodTrajectoryComparisonConfig(
				t_span=(0.0, 1.0),
				integration_step=0.3,
			)


if __name__ == "__main__":
	unittest.main()
