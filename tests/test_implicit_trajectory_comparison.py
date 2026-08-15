"""Reusable comparison of ten trajectories across four implicit methods."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from studies import (
	IMPLICIT_METHOD_NAMES,
	ImplicitTrajectoryComparisonConfig,
	RandomPotentialConfig,
	random_gc_configuration,
	run_implicit_trajectory_comparison,
)
from visualization import (
	animate_implicit_method_trajectories,
	plot_implicit_method_iterations,
	plot_implicit_trajectory_differences,
	records_table_html,
)


class ImplicitTrajectoryComparisonTests(unittest.TestCase):
	"""Verify aligned runs, pairwise distances, work summaries, and views."""

	def test_ten_particle_broyden_comparison_is_aligned(self) -> None:
		potential = RandomPotentialConfig(
			amplitude=0.08,
			max_wave_number=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=3,
		).build()
		grid = potential.grid
		configuration = random_gc_configuration(
			potential,
			particle_count=10,
			seed=20260815,
			x_bounds=(grid.xmin + 0.2 * grid.period, grid.xmin + 0.8 * grid.period),
			y_bounds=(grid.ymin + 0.2 * grid.period, grid.ymin + 0.8 * grid.period),
		)
		config = ImplicitTrajectoryComparisonConfig(
			rho=0.05,
			t_span=(0.0, 0.1),
			integration_step=0.05,
		)
		result = run_implicit_trajectory_comparison(
			potential,
			configuration,
			config=config,
		)

		self.assertEqual(tuple(result.solutions), IMPLICIT_METHOD_NAMES)
		self.assertEqual(configuration.particle_count(configuration.initial_state), 10)
		for solution in result.solutions.values():
			self.assertEqual(solution.states.shape, (20, 3))
			self.assertEqual(solution.diagnostics["nonlinear_solver"], "broyden")
			self.assertEqual(
				np.asarray(solution.diagnostics["nonlinear_iterations"]).shape,
				(2,),
			)

		differences = result.trajectory_difference_summaries()
		self.assertEqual(len(differences), 6)
		self.assertTrue(all(row.maximum_distance >= 0.0 for row in differences))
		self.assertEqual(len(result.iteration_summaries()), 4)
		self.assertTrue(
			all(row.step_count == config.step_count for row in result.iteration_summaries())
		)

		html = records_table_html(
			result.trajectory_difference_summaries(),
			columns=(
				("first_method", "First method", None),
				("maximum_distance", "Maximum distance", ".3e"),
			),
		)
		self.assertIn("ImplicitABBA1", html)
		figure, axes = plot_implicit_method_iterations(result.solutions)
		self.assertEqual(axes.shape, (3,))
		figure.canvas.draw()
		plt.close(figure)
		difference_figure, difference_axis = plot_implicit_trajectory_differences(
			result.effective_potential,
			result.solutions,
		)
		self.assertEqual(np.asarray(difference_axis.images[0].get_array()).shape, (4, 4))
		self.assertEqual(len(difference_axis.texts), 16)
		difference_figure.canvas.draw()
		plt.close(difference_figure)
		animation = animate_implicit_method_trajectories(
			result.effective_potential,
			result.solutions,
			frames=2,
			interval=10,
		)
		self.assertEqual(len(animation._func(1)), 9)
		animation._draw_was_started = True
		plt.close(animation._fig)

	def test_random_configuration_is_seed_reproducible(self) -> None:
		potential = RandomPotentialConfig(
			amplitude=0.08,
			max_wave_number=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=3,
		).build()
		first = random_gc_configuration(
			potential, particle_count=10, seed=17
		)
		second = random_gc_configuration(
			potential, particle_count=10, seed=17
		)
		np.testing.assert_array_equal(first.initial_state, second.initial_state)


if __name__ == "__main__":
	unittest.main()
