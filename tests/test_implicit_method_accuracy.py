"""Tests for four-method implicit accuracy and Newton refinement."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from studies import (
	IMPLICIT_ACCURACY_METHOD_NAMES,
	HighPrecisionReferenceConfig,
	ImplicitMethodAccuracyConfig,
	RandomPotentialConfig,
	random_gc_configuration,
	run_high_precision_reference_trajectory,
	run_implicit_method_accuracy_study,
)
from visualization import (
	plot_implicit_method_accuracy_cost,
	plot_implicit_method_accuracy_refinement,
	plot_implicit_method_newton_refinement,
	plot_implicit_method_observed_orders,
)


class ImplicitMethodAccuracyTests(unittest.TestCase):
	"""Verify aligned errors, Newton metrics, orders, and plots."""

	def test_short_four_method_refinement(self) -> None:
		potential_config = RandomPotentialConfig(
			amplitude=0.02,
			max_wave_number=2,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=3,
		)
		potential = potential_config.build()
		initial_metadata = {
			"particle_count": 2,
			"seed": 41,
			"sampling": "numpy.default_rng.uniform_half_open",
		}
		configuration = random_gc_configuration(
			potential,
			particle_count=2,
			seed=41,
		)
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			(root / "pyproject.toml").write_text("[project]\nname='test'\n")
			reference = run_high_precision_reference_trajectory(
				potential,
				configuration,
				notebook_path=(
					root / "notebooks/developements/accuracy/create_reference.ipynb"
				),
				config=HighPrecisionReferenceConfig(
					t_span=(0.0, 0.1),
					save_interval=0.025,
					rho=0.05,
					relative_tolerance=1e-11,
					absolute_tolerance=1e-13,
					maximum_step=0.005,
					audit_relative_tolerance=1e-11,
					audit_absolute_tolerance=1e-13,
					audit_maximum_step=0.0025,
					distance_convention="euclidean",
				),
				potential_metadata=potential_config.metadata(),
				initial_condition_metadata=initial_metadata,
				project_root=root,
			).trajectory

			config = ImplicitMethodAccuracyConfig(
				integration_steps=(0.05, 0.025),
				t_span=(0.0, 0.1),
				save_interval=0.05,
				rho=0.05,
				coupling_frequency=0.2,
				absolute_tolerance=1e-12,
				relative_tolerance=1e-11,
				max_iterations=20,
			)
			result = run_implicit_method_accuracy_study(
				potential,
				configuration,
				reference,
				config=config,
				potential_metadata=potential_config.metadata(),
				initial_condition_metadata=initial_metadata,
			)

		self.assertEqual(tuple(result.solutions), IMPLICIT_ACCURACY_METHOD_NAMES)
		self.assertEqual(result.times.shape, (3,))
		self.assertEqual(len(result.summaries()), 8)
		self.assertEqual(len(result.convergence_orders()), 4)
		self.assertEqual(tuple(result.finest_series), IMPLICIT_ACCURACY_METHOD_NAMES)
		for row in result.summaries():
			self.assertLessEqual(row.maximum_residual_to_tolerance, 1.0)
			self.assertGreaterEqual(row.mean_iterations_per_solve, 0.0)
			self.assertAlmostEqual(
				row.zero_iteration_fraction
				+ row.one_iteration_fraction
				+ row.two_iteration_fraction
				+ row.three_or_more_iteration_fraction,
				1.0,
			)
			frequencies = result.iteration_frequencies(
				row.method_name,
				row.integration_step,
			)
			self.assertEqual(
				sum(frequencies.values()),
				row.step_count * row.nonlinear_solves_per_step,
			)
		for method_name in IMPLICIT_ACCURACY_METHOD_NAMES:
			for step in config.integration_steps:
				series = result.series[method_name][step]
				self.assertEqual(series.distances.shape, (2, 3))
				np.testing.assert_array_equal(series.distances[:, 0], 0.0)

		accuracy_figure, accuracy_axis = plot_implicit_method_accuracy_refinement(
			result.summaries(), reference_floor=result.reference_floor
		)
		order_figure, order_axes = plot_implicit_method_observed_orders(
			result.convergence_orders()
		)
		newton_figure, newton_axes = plot_implicit_method_newton_refinement(
			result.summaries()
		)
		cost_figure, cost_axis = plot_implicit_method_accuracy_cost(
			result.summaries()
		)
		self.assertGreaterEqual(len(accuracy_axis.lines), 7)
		order_two_guide = next(
			line
			for line in accuracy_axis.lines
			if line.get_label().startswith("$O(h^{2})$")
		)
		order_four_guide = next(
			line
			for line in accuracy_axis.lines
			if line.get_label().startswith("$O(h^{4})$")
		)
		self.assertIn("Implicit ABBA", order_two_guide.get_label())
		self.assertIn("Implicit ABBA4", order_four_guide.get_label())
		self.assertLessEqual(len(order_two_guide.get_xdata()), 3)
		self.assertLessEqual(len(order_four_guide.get_xdata()), 3)
		self.assertEqual(order_axes.shape, (2,))
		self.assertEqual(newton_axes.shape, (2, 2))
		self.assertGreaterEqual(len(cost_axis.lines), 4)
		for figure in (accuracy_figure, order_figure, newton_figure, cost_figure):
			figure.canvas.draw()
			plt.close(figure)

	def test_steps_must_align_with_saved_times(self) -> None:
		with self.assertRaisesRegex(ValueError, "save_interval / integration step"):
			ImplicitMethodAccuracyConfig(
				integration_steps=(0.004, 0.002),
				t_span=(0.0, 0.1),
				save_interval=0.01,
			)

	def test_default_grid_adds_one_more_exact_halving(self) -> None:
		config = ImplicitMethodAccuracyConfig()
		self.assertEqual(config.integration_steps[-1], 0.00015625)
		self.assertEqual(config.integration_steps[-2] / config.integration_steps[-1], 2.0)
		self.assertEqual(config.step_count(config.integration_steps[-1]), 25_600)


if __name__ == "__main__":
	unittest.main()
