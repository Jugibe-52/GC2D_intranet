"""Contracts for the five-method SDIRK and classical-RK4 comparison."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import unittest

import matplotlib.pyplot as plt
import numpy as np

from studies import (
	FIVE_METHOD_COMPARISON_METHODS,
	FIVE_METHOD_IMPLICIT_METHODS,
	FiveMethodComparisonConfig,
	RandomPotentialConfig,
	latin_hypercube_gc_configuration,
	run_five_method_comparison,
)
from visualization import (
	animate_implicit_method_trajectories,
	plot_runtime_comparison,
)


class FiveMethodComparisonTests(unittest.TestCase):
	"""Verify RK4 inclusion, live logs, summaries, and runtime plots."""

	def test_short_five_method_study_with_progress_logs(self) -> None:
		potential = RandomPotentialConfig(
			amplitude=0.2,
			max_wave_number=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=5,
		).build()
		initial_configuration = latin_hypercube_gc_configuration(
			potential,
			particle_count=3,
			seed=20260905,
			domain_margin_fraction=0.35,
		)
		config = FiveMethodComparisonConfig(
			rho=0.3,
			t_span=(0.0, 0.2),
			integration_step=0.1,
			save_interval=0.1,
			absolute_tolerance=1e-13,
			relative_tolerance=1e-12,
			max_iterations=20,
			reference_relative_tolerance=1e-12,
			reference_absolute_tolerance=1e-14,
			reference_maximum_step=0.02,
			audit_relative_tolerance=1e-12,
			audit_absolute_tolerance=1e-14,
			audit_maximum_step=0.01,
			timing_warmups=0,
			timing_repeats=1,
			distance_convention="euclidean",
			progress=True,
		)
		progress_output = io.StringIO()
		with redirect_stderr(progress_output):
			result = run_five_method_comparison(
				potential,
				initial_configuration,
				config=config,
			)

		self.assertEqual(
			FIVE_METHOD_COMPARISON_METHODS,
			(
				"ABBA4ImplicitSingleProjection",
				"GaussLegendre4",
				"BM4Implicit",
				"SDIRK4",
				"RK4",
			),
		)
		self.assertEqual(FIVE_METHOD_IMPLICIT_METHODS, FIVE_METHOD_COMPARISON_METHODS[:-1])
		self.assertEqual(tuple(result.solutions), FIVE_METHOD_COMPARISON_METHODS)
		self.assertEqual(tuple(result.accuracy), FIVE_METHOD_COMPARISON_METHODS)
		self.assertEqual(tuple(result.energy_accuracy), FIVE_METHOD_COMPARISON_METHODS)
		self.assertEqual(tuple(result.runtime_samples), FIVE_METHOD_COMPARISON_METHODS)
		self.assertEqual(len(result.execution_log), 6)
		self.assertEqual(result.execution_log[0].phase, "reference")
		self.assertTrue(all(entry.trajectory_count == 3 for entry in result.execution_log))
		self.assertNotIn("nonlinear_solver", result.solutions["RK4"].diagnostics)
		self.assertEqual(result.solutions["RK4"].diagnostics["step_count"], 2)

		log = progress_output.getvalue()
		self.assertIn("Starting DOP853 reference and tighter Radau audit", log)
		self.assertIn("Starting timing 1/1: Classical explicit RK4", log)
		self.assertIn("3 trajectories together, 2 steps", log)
		self.assertIn("campaign 5/5 (100.0%)", log)
		self.assertIn("Completed study", log)

		summaries = result.summaries()
		self.assertEqual(len(summaries), 5)
		self.assertEqual(tuple(row.method_name for row in summaries), FIVE_METHOD_COMPARISON_METHODS)
		self.assertEqual(summaries[-1].solver, "Explicit")
		self.assertTrue(all(row.trajectory_count == 3 for row in summaries))
		self.assertTrue(all(row.step_count == 2 for row in summaries))
		self.assertTrue(all(row.runtime_seconds > 0.0 for row in summaries))

		work = result.nonlinear_work_summaries()
		self.assertEqual(tuple(row.method_name for row in work), FIVE_METHOD_IMPLICIT_METHODS)
		self.assertNotIn("RK4", tuple(row.method_name for row in work))
		self.assertTrue(all(row.maximum_residual_to_tolerance <= 1.0 for row in work))

		runtime_figure, runtime_axes = plot_runtime_comparison(summaries)
		self.assertEqual(runtime_axes.shape, (2,))
		self.assertEqual(len(runtime_axes[0].patches), 5)
		self.assertEqual(len(runtime_axes[1].patches), 5)
		animation = animate_implicit_method_trajectories(
			result.effective_potential,
			result.solutions,
			frames=2,
			interval=10,
			repeat=False,
			title_family="numerical methods",
		)
		self.assertEqual(len(animation._func(1)), 11)
		self.assertIn("Five numerical methods", animation._fig.axes[0].get_title())
		animation._draw_was_started = True
		plt.close(runtime_figure)
		plt.close(animation._fig)


if __name__ == "__main__":
	unittest.main()
