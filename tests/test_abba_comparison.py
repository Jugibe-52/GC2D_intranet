"""Contracts for the single-run comparison of all three ABBA methods."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from studies import (
	ABBA_METHOD_NAMES,
	ABBAComparisonConfig,
	RandomPotentialConfig,
	centered_square,
	run_abba_comparison,
)


class ABBAComparisonStudyTests(unittest.TestCase):
	"""Verify timing, trajectory metrics, and per-method animation contracts."""

	def test_one_run_per_method_returns_aligned_comparison(self) -> None:
		potential = RandomPotentialConfig(
			amplitude=0.08,
			max_wave_number=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=3,
		).build()
		area = centered_square(
			potential,
			side=0.5,
			points_per_side=1,
			rho=0.05,
		)
		config = ABBAComparisonConfig(
			integration_step=np.pi / 400,
			step_label=r"$\Delta t=\pi/400$",
			t_span=(0.0, np.pi / 100),
			save_interval=np.pi / 200,
			chunk_size=2,
			finite_difference_relative_step=float(
				np.cbrt(np.finfo(float).eps)
			),
		)

		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			result = run_abba_comparison(
				potential,
				area,
				notebook_path=(
					root
					/ "notebooks"
					/ "experiments"
					/ "abba_comparison"
					/ "comparison.ipynb"
				),
				config=config,
				project_root=root,
			)

		self.assertEqual(tuple(result.studies), ABBA_METHOD_NAMES)
		self.assertEqual(tuple(result.solutions), ABBA_METHOD_NAMES)
		self.assertEqual(len(result.runtime_summaries()), 3)
		self.assertTrue(all(row.seconds > 0 for row in result.runtime_summaries()))
		for study in result.studies.values():
			self.assertEqual(tuple(study.solutions), (config.step_label,))
			self.assertGreater(
				study.simulation_runtime_seconds[config.step_label],
				0.0,
			)
			self.assertGreater(
				study.symplecticity_runtime_seconds[config.step_label],
				0.0,
			)
			self.assertEqual(
				study.solutions[config.step_label].states.shape[1],
				config.output_sample_count,
			)
		for method_name in ABBA_METHOD_NAMES:
			self.assertEqual(
				result.runtimes[method_name],
				result.studies[method_name]
				.simulation_runtime_seconds[config.step_label],
			)

		# Both nonlinear formulations converge to the same projected physical map.
		np.testing.assert_allclose(
			result.solutions["ImplicitABBA1"].states,
			result.solutions["ImplicitABBA2"].states,
			rtol=0.0,
			atol=1e-14,
		)
		differences = result.trajectory_difference_summaries()
		self.assertEqual(len(differences), 3)
		implicit_pair = next(
			row
			for row in differences
			if row.first_method == "ImplicitABBA1"
			and row.second_method == "ImplicitABBA2"
		)
		self.assertLess(implicit_pair.max_distance, 1e-14)

		runtime_figure, runtime_axis = result.plot_runtime_comparison()
		difference_figure, difference_axis = result.plot_trajectory_differences()
		self.assertEqual(len(runtime_axis.patches), 3)
		self.assertEqual(len(difference_axis.lines), 6)
		animation = result.animate("ImplicitABBA2", frames=2, interval=10)
		self.assertGreater(len(animation._func(1)), 0)
		animation._draw_was_started = True
		plt.close(runtime_figure)
		plt.close(difference_figure)

	def test_sampling_grid_must_be_integral(self) -> None:
		with self.assertRaisesRegex(ValueError, "positive integer ratio"):
			ABBAComparisonConfig(
				integration_step=0.03,
				step_label="step",
				t_span=(0.0, 0.1),
				save_interval=0.05,
			)


if __name__ == "__main__":
	unittest.main()
