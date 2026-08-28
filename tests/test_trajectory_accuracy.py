"""Tests for certified references and ten-method numerical accuracy."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostics import load_reference_trajectory
from studies import (
	HighPrecisionReferenceConfig,
	RandomPotentialConfig,
	TenMethodTrajectoryComparisonConfig,
	periodic_particle_distances,
	random_gc_configuration,
	run_high_precision_reference_trajectory,
	run_ten_method_accuracy_refinement_study,
	run_ten_method_accuracy_study,
)
from visualization import (
	plot_accuracy_summary,
	plot_accuracy_runtime_tradeoff,
	plot_reference_trajectory_points,
	plot_ten_method_accuracy_over_time,
	plot_ten_method_accuracy_refinement,
	plot_ten_method_accuracy_summary,
)


class TrajectoryAccuracyTests(unittest.TestCase):
	"""Verify reference persistence, periodic errors, and all ten variants."""

	def test_reference_roundtrip_and_ten_method_accuracy(self) -> None:
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
			"x_bounds": [potential.grid.xmin, potential.grid.xmin + potential.grid.period],
			"y_bounds": [potential.grid.ymin, potential.grid.ymin + potential.grid.period],
		}
		configuration = random_gc_configuration(
			potential,
			particle_count=2,
			seed=41,
		)
		reference_config = HighPrecisionReferenceConfig(
			t_span=(0.0, 0.1),
			save_interval=0.025,
			rho=0.05,
			distance_convention="euclidean",
			relative_tolerance=1e-11,
			absolute_tolerance=1e-13,
			maximum_step=0.005,
			audit_relative_tolerance=1e-11,
			audit_absolute_tolerance=1e-13,
			audit_maximum_step=0.0025,
		)
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			(root / "pyproject.toml").write_text("[project]\nname='test'\n")
			notebook = root / "notebooks/developements/accuracy/create_reference.ipynb"
			result = run_high_precision_reference_trajectory(
				potential,
				configuration,
				notebook_path=notebook,
				config=reference_config,
				potential_metadata=potential_config.metadata(),
				initial_condition_metadata=initial_metadata,
				project_root=root,
			)
			expected_directory = (
				root
				/ "outputs/developements/accuracy/example_trajectory/v1"
			)
			self.assertEqual(result.trajectory.paths.directory, expected_directory.resolve())
			self.assertTrue(result.trajectory.paths.trajectory.is_file())
			self.assertTrue(result.trajectory.paths.metadata.is_file())
			self.assertTrue(result.trajectory.paths.readme.is_file())
			self.assertIn("SciPy Radau", result.trajectory.paths.readme.read_text())
			loaded = load_reference_trajectory(expected_directory)
			np.testing.assert_array_equal(loaded.times, result.trajectory.times)
			np.testing.assert_array_equal(loaded.states, result.trajectory.states)
			self.assertFalse(loaded.states.flags.writeable)
			self.assertEqual(loaded.audit_distances.shape, (2, 5))
			self.assertEqual(loaded.metadata["schema_version"], 2)
			self.assertEqual(
				loaded.metadata["config"]["distance_convention"],
				"euclidean",
			)
			with np.load(loaded.paths.trajectory, allow_pickle=False) as archive:
				self.assertIn("audit_distances", archive.files)
				self.assertNotIn("audit_periodic_distances", archive.files)
			self.assertIn("dynamics_fingerprint_sha256", loaded.metadata)
			self.assertFalse(any(expected_directory.glob(".*-*.npz")))
			with self.assertRaisesRegex(FileExistsError, "versioned reference"):
				run_high_precision_reference_trajectory(
					potential,
					configuration,
					notebook_path=notebook,
					config=reference_config,
					potential_metadata=potential_config.metadata(),
					initial_condition_metadata=initial_metadata,
					project_root=root,
				)

			comparison_config = TenMethodTrajectoryComparisonConfig(
				rho=0.05,
				t_span=(0.0, 0.1),
				integration_step=0.05,
				save_interval=0.05,
			)
			accuracy = run_ten_method_accuracy_study(
				potential,
				configuration,
				loaded,
				config=comparison_config,
				potential_metadata=potential_config.metadata(),
				initial_condition_metadata=initial_metadata,
			)
			self.assertEqual(len(accuracy.series), 10)
			summaries = accuracy.summaries()
			self.assertEqual(len(summaries), 10)
			summary_by_name = {row.method_name: row for row in summaries}
			self.assertLess(
				summary_by_name["Midpoint BM4"].global_rms_distance,
				summary_by_name["Midpoint ABBA"].global_rms_distance,
			)
			for series in accuracy.series.values():
				self.assertEqual(series.distances.shape, (2, 3))
				np.testing.assert_array_equal(series.distances[:, 0], 0.0)
				self.assertTrue(np.all(series.distances >= 0.0))
			self.assertAlmostEqual(
				accuracy.reference_floor,
				float(np.sqrt(np.mean(loaded.audit_distances[:, ::2] ** 2))),
			)
			first_label = next(iter(accuracy.series))
			first_states = accuracy.comparison.solutions[first_label].states
			first_difference = first_states - loaded.states[:, ::2]
			expected_distances = np.hypot(
				first_difference[:2],
				first_difference[2:],
			)
			np.testing.assert_allclose(
				accuracy.series[first_label].distances,
				expected_distances,
				rtol=0.0,
				atol=0.0,
			)

			trajectory_figure, trajectory_axis = plot_reference_trajectory_points(loaded)
			self.assertEqual(len(trajectory_axis.lines), 2)
			trajectory_figure.canvas.draw()
			plt.close(trajectory_figure)
			time_figure, time_axes = plot_ten_method_accuracy_over_time(
				accuracy.times,
				accuracy.series,
				reference_floor=accuracy.reference_floor,
			)
			self.assertEqual(time_axes.shape, (2,))
			time_figure.canvas.draw()
			plt.close(time_figure)
			summary_figure, summary_axis = plot_ten_method_accuracy_summary(
				summaries
			)
			self.assertEqual(len(summary_axis.patches), 20)
			summary_figure.canvas.draw()
			plt.close(summary_figure)
			tradeoff_figure, tradeoff_axis = plot_accuracy_runtime_tradeoff(
				summaries
			)
			self.assertEqual(len(tradeoff_axis.collections), 10)
			tradeoff_figure.canvas.draw()
			plt.close(tradeoff_figure)
			eleven_summaries = (
				*summaries,
				replace(summaries[0], method_name="Additional method"),
			)
			eleven_figure, eleven_axis = plot_accuracy_summary(eleven_summaries)
			self.assertEqual(len(eleven_axis.patches), 22)
			eleven_figure.canvas.draw()
			plt.close(eleven_figure)
			eleven_tradeoff_figure, eleven_tradeoff_axis = (
				plot_accuracy_runtime_tradeoff(eleven_summaries)
			)
			self.assertEqual(len(eleven_tradeoff_axis.collections), 11)
			eleven_tradeoff_figure.canvas.draw()
			plt.close(eleven_tradeoff_figure)

			refinement = run_ten_method_accuracy_refinement_study(
				potential,
				configuration,
				loaded,
				base_config=comparison_config,
				integration_steps=(0.05, 0.025),
				potential_metadata=potential_config.metadata(),
				initial_condition_metadata=initial_metadata,
			)
			self.assertEqual(refinement.integration_steps, (0.05, 0.025))
			self.assertEqual(len(refinement.summaries()), 20)
			orders = refinement.convergence_orders()
			self.assertEqual(len(orders), 10)
			audit_distances = loaded.audit_distances[:, ::2]
			expected_floor = float(
				np.sqrt(
					np.trapz(np.mean(audit_distances**2, axis=0), loaded.times[::2])
					/ 0.1
				)
			)
			self.assertAlmostEqual(refinement.reference_floor, expected_floor)
			resolved_orders = tuple(
				row for row in orders if row.resolved_above_reference_floor
			)
			self.assertTrue(resolved_orders)
			for row in resolved_orders:
				self.assertAlmostEqual(
					row.time_integrated_rms_order,
					float(np.log(row.time_integrated_rms_gain) / np.log(2.0)),
				)
				self.assertAlmostEqual(
					row.final_rms_order,
					float(np.log(row.final_rms_gain) / np.log(2.0)),
				)
			for step, expected_step_count in ((0.05, 2), (0.025, 4)):
				run = refinement.results[step]
				np.testing.assert_array_equal(run.times, loaded.times[::2])
				for solution in run.comparison.solutions.values():
					self.assertEqual(
						int(solution.diagnostics["step_count"]),
						expected_step_count,
					)
			refinement_figure, refinement_axes = (
				plot_ten_method_accuracy_refinement(refinement.summaries())
			)
			self.assertEqual(refinement_axes.shape, (2,))
			self.assertEqual([len(axis.lines) for axis in refinement_axes], [5, 5])
			refinement_figure.canvas.draw()
			plt.close(refinement_figure)
			with self.assertRaisesRegex(ValueError, "coarse step / fine step"):
				run_ten_method_accuracy_refinement_study(
					potential,
					configuration,
					loaded,
					base_config=TenMethodTrajectoryComparisonConfig(
						rho=0.05,
						t_span=(0.0, 0.1),
						integration_step=0.05,
						save_interval=0.1,
					),
					integration_steps=(0.05, 0.02),
					potential_metadata=potential_config.metadata(),
					initial_condition_metadata=initial_metadata,
				)

			with self.assertRaisesRegex(ValueError, "Potential metadata"):
				run_ten_method_accuracy_study(
					potential,
					configuration,
					loaded,
					config=comparison_config,
					potential_metadata={"seed": -1},
					initial_condition_metadata=initial_metadata,
				)

			wrong_potential = RandomPotentialConfig(
				amplitude=0.02,
				max_wave_number=2,
				nx=16,
				ny=16,
				seed=28,
				interpolation_order=3,
			).build()
			with self.assertRaisesRegex(ValueError, "interpolated ODE"):
				run_ten_method_accuracy_study(
					wrong_potential,
					configuration,
					loaded,
					config=comparison_config,
					potential_metadata=potential_config.metadata(),
					initial_condition_metadata=initial_metadata,
				)

	def test_periodic_distance_uses_the_nearest_cell_image(self) -> None:
		period = 2.0 * np.pi
		states = np.asarray([[0.01], [0.25]])
		reference = np.asarray([[period - 0.01], [0.25]])
		distances = periodic_particle_distances(states, reference, period=period)
		np.testing.assert_allclose(distances, [[0.02]], rtol=0.0, atol=1e-15)

	def test_reference_requires_a_stricter_radau_step(self) -> None:
		with self.assertRaisesRegex(ValueError, "Radau audit maximum step"):
			HighPrecisionReferenceConfig(
				maximum_step=0.0025,
				audit_maximum_step=0.005,
			)

	def test_reference_rejects_looser_radau_tolerances(self) -> None:
		with self.assertRaisesRegex(ValueError, "relative tolerance"):
			HighPrecisionReferenceConfig(
				relative_tolerance=1e-13,
				audit_relative_tolerance=1e-10,
			)
		with self.assertRaisesRegex(ValueError, "absolute tolerance"):
			HighPrecisionReferenceConfig(
				absolute_tolerance=1e-15,
				audit_absolute_tolerance=1e-12,
			)

	def test_reference_rejects_an_unknown_distance_convention(self) -> None:
		with self.assertRaisesRegex(ValueError, "distance_convention"):
			HighPrecisionReferenceConfig(
				distance_convention="wrapped",  # type: ignore[arg-type]
			)


if __name__ == "__main__":
	unittest.main()
