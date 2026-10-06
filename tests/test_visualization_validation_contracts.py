"""Resource and input-policy contracts for prepared visualization data."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import matplotlib.pyplot as plt
import numpy as np

from initial_conditions import Area, GCInitialConfiguration
from potential import Potential
from solution import Solution
from visualization.gauss_legendre4 import plot_gauss_legendre4_energy
from visualization.gc_area import animate_gc_area_solution
from visualization.implicit_comparison import plot_implicit_method_iterations
from visualization.particles import animate_gc_particle_solution
from visualization.poincare_comparison import export_poincare_panel_comparison
from visualization.potential import animate_potential
from visualization.trajectory_accuracy import plot_trajectory_accuracy_over_time
from visualization.trajectory_symplecticity import plot_trajectory_symplecticity


class VisualizationValidationContracts(unittest.TestCase):
	"""Reject incompatible data before creating figures or publishing HTML."""

	def tearDown(self) -> None:
		plt.close("all")

	def test_later_accuracy_and_energy_errors_do_not_leave_partial_figures(self) -> None:
		plt.figure()
		before = plt.get_fignums()
		times = np.array([0.0, 1.0])
		good = SimpleNamespace(rms_distance=np.ones(2), maximum_distance=np.ones(2))
		bad = SimpleNamespace(rms_distance=np.ones(2), maximum_distance=np.array([1.0, -1.0]))
		with self.assertRaisesRegex(ValueError, "finite and non-negative"):
			plot_trajectory_accuracy_over_time(times, {"first": good, "second": bad}, reference_floor=0.0)
		self.assertEqual(plt.get_fignums(), before)
		with self.assertRaisesRegex(ValueError, "align with saved times"):
			plot_gauss_legendre4_energy({0.1: times, 0.05: times}, {0.1: np.ones(2), 0.05: np.ones(3)})
		self.assertEqual(plt.get_fignums(), before)

	def test_symplecticity_alignment_fails_before_figure_creation(self) -> None:
		def record(time: float) -> SimpleNamespace:
			return SimpleNamespace(
				time=time, mean_local_relative_defect=1e-8, std_local_relative_defect=0.0,
				mean_accumulated_relative_defect=1e-7, std_accumulated_relative_defect=0.0,
			)
		with self.assertRaisesRegex(ValueError, "share one saved-time grid"):
			plot_trajectory_symplecticity(
				{"first": [record(0.0), record(1.0)], "second": [record(0.0), record(2.0)]},
				method_name="ABBA",
			)
		self.assertEqual(plt.get_fignums(), [])

	def test_iteration_history_alignment_fails_before_figure_creation(self) -> None:
		times = np.array([0.0, 1.0, 2.0])
		states = np.array([[1.0, 1.1, 1.2], [1.2, 1.3, 1.4]])
		source = GCInitialConfiguration(states[:, 0])
		good = dict(nonlinear_iterations=np.ones(2), residual_evaluations=np.ones(2),
			nonlinear_residual_norms=np.ones(2), nonlinear_tolerances=np.ones(2))
		bad = dict(good, nonlinear_iterations=np.ones(1))
		with self.assertRaisesRegex(ValueError, "complete integration step count"):
			plot_implicit_method_iterations({
				"first": Solution(t=times, states=states, source=source, diagnostics=good),
				"second": Solution(t=times, states=states, source=source, diagnostics=bad),
			})
		self.assertEqual(plt.get_fignums(), [])

	def test_animators_preserve_distinct_frame_and_interval_policies(self) -> None:
		potential = Potential.random(A=0.03, M=2, nx=8, ny=8, seed=9)
		times = np.array([0.0, 0.1, 0.2])
		states = np.array([[1.0, 1.1, 1.2], [1.2, 1.3, 1.4]])
		solution = Solution(t=times, states=states, source=GCInitialConfiguration(states[:, 0]), diagnostics={})
		animation = animate_gc_particle_solution(potential, solution, frames=np.int64(2), interval=200.5)
		animation._draw_was_started = True
		self.assertEqual(animation._save_count, 2)
		self.assertEqual(animation.event_source.interval, 200)
		with self.assertRaisesRegex(ValueError, "integer from 2 to the sample count"):
			animate_gc_particle_solution(potential, solution, frames=4)
		with self.assertRaisesRegex(ValueError, "integer of at least 2"):
			animate_potential(potential, frames=np.int64(2))
		with self.assertRaisesRegex(ValueError, "interval"):
			animate_gc_particle_solution(potential, solution, interval=np.bool_(True))

	def test_area_animation_caps_frames_but_keeps_strict_interval_validation(self) -> None:
		potential = Potential.random(A=0.03, M=2, nx=8, ny=8, seed=9)
		area = Area.square(center=(1.0, 1.0), side=0.5, points_per_side=1)
		initial = area.initial_state
		assert initial is not None
		times = np.array([0.0, 0.1, 0.2])
		states = np.repeat(initial[:, np.newaxis], 3, axis=1)
		solution = Solution(t=times, states=states, source=area, diagnostics={})
		animation = animate_gc_area_solution(potential, area, solution, frames=np.int64(20))
		animation._draw_was_started = True
		self.assertEqual(animation._save_count, 3)
		with self.assertRaisesRegex(ValueError, "interval"):
			animate_gc_area_solution(potential, area, solution, interval=200.5)

	def test_incompatible_dataset_does_not_replace_an_existing_export(self) -> None:
		panel = dict(title="sample", coordinates=np.zeros((2, 1, 2)), particle_ids=[7], colors=["#000000"])
		other = dict(panel, particle_ids=[8])
		with TemporaryDirectory() as temporary:
			path = Path(temporary) / "view.html"
			path.write_text("previous export", encoding="utf-8")
			with self.assertRaisesRegex(ValueError, "share cycle counts, panel shapes, IDs and colors"):
				export_poincare_panel_comparison(
					path, [panel], datasets=[dict(key="other", label="Other", panels=[other])],
					selected_dataset="other",
				)
			self.assertEqual(path.read_text(encoding="utf-8"), "previous export")


if __name__ == "__main__":
	unittest.main()
