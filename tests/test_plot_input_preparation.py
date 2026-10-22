"""Reject later invalid plot series without allocating or disturbing figures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
import unittest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from diagnostics.implicit_iterations import ImplicitIterationRecord
from formulations.layout import GCStateLayout
from states.history import States
from visualization.energy import (
	plot_generalized_energy_errors, plot_time_extended_symplecticity,
	plot_reduced_time_extended_symplecticity,
)
from visualization.hbvm42 import plot_hbvm42_energy_errors
from visualization.implicit_iterations import plot_implicit_iteration_comparison
from visualization.implicit_method_accuracy import (
	plot_implicit_method_accuracy_refinement, plot_implicit_method_newton_refinement,
	plot_implicit_method_accuracy_cost,
)
from visualization.ten_method_comparison import plot_ten_method_nonlinear_work
from visualization.trajectory_accuracy import plot_accuracy_runtime_tradeoff


def _solution(**diagnostics: np.ndarray) -> States:
	return States(np.array([0., 1., 2.]), np.zeros((2, 3)), diagnostics,
		layout=GCStateLayout())


def _summary(method: str, step: float, **changes: float) -> SimpleNamespace:
	values = dict(method_name=method, method_label=method, designed_order=4.,
		integration_step=step, time_integrated_rms_distance=step**4,
		mean_iterations_per_solve=2., maximum_iterations_per_solve=3,
		mean_residual_evaluations_per_solve=4., maximum_residual_to_tolerance=.5,
		runtime_seconds=1. / step, global_rms_distance=step**4)
	return SimpleNamespace(**(values | changes))


class PlotInputPreparationTests(unittest.TestCase):
	"""Check resource boundaries and actual curves through public plot functions."""

	def setUp(self) -> None:
		self.existing, self.existing_axis = plt.subplots()
		self.existing_axis.plot([0., 1.], [3., 4.])

	def tearDown(self) -> None:
		plt.close("all")

	def assert_rejected(self, plot: Callable[..., Any], values: Any, message: str, **kwargs: Any) -> None:
		before = plt.get_fignums()
		with self.assertRaisesRegex(ValueError, message):
			plot(values, **kwargs)
		self.assertEqual(plt.get_fignums(), before)
		np.testing.assert_array_equal(self.existing_axis.lines[0].get_ydata(), [3., 4.])

	def assert_curve(self, figure: Figure, axis: Axes, x: ArrayLike, y: ArrayLike) -> None:
		np.testing.assert_allclose(axis.lines[0].get_xdata(), x)
		np.testing.assert_allclose(axis.lines[0].get_ydata(), y)
		figure.canvas.draw()
		plt.close(figure)

	def test_generalized_energy_rejects_later_zero_envelope(self) -> None:
		def run(error: ArrayLike) -> SimpleNamespace:
			return SimpleNamespace(actual_step=.1, times=np.arange(3.),
				hamiltonian=np.ones(3), kappa=np.zeros(3), generalized_energy=np.ones(3),
				relative_errors=np.array(error), running_max_relative_errors=np.abs(error))
		good = run([0., .01, .02])
		self.assert_rejected(plot_generalized_energy_errors, [good, run([0., 0., 0.])],
			"non-zero error sample", method_name="sample")
		figure, axes = plot_generalized_energy_errors([good, good], method_name="sample")
		np.testing.assert_allclose(axes[1].lines[0].get_ydata(), [.005, .01, .02])
		self.assert_curve(figure, axes[0], np.arange(3.), [0., .01, .02])

	def test_both_extended_plots_reject_later_invalid_defects(self) -> None:
		for plot, prefix in ((plot_time_extended_symplecticity, "extended"),
			(plot_reduced_time_extended_symplecticity, "reduced_extended")):
			with self.subTest(plot=plot.__name__):
				attributes = {"actual_step": .1, f"{prefix}_symplecticity_times": np.array([1., 2.]),
					f"{prefix}_relative_defects": np.array([1e-8, 2e-8]),
					f"{prefix}_determinant_errors": np.array([0., 1e-9])}
				good = SimpleNamespace(**attributes)
				for invalid in (np.array([-1., 1.]), np.array([np.nan, 1.]), np.ones(3)):
					bad = SimpleNamespace(**(attributes | {f"{prefix}_relative_defects": invalid}))
					self.assert_rejected(plot, [good, bad], "finite and aligned", method_name="sample")
				figure, axes = plot([good, good], method_name="sample")
				self.assert_curve(figure, axes[0], [1., 2.], [1e-8, 2e-8])

	def test_hbvm_rejects_later_missing_or_misaligned_energy(self) -> None:
		good = _solution(energy_drift=np.array([[0., -1e-8, 2e-8]]))
		for bad, message in ((_solution(), "energy diagnostics"),
			(_solution(energy_drift=np.ones((2, 3))), "one planar trajectory")):
			with self.subTest(message=message):
				self.assert_rejected(plot_hbvm42_energy_errors, {.1: good, .05: bad}, message)
		figure, axis = plot_hbvm42_energy_errors({.1: good, .05: good})
		self.assert_curve(figure, axis, [0., 1., 2.], [0., -1e-8, 2e-8])

	def test_iteration_comparison_rejects_later_label_and_record_sequence(self) -> None:
		record = ImplicitIterationRecord(observation_index=0, step_index=1,
			start_time=0., end_time=1., duration=1., method_name="sample",
			formulation_name="physical", nonlinear_solver="newton", newton_iterations=2,
			residual_evaluations=3, newton_residual_norm=1e-9, newton_tolerance=1e-8,
			residual_to_tolerance_ratio=.1, projection_multiplier_norm=1e-5)
		good = [record, replace(record, step_index=2, end_time=2.)]
		self.assert_rejected(plot_implicit_iteration_comparison, {"first": good, "": good},
			"non-empty strings")
		self.assert_rejected(plot_implicit_iteration_comparison, {"first": good, "second": [record, record]},
			"ordered by increasing step")
		figure, axes = plot_implicit_iteration_comparison({"first": good, "second": good})
		self.assert_curve(figure, axes[0], [1., 2.], [2, 2])

	def test_implicit_refinement_plots_reject_later_invalid_method(self) -> None:
		rows = [_summary(method, step) for method in ("ABBA4Implicit", "BM4Implicit") for step in (.2, .1)]
		cases = (
			(plot_implicit_method_accuracy_refinement, "time_integrated_rms_distance", 0.,
				"Accuracy errors", {"reference_floor": 0.}),
			(plot_implicit_method_newton_refinement, "maximum_residual_to_tolerance", -1.,
				"Newton summaries", {}),
			(plot_implicit_method_accuracy_cost, "runtime_seconds", np.nan,
				"Accuracy-cost values", {}),
		)
		for plot, field, invalid, message, kwargs in cases:
			with self.subTest(plot=plot.__name__):
				bad = rows[:-1] + [SimpleNamespace(**(vars(rows[-1]) | {field: invalid}))]
				self.assert_rejected(plot, bad, message, **kwargs)
				figure, axes = plot(rows, **kwargs)
				if plot is plot_implicit_method_newton_refinement:
					self.assert_curve(figure, axes[0, 0], [-.12, .88], [2., 2.])
				else:
					x = [5., 10.] if plot is plot_implicit_method_accuracy_cost else [0., 1.]
					self.assert_curve(figure, axes, x, [.2**4, .1**4])

	def test_six_variant_work_rejects_later_misaligned_diagnostics(self) -> None:
		diagnostics = dict(nonlinear_iterations=np.array([1, 2]),
			residual_evaluations=np.array([2, 3]), nonlinear_residual_norms=np.array([.1, .2]),
			nonlinear_tolerances=np.ones(2))
		solutions = {f"method {index}": _solution(**diagnostics) for index in range(6)}
		for field in diagnostics:
			with self.subTest(field=field):
				bad = solutions | {"method 5": _solution(**(diagnostics | {field: np.ones(3)}))}
				self.assert_rejected(plot_ten_method_nonlinear_work, bad, "every saved step")
		figure, axes = plot_ten_method_nonlinear_work(solutions)
		self.assert_curve(figure, axes[0], [1., 2.], [1, 2])

	def test_runtime_tradeoff_rejects_later_invalid_row(self) -> None:
		good = _summary("first", .1)
		for changes in ({"runtime_seconds": 0.}, {"global_rms_distance": np.inf}):
			self.assert_rejected(plot_accuracy_runtime_tradeoff,
				[good, _summary("second", .2, **changes)], "positive and finite")
		figure, axis = plot_accuracy_runtime_tradeoff([good, _summary("second", .2)])
		np.testing.assert_allclose(axis.collections[0].get_offsets(), [[10., .1**4]])
		np.testing.assert_allclose(axis.collections[1].get_offsets(), [[5., .2**4]])
		figure.canvas.draw()
		plt.close(figure)


if __name__ == "__main__":
	unittest.main()
