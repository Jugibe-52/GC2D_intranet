"""Accuracy-refinement and Newton-work plots for four implicit methods."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


IMPLICIT_ACCURACY_COLORS = {
	"ABBA2Implicit[reduced_multiplier]": "#1f77b4",
	"ABBA4Implicit": "#ff7f0e",
	"ABBA4ImplicitSingleProjection": "#2ca02c",
	"BM4Implicit1": "#d62728",
}
_LINE_STYLES = {
	"ABBA2Implicit[reduced_multiplier]": "-",
	"ABBA4Implicit": "-",
	"ABBA4ImplicitSingleProjection": "--",
	"BM4Implicit1": "-",
}
_MARKERS = {
	"ABBA2Implicit[reduced_multiplier]": "o",
	"ABBA4Implicit": "s",
	"ABBA4ImplicitSingleProjection": "+",
	"BM4Implicit1": "o",
}


class ImplicitMethodAccuracySummaryView(Protocol):
	"""Step-summary fields consumed by implicit accuracy plots."""

	method_name: str
	method_label: str
	designed_order: float
	integration_step: float
	time_integrated_rms_distance: float
	mean_iterations_per_solve: float
	maximum_iterations_per_solve: int
	mean_residual_evaluations_per_solve: float
	maximum_residual_to_tolerance: float
	runtime_seconds: float


class ImplicitMethodAccuracyOrderView(Protocol):
	"""Adjacent-refinement fields consumed by the observed-order plot."""

	method_name: str
	method_label: str
	designed_order: float
	fine_step: float
	time_integrated_rms_order: float
	final_rms_order: float


def _group_summaries(
	summaries: Sequence[ImplicitMethodAccuracySummaryView],
) -> tuple[tuple[str, ...], tuple[float, ...], dict[str, dict[float, ImplicitMethodAccuracySummaryView]]]:
	"""Group a complete method-by-step table while preserving display order."""
	rows = tuple(summaries)
	if not rows:
		raise ValueError("At least one implicit accuracy summary is required.")
	methods = tuple(dict.fromkeys(row.method_name for row in rows))
	steps = tuple(dict.fromkeys(float(row.integration_step) for row in rows))
	grouped: dict[str, dict[float, ImplicitMethodAccuracySummaryView]] = {
		method: {} for method in methods
	}
	for row in rows:
		step = float(row.integration_step)
		if step in grouped[row.method_name]:
			raise ValueError("Implicit accuracy summaries contain a duplicate run.")
		grouped[row.method_name][step] = row
	if any(tuple(values) != steps for values in grouped.values()):
		raise ValueError("Every method must contain the same ordered refinement grid.")
	return methods, steps, grouped


def plot_implicit_method_accuracy_refinement(
	summaries: Sequence[ImplicitMethodAccuracySummaryView],
	*,
	reference_floor: float,
) -> tuple[Figure, Axes]:
	"""Plot time-integrated reference error and second/fourth-order guides."""
	methods, steps, grouped = _group_summaries(summaries)
	step_values = np.asarray(steps, dtype=float)
	positions = np.arange(len(steps), dtype=float)
	floor = float(reference_floor)
	if not np.isfinite(floor) or floor < 0.0:
		raise ValueError("`reference_floor` must be finite and non-negative.")
	figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
	method_errors: dict[str, np.ndarray] = {}
	for method in methods:
		rows = grouped[method]
		errors = np.asarray(
			[rows[step].time_integrated_rms_distance for step in steps],
			dtype=float,
		)
		if not np.all(np.isfinite(errors)) or np.any(errors <= 0.0):
			raise ValueError("Accuracy errors must be positive and finite.")
		method_errors[method] = errors
		axis.semilogy(
			positions,
			errors,
			marker=_MARKERS.get(method, "o"),
			linestyle=_LINE_STYLES.get(method, "-"),
			linewidth=1.8,
			color=IMPLICIT_ACCURACY_COLORS.get(method),
			label=rows[steps[0]].method_label,
		)
	guide_definitions = (
		(
			"ABBA2Implicit[reduced_multiplier]",
			2.0,
			"0.25",
			"Implicit ABBA2",
		),
		("ABBA4Implicit", 4.0, "0.55", "Implicit ABBA4"),
	)
	guide_positions = positions[-3:]
	guide_steps = step_values[-3:]
	anchor_index = -2
	for method, order, color, method_label in guide_definitions:
		if method not in method_errors:
			continue
		anchor = 1.8 * method_errors[method][anchor_index]
		anchor_step = step_values[anchor_index]
		axis.semilogy(
			guide_positions,
			anchor * (guide_steps / anchor_step) ** order,
			linestyle="--",
			color=color,
			linewidth=1.6,
			label=rf"$O(h^{{{order:g}}})$ guide — {method_label}",
		)
	if floor > 0.0:
		axis.axhline(
			floor,
			color="black",
			linestyle=":",
			label="DOP853/Radau reference discrepancy",
		)
	axis.set_xticks(positions, labels=[f"{step:g}" for step in step_values])
	axis.set_xlim(-0.25, float(len(steps)) - 0.75)
	axis.set(
		title="Implicit methods: trajectory-accuracy refinement",
		xlabel="Complete integration step $h$",
		ylabel="Time-integrated RMS Euclidean distance",
	)
	axis.grid(which="both", alpha=0.25)
	axis.legend(fontsize=8)
	return figure, axis


def plot_implicit_method_observed_orders(
	orders: Sequence[ImplicitMethodAccuracyOrderView],
) -> tuple[Figure, np.ndarray]:
	"""Plot time-integrated and final RMS orders against designed orders."""
	rows = tuple(orders)
	if not rows:
		raise ValueError("At least one implicit accuracy order is required.")
	methods = tuple(dict.fromkeys(row.method_name for row in rows))
	step_values = tuple(
		sorted({float(row.fine_step) for row in rows}, reverse=True)
	)
	positions = np.arange(len(step_values), dtype=float)
	position_by_step = {
		step: position for step, position in zip(step_values, positions, strict=True)
	}
	jitter_by_method = {
		method: jitter
		for method, jitter in zip(
			methods,
			np.linspace(-0.09, 0.09, len(methods)),
			strict=True,
		)
	}
	figure, axes = plt.subplots(
		2,
		1,
		figsize=(10, 8),
		sharex=True,
		constrained_layout=True,
	)
	for method in methods:
		method_rows = sorted(
			(row for row in rows if row.method_name == method),
			key=lambda row: row.fine_step,
			reverse=True,
		)
		method_positions = np.asarray(
			[
				position_by_step[float(row.fine_step)] + jitter_by_method[method]
				for row in method_rows
			],
			dtype=float,
		)
		color = IMPLICIT_ACCURACY_COLORS.get(method)
		for axis, field, title in (
			(axes[0], "time_integrated_rms_order", "Time-integrated RMS"),
			(axes[1], "final_rms_order", "Final RMS"),
		):
			values = np.asarray([getattr(row, field) for row in method_rows], dtype=float)
			mask = np.isfinite(values)
			axis.plot(
				method_positions[mask],
				values[mask],
				marker=_MARKERS.get(method, "o"),
				linestyle=_LINE_STYLES.get(method, "-"),
				color=color,
				label=f"{method_rows[0].method_label} — {title}",
			)
	for axis in axes:
		axis.axhline(2.0, color="0.25", linestyle=":", label="designed order 2")
		axis.axhline(4.0, color="0.55", linestyle=":", label="designed order 4")
		axis.set_xlim(-0.3, float(len(step_values)) - 0.7)
		axis.grid(alpha=0.25)
		axis.legend(fontsize=7, ncol=2)
	axes[0].set(title="Observed time-integrated convergence order", ylabel="Order $p$")
	axes[1].set(
		title="Observed final-time convergence order",
		xlabel="Fine complete step $h$",
		ylabel="Order $p$",
	)
	axes[1].set_xticks(
		positions,
		labels=[f"{step:g}" for step in step_values],
	)
	return figure, axes


def plot_implicit_method_newton_refinement(
	summaries: Sequence[ImplicitMethodAccuracySummaryView],
) -> tuple[Figure, np.ndarray]:
	"""Compare Newton corrections, residual work, and accepted residuals."""
	methods, steps, grouped = _group_summaries(summaries)
	positions = np.arange(len(steps), dtype=float)
	jitter_by_method = {
		method: jitter
		for method, jitter in zip(
			methods,
			np.linspace(-0.12, 0.12, len(methods)),
			strict=True,
		)
	}
	figure, axes = plt.subplots(
		2,
		2,
		figsize=(13, 9),
		sharex=True,
		constrained_layout=True,
	)
	fields = (
		("mean_iterations_per_solve", "Mean Newton corrections / solve"),
		("maximum_iterations_per_solve", "Maximum Newton corrections / solve"),
		("mean_residual_evaluations_per_solve", "Mean residual evaluations / solve"),
		("maximum_residual_to_tolerance", "Maximum final residual / tolerance"),
	)
	for method in methods:
		method_rows = grouped[method]
		method_positions = positions + jitter_by_method[method]
		for axis, (field, _) in zip(axes.flat, fields, strict=True):
			values = np.asarray(
				[getattr(method_rows[step], field) for step in steps],
				dtype=float,
			)
			if not np.all(np.isfinite(values)) or np.any(values < 0.0):
				raise ValueError("Newton summaries must be finite and non-negative.")
			axis.plot(
				method_positions,
				np.maximum(values, np.finfo(float).tiny),
				marker=_MARKERS.get(method, "o"),
				linestyle=_LINE_STYLES.get(method, "-"),
				color=IMPLICIT_ACCURACY_COLORS.get(method),
				label=method_rows[steps[0]].method_label,
			)
	for axis, (_, title) in zip(axes.flat, fields, strict=True):
		axis.set_title(title)
		axis.set_xlim(-0.3, float(len(steps)) - 0.7)
		axis.grid(alpha=0.25)
		axis.legend(fontsize=7)
	for axis in axes[1]:
		axis.set_xticks(positions, labels=[f"{step:g}" for step in steps])
		axis.set_xlabel("Complete integration step $h$")
	axes[1, 1].set_yscale("log")
	axes[1, 1].axhline(1.0, color="black", linestyle=":", label="acceptance limit")
	axes[1, 1].legend(fontsize=7)
	return figure, axes


def plot_implicit_method_accuracy_cost(
	summaries: Sequence[ImplicitMethodAccuracySummaryView],
) -> tuple[Figure, Axes]:
	"""Plot the runtime/error trajectory traced by every step refinement."""
	methods, steps, grouped = _group_summaries(summaries)
	figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
	annotation_offsets = {
		"ABBA2Implicit[reduced_multiplier]": (5, 5),
		"ABBA4Implicit": (5, 6),
		"ABBA4ImplicitSingleProjection": (5, -13),
		"BM4Implicit1": (5, 5),
	}
	for method in methods:
		rows = grouped[method]
		runtimes = np.asarray([rows[step].runtime_seconds for step in steps], dtype=float)
		errors = np.asarray(
			[rows[step].time_integrated_rms_distance for step in steps], dtype=float
		)
		if (
			not np.all(np.isfinite(runtimes))
			or np.any(runtimes <= 0.0)
			or not np.all(np.isfinite(errors))
			or np.any(errors <= 0.0)
		):
			raise ValueError("Accuracy-cost values must be positive and finite.")
		axis.loglog(
			runtimes,
			errors,
			marker=_MARKERS.get(method, "o"),
			linestyle=_LINE_STYLES.get(method, "-"),
			color=IMPLICIT_ACCURACY_COLORS.get(method),
			label=rows[steps[0]].method_label,
		)
		for runtime, error, step in zip(runtimes, errors, steps):
			axis.annotate(
				f"{step:g}",
				(runtime, error),
				xytext=annotation_offsets.get(method, (5, 5)),
				textcoords="offset points",
				fontsize=7,
				color=IMPLICIT_ACCURACY_COLORS.get(method),
			)
	axis.set(
		title="Implicit methods: measured accuracy-cost trade-off",
		xlabel="Runtime [s]",
		ylabel="Time-integrated RMS Euclidean distance",
	)
	axis.grid(which="both", alpha=0.25)
	axis.legend(fontsize=8)
	return figure, axis


__all__ = [
	"IMPLICIT_ACCURACY_COLORS",
	"ImplicitMethodAccuracyOrderView",
	"ImplicitMethodAccuracySummaryView",
	"plot_implicit_method_accuracy_cost",
	"plot_implicit_method_accuracy_refinement",
	"plot_implicit_method_newton_refinement",
	"plot_implicit_method_observed_orders",
]
