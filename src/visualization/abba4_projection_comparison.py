"""Plots comparing three-projection ABBA4 with single-projection SP-ABBA4."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


ABBA4_PROJECTION_COLORS: dict[str, str] = {
	"ABBA4Implicit1": "tab:blue",
	"ABBA4SingleProjectionImplicit1": "tab:orange",
}


class ABBA4ProjectionSummaryView(Protocol):
	"""Scalar fields consumed by the ABBA4 projection comparison plots."""

	method_name: str
	method_label: str
	integration_step: float
	time_integrated_rms_distance: float
	mean_iterations_per_step: float
	mean_iterations_per_solve: float
	mean_residual_evaluations_per_step: float
	mean_unprojected_abba_map_evaluations_per_step: float
	mean_newton_tangent_abba_map_evaluations_per_step: float
	maximum_residual_to_tolerance: float
	maximum_projection_multiplier_norm: float
	runtime_seconds: float
	runtime_first_quartile_seconds: float
	runtime_third_quartile_seconds: float


class ABBA4ProjectionOrderView(Protocol):
	"""Observed-order fields consumed by the order-reduction plot."""

	method_name: str
	method_label: str
	fine_step: float
	time_integrated_rms_order: float
	final_rms_order: float
	time_integrated_order_reduction: float
	final_order_reduction: float
	time_integrated_order_reduction_detected: bool
	final_order_reduction_detected: bool
	projection_multiplier_order: float


def _group_summaries(
	summaries: Sequence[ABBA4ProjectionSummaryView],
) -> tuple[tuple[str, ...], tuple[float, ...], dict[str, dict[float, ABBA4ProjectionSummaryView]]]:
	"""Validate and group one complete method-by-refinement summary grid."""
	rows = tuple(summaries)
	if not rows:
		raise ValueError("At least one ABBA4 projection summary is required.")
	method_names = tuple(dict.fromkeys(row.method_name for row in rows))
	steps = tuple(
		sorted(
			{float(row.integration_step) for row in rows},
			reverse=True,
		)
	)
	if len(method_names) != 2 or len(steps) < 2:
		raise ValueError("The comparison requires two methods and at least two steps.")
	grouped: dict[str, dict[float, ABBA4ProjectionSummaryView]] = {
		method_name: {} for method_name in method_names
	}
	for row in rows:
		step = float(row.integration_step)
		if not np.isfinite(step) or step <= 0.0:
			raise ValueError("Integration steps must be positive and finite.")
		if step in grouped[row.method_name]:
			raise ValueError("Each method may contain only one row per step.")
		grouped[row.method_name][step] = row
	if any(set(values) != set(steps) for values in grouped.values()):
		raise ValueError("Both ABBA4 methods must share one coarse-to-fine grid.")
	return method_names, steps, grouped


def _positive(value: float, *, name: str) -> float:
	"""Validate one positive scalar used on a logarithmic axis."""
	result = float(value)
	if not np.isfinite(result) or result <= 0.0:
		raise ValueError(f"`{name}` must be positive and finite.")
	return result


def plot_abba4_projection_accuracy(
	summaries: Sequence[ABBA4ProjectionSummaryView],
	*,
	designed_order: float = 4.0,
	reference_floor: float = 0.0,
) -> tuple[Figure, Axes]:
	"""Plot reference error for both projection strategies under refinement."""
	method_names, steps, grouped = _group_summaries(summaries)
	order = _positive(designed_order, name="designed_order")
	floor = float(reference_floor)
	if not np.isfinite(floor) or floor < 0.0:
		raise ValueError("`reference_floor` must be finite and non-negative.")
	step_values = np.asarray(steps, dtype=float)
	figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
	coarse_errors: list[float] = []
	for method_name in method_names:
		method_rows = grouped[method_name]
		errors = np.asarray(
			[method_rows[step].time_integrated_rms_distance for step in steps],
			dtype=float,
		)
		if not np.all(np.isfinite(errors)) or np.any(errors <= 0.0):
			raise ValueError("Accuracy errors must be positive and finite.")
		coarse_errors.append(float(errors[0]))
		axis.loglog(
			step_values,
			errors,
			marker="o",
			linewidth=1.8,
			color=ABBA4_PROJECTION_COLORS.get(method_name),
			label=method_rows[steps[0]].method_label,
		)
	guide = max(coarse_errors) * (step_values / step_values[0]) ** order
	axis.loglog(
		step_values,
		guide,
		color="black",
		linestyle="--",
		label=rf"$O(h^{{{order:g}}})$ guide",
	)
	if floor > 0.0:
		axis.axhline(
			floor,
			color="0.35",
			linestyle=":",
			label="DOP853/Radau reference discrepancy",
		)
	axis.invert_xaxis()
	axis.set_xticks(step_values, labels=[f"{step:g}" for step in step_values])
	axis.set(
		title="ABBA4 projection strategies: trajectory-accuracy refinement",
		xlabel="Complete integration step $h$",
		ylabel="Time-integrated RMS periodic distance",
	)
	axis.grid(which="both", alpha=0.25)
	axis.legend()
	return figure, axis


def plot_abba4_projection_order_reduction(
	orders: Sequence[ABBA4ProjectionOrderView],
	*,
	designed_order: float = 4.0,
	reduction_threshold: float = 0.5,
) -> tuple[Figure, np.ndarray]:
	"""Plot observed orders and the explicit deficit from the designed order."""
	rows = tuple(orders)
	if not rows:
		raise ValueError("At least one ABBA4 projection order row is required.")
	order = _positive(designed_order, name="designed_order")
	threshold = float(reduction_threshold)
	if not np.isfinite(threshold) or threshold < 0.0:
		raise ValueError("`reduction_threshold` must be finite and non-negative.")
	method_names = tuple(dict.fromkeys(row.method_name for row in rows))
	if len(method_names) != 2:
		raise ValueError("Order reduction requires exactly two projection methods.")
	figure, axes = plt.subplots(
		2,
		1,
		figsize=(10, 8),
		sharex=True,
		constrained_layout=True,
	)
	all_steps: set[float] = set()
	for method_name in method_names:
		method_rows = sorted(
			(row for row in rows if row.method_name == method_name),
			key=lambda row: row.fine_step,
			reverse=True,
		)
		steps = np.asarray([row.fine_step for row in method_rows], dtype=float)
		all_steps.update(float(step) for step in steps)
		color = ABBA4_PROJECTION_COLORS.get(method_name)
		label = method_rows[0].method_label
		for field_name, linestyle, suffix in (
			("time_integrated_rms_order", "-", "time-integrated RMS"),
			("final_rms_order", "--", "final RMS"),
		):
			values = np.asarray(
				[getattr(row, field_name) for row in method_rows], dtype=float
			)
			mask = np.isfinite(values)
			axes[0].plot(
				steps[mask],
				values[mask],
				marker="o",
				linestyle=linestyle,
				color=color,
				label=f"{label} — {suffix}",
			)
		for field_name, linestyle, flag_name, suffix in (
			(
				"time_integrated_order_reduction",
				"-",
				"time_integrated_order_reduction_detected",
				"time-integrated RMS",
			),
			(
				"final_order_reduction",
				"--",
				"final_order_reduction_detected",
				"final RMS",
			),
		):
			values = np.asarray(
				[getattr(row, field_name) for row in method_rows], dtype=float
			)
			mask = np.isfinite(values)
			axes[1].plot(
				steps[mask],
				values[mask],
				marker="o",
				linestyle=linestyle,
				color=color,
				label=f"{label} — {suffix}",
			)
			detected = np.asarray(
				[getattr(row, flag_name) for row in method_rows], dtype=bool
			)
			flagged = mask & detected
			axes[1].scatter(
				steps[flagged],
				values[flagged],
				marker="x",
				s=70,
				color="red",
				zorder=4,
			)
	axes[0].axhline(order, color="black", linestyle=":", label="designed order")
	axes[1].axhline(0.0, color="black", linestyle=":")
	axes[1].axhline(
		threshold,
		color="red",
		linestyle=":",
		label="reduction threshold",
	)
	axes[0].set(
		title="Observed convergence order",
		ylabel="Observed order $p$",
	)
	axes[1].set(
		title="Order reduction relative to the fourth-order design",
		xlabel="Fine complete step $h$",
		ylabel=rf"${order:g}-p$",
	)
	step_values = np.asarray(sorted(all_steps, reverse=True), dtype=float)
	axes[1].set_xscale("log")
	axes[1].invert_xaxis()
	axes[1].set_xticks(step_values, labels=[f"{step:g}" for step in step_values])
	for axis in axes:
		axis.grid(alpha=0.25)
		axis.legend(fontsize=8)
	return figure, axes


def plot_abba4_projection_newton_work(
	summaries: Sequence[ABBA4ProjectionSummaryView],
) -> tuple[Figure, np.ndarray]:
	"""Compare nonlinear corrections, residual work, and accepted convergence."""
	method_names, steps, grouped = _group_summaries(summaries)
	step_values = np.asarray(steps, dtype=float)
	figure, axes = plt.subplots(
		2,
		2,
		figsize=(13, 8),
		sharex=True,
		constrained_layout=True,
	)
	fields = (
		("mean_iterations_per_solve", "Mean corrections / nonlinear solve"),
		(
			"mean_newton_tangent_abba_map_evaluations_per_step",
			"Mean differentiated ABBA maps / complete step",
		),
		(
			"mean_unprojected_abba_map_evaluations_per_step",
			"Mean residual ABBA maps / complete step",
		),
		("maximum_residual_to_tolerance", "Maximum final residual / tolerance"),
	)
	for method_name in method_names:
		method_rows = grouped[method_name]
		for axis, (field_name, _) in zip(axes.flat, fields, strict=True):
			values = np.asarray(
				[getattr(method_rows[step], field_name) for step in steps],
				dtype=float,
			)
			if not np.all(np.isfinite(values)) or np.any(values < 0.0):
				raise ValueError("Nonlinear-work summaries must be finite and non-negative.")
			plot_values = (
				np.maximum(values, np.finfo(float).tiny)
				if field_name == "maximum_residual_to_tolerance"
				else values
			)
			axis.plot(
				step_values,
				plot_values,
				marker="o",
				color=ABBA4_PROJECTION_COLORS.get(method_name),
				label=method_rows[steps[0]].method_label,
			)
	for axis, (_, title) in zip(axes.flat, fields, strict=True):
		axis.set_title(title)
		axis.grid(alpha=0.25)
		axis.legend(fontsize=8)
	for axis in axes[1]:
		axis.set_xscale("log")
		axis.invert_xaxis()
		axis.set_xticks(step_values, labels=[f"{step:g}" for step in step_values])
		axis.set_xlabel("Complete integration step $h$")
	axes[1, 1].set_yscale("log")
	axes[1, 1].axhline(
		1.0,
		color="black",
		linestyle=":",
		label="acceptance limit",
	)
	axes[1, 1].legend(fontsize=8)
	return figure, axes


def plot_abba4_projection_multiplier_scaling(
	summaries: Sequence[ABBA4ProjectionSummaryView],
) -> tuple[Figure, Axes]:
	"""Plot maximum projection multipliers with expected cubic and quintic guides."""
	method_names, steps, grouped = _group_summaries(summaries)
	step_values = np.asarray(steps, dtype=float)
	figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
	for method_name in method_names:
		method_rows = grouped[method_name]
		values = np.asarray(
			[method_rows[step].maximum_projection_multiplier_norm for step in steps],
			dtype=float,
		)
		if not np.all(np.isfinite(values)) or np.any(values < 0.0):
			raise ValueError(
				"Projection multiplier norms must be finite and non-negative."
			)
		plot_values = np.maximum(values, np.finfo(float).tiny)
		color = ABBA4_PROJECTION_COLORS.get(method_name)
		axis.loglog(
			step_values,
			plot_values,
			marker="o",
			color=color,
			label=method_rows[steps[0]].method_label,
		)
		expected_order = 3.0 if method_name == "ABBA4Implicit1" else 5.0
		guide = plot_values[0] * (step_values / step_values[0]) ** expected_order
		axis.loglog(
			step_values,
			guide,
			linestyle=":",
			color=color,
			alpha=0.75,
			label=rf"$O(h^{{{expected_order:g}}})$ guide",
		)
	axis.invert_xaxis()
	axis.set_xticks(step_values, labels=[f"{step:g}" for step in step_values])
	axis.set(
		title="Projection-multiplier scaling",
		xlabel="Complete integration step $h$",
		ylabel=r"Maximum $\|\mu_n\|_\infty$",
	)
	axis.grid(which="both", alpha=0.25)
	axis.legend()
	return figure, axis


def plot_abba4_projection_runtime(
	summaries: Sequence[ABBA4ProjectionSummaryView],
) -> tuple[Figure, Axes]:
	"""Plot median runtime with the interquartile timing interval."""
	method_names, steps, grouped = _group_summaries(summaries)
	step_values = np.asarray(steps, dtype=float)
	figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
	for method_name in method_names:
		method_rows = grouped[method_name]
		median = np.asarray(
			[method_rows[step].runtime_seconds for step in steps], dtype=float
		)
		first = np.asarray(
			[
				method_rows[step].runtime_first_quartile_seconds
				for step in steps
			],
			dtype=float,
		)
		third = np.asarray(
			[
				method_rows[step].runtime_third_quartile_seconds
				for step in steps
			],
			dtype=float,
		)
		if (
			not np.all(np.isfinite(median))
			or not np.all(np.isfinite(first))
			or not np.all(np.isfinite(third))
			or np.any(first <= 0.0)
			or np.any(first > median)
			or np.any(median > third)
		):
			raise ValueError("Runtime quartiles must be positive and ordered.")
		color = ABBA4_PROJECTION_COLORS.get(method_name)
		axis.loglog(
			step_values,
			median,
			marker="o",
			color=color,
			label=method_rows[steps[0]].method_label,
		)
		axis.fill_between(step_values, first, third, color=color, alpha=0.18)
	axis.invert_xaxis()
	axis.set_xticks(step_values, labels=[f"{step:g}" for step in step_values])
	axis.set(
		title="ABBA4 projection-strategy runtime",
		xlabel="Complete integration step $h$",
		ylabel="Median integration runtime [s]",
	)
	axis.grid(which="both", alpha=0.25)
	axis.legend()
	return figure, axis


__all__ = [
	"ABBA4_PROJECTION_COLORS",
	"ABBA4ProjectionOrderView",
	"ABBA4ProjectionSummaryView",
	"plot_abba4_projection_accuracy",
	"plot_abba4_projection_multiplier_scaling",
	"plot_abba4_projection_newton_work",
	"plot_abba4_projection_order_reduction",
	"plot_abba4_projection_runtime",
]
