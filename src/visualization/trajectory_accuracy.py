"""Plots for a certified reference trajectory and ten-method accuracy study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from diagnostics import StoredReferenceTrajectory

from .ten_method_comparison import TEN_METHOD_COLORS, TEN_METHOD_SHORT_LABELS


class AccuracySeriesView(Protocol):
	"""Time-dependent fields consumed by the accuracy comparison plot."""

	@property
	def rms_distance(self) -> np.ndarray:
		"""Particle-RMS error at each saved time."""
		...

	@property
	def maximum_distance(self) -> np.ndarray:
		"""Maximum particle error at each saved time."""
		...


class AccuracySummaryView(Protocol):
	"""Scalar fields consumed by accuracy and cost plots."""

	@property
	def method_name(self) -> str:
		"""Stable method label."""
		...

	@property
	def global_rms_distance(self) -> float:
		"""RMS distance across every particle and saved time."""
		...

	@property
	def final_rms_distance(self) -> float:
		"""Particle-RMS distance at the final time."""
		...

	@property
	def runtime_seconds(self) -> float:
		"""Measured integration runtime."""
		...


class StepAccuracySummaryView(Protocol):
	"""Step-dependent fields consumed by the refinement plot."""

	@property
	def integration_step(self) -> float:
		"""Complete integration step size."""
		...

	@property
	def method_name(self) -> str:
		"""Stable method label."""
		...

	@property
	def time_integrated_rms_distance(self) -> float:
		"""Comparable space-time RMS error on the common saved grid."""
		...


def _positive(values: np.ndarray, *, floor: float) -> np.ndarray:
	"""Validate distances and floor exact zeros only for logarithmic axes."""
	array = np.asarray(values, dtype=float)
	if not np.all(np.isfinite(array)) or np.any(array < 0.0):
		raise ValueError("Accuracy distances must be finite and non-negative.")
	return np.where(array == 0.0, floor, array)


def plot_reference_trajectory_points(
	reference: StoredReferenceTrajectory,
) -> tuple[Figure, Axes]:
	"""Plot every saved DOP853 reference path using points without connecting lines."""
	if not isinstance(reference, StoredReferenceTrajectory):
		raise TypeError("`reference` must be a StoredReferenceTrajectory.")
	if reference.states.shape[0] % 2:
		raise ValueError("A planar reference requires x and y component blocks.")
	particle_count = reference.states.shape[0] // 2
	x = reference.states[:particle_count]
	y = reference.states[particle_count:]
	figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
	for particle in range(particle_count):
		line = axis.plot(
			x[particle],
			y[particle],
			linestyle="None",
			marker=".",
			markersize=5,
			label=f"Trajectory {particle + 1}",
		)[0]
		axis.scatter(
			x[particle, 0],
			y[particle, 0],
			marker="o",
			s=28,
			facecolor="none",
			edgecolor=line.get_color(),
			zorder=3,
		)
		axis.scatter(
			x[particle, -1],
			y[particle, -1],
			marker="x",
			s=32,
			color=line.get_color(),
			zorder=3,
		)
	axis.set(
		title="High-precision DOP853 reference trajectories",
		xlabel="$x$",
		ylabel="$y$",
	)
	axis.set_aspect("equal", adjustable="datalim")
	axis.grid(alpha=0.25)
	axis.legend(fontsize=8, ncol=2)
	return figure, axis


def plot_trajectory_accuracy_over_time(
	times: np.ndarray,
	series: Mapping[str, AccuracySeriesView],
	*,
	reference_floor: float,
) -> tuple[Figure, np.ndarray]:
	"""Plot particle-RMS and maximum reference error for labeled trajectories."""
	time_values = np.asarray(times, dtype=float)
	if (
		time_values.ndim != 1
		or time_values.size < 2
		or not np.all(np.isfinite(time_values))
		or np.any(np.diff(time_values) <= 0.0)
	):
		raise ValueError("Accuracy times must be finite and strictly increasing.")
	if not series:
		raise ValueError("At least one accuracy series is required.")
	if not np.isfinite(reference_floor) or reference_floor < 0.0:
		raise ValueError("`reference_floor` must be finite and non-negative.")
	plot_floor = max(reference_floor / 10.0, float(np.finfo(float).tiny))
	figure, axes = plt.subplots(
		2,
		1,
		figsize=(12, 9),
		sharex=True,
		constrained_layout=True,
	)
	for index, (label, values) in enumerate(series.items()):
		color = TEN_METHOD_COLORS.get(label, f"C{index}")
		linestyle = "--" if "Broyden" in label else "-"
		for axis, distance in zip(
			axes,
			(values.rms_distance, values.maximum_distance),
			strict=True,
		):
			array = np.asarray(distance, dtype=float)
			if array.shape != time_values.shape:
				raise ValueError("Every accuracy series must match the time grid.")
			axis.semilogy(
				time_values,
				_positive(array, floor=plot_floor),
				color=color,
				linestyle=linestyle,
				marker=".",
				markersize=3,
				label=label,
			)
	for axis in axes:
		if reference_floor > 0.0:
			axis.axhline(
				reference_floor,
				color="black",
				linestyle=":",
				linewidth=1.2,
				label="DOP853/Radau reference discrepancy",
			)
		axis.grid(which="both", alpha=0.25)
		axis.legend(fontsize=7, ncol=2)
	axes[0].set(
		title="Particle-RMS distance to the high-precision reference",
		ylabel="RMS distance",
	)
	axes[1].set(
		title="Maximum particle distance to the high-precision reference",
		xlabel="Time",
		ylabel="Maximum distance",
	)
	return figure, axes


def plot_ten_method_accuracy_over_time(
	times: np.ndarray,
	series: Mapping[str, AccuracySeriesView],
	*,
	reference_floor: float,
) -> tuple[Figure, np.ndarray]:
	"""Plot the ten-method errors through the generic trajectory helper."""
	return plot_trajectory_accuracy_over_time(
		times,
		series,
		reference_floor=reference_floor,
	)


def plot_single_method_accuracy_refinement(
	summaries: Sequence[StepAccuracySummaryView],
	*,
	expected_order: float,
	reference_floor: float = 0.0,
) -> tuple[Figure, Axes]:
	"""Plot one method's step refinement with an anchored order guide."""
	rows = tuple(summaries)
	if len(rows) < 2:
		raise ValueError("At least two refinement summaries are required.")
	method_names = {row.method_name for row in rows}
	if len(method_names) != 1:
		raise ValueError("Single-method refinement requires one method label.")
	order = float(expected_order)
	if not np.isfinite(order) or order <= 0.0:
		raise ValueError("`expected_order` must be positive and finite.")
	floor = float(reference_floor)
	if not np.isfinite(floor) or floor < 0.0:
		raise ValueError("`reference_floor` must be finite and non-negative.")
	steps = np.asarray([row.integration_step for row in rows], dtype=float)
	errors = np.asarray(
		[row.time_integrated_rms_distance for row in rows],
		dtype=float,
	)
	if (
		not np.all(np.isfinite(steps))
		or np.any(steps <= 0.0)
		or np.any(np.diff(steps) >= 0.0)
		or not np.all(np.isfinite(errors))
		or np.any(errors <= 0.0)
	):
		raise ValueError("Refinement steps and errors must be positive and ordered.")
	reference = errors[0] * (steps / steps[0]) ** order
	figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
	axis.loglog(
		steps,
		errors,
		marker="o",
		linewidth=1.6,
		label=next(iter(method_names)),
	)
	axis.loglog(
		steps,
		reference,
		linestyle="--",
		color="black",
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
	axis.set_xticks(steps, labels=[f"{step:g}" for step in steps])
	axis.set(
		title="Single-method trajectory-accuracy refinement",
		xlabel="Complete integration step $h$",
		ylabel="Time-integrated RMS distance",
	)
	axis.grid(which="both", alpha=0.25)
	axis.legend()
	return figure, axis


def plot_accuracy_summary(
	summaries: Sequence[AccuracySummaryView],
) -> tuple[Figure, Axes]:
	"""Compare global and final RMS reference errors for labeled methods."""
	rows = tuple(summaries)
	if not rows:
		raise ValueError("At least one accuracy summary is required.")
	labels = [row.method_name for row in rows]
	global_errors = np.asarray([row.global_rms_distance for row in rows], dtype=float)
	final_errors = np.asarray([row.final_rms_distance for row in rows], dtype=float)
	if (
		not np.all(np.isfinite(global_errors))
		or not np.all(np.isfinite(final_errors))
		or np.any(global_errors <= 0.0)
		or np.any(final_errors <= 0.0)
	):
		raise ValueError("Summary RMS errors must be positive and finite.")
	positions = np.arange(len(rows))
	figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
	axis.barh(
		positions - 0.18,
		global_errors,
		height=0.34,
		color=[TEN_METHOD_COLORS.get(label, f"C{index}") for index, label in enumerate(labels)],
		alpha=0.9,
		label="Global RMS",
	)
	axis.barh(
		positions + 0.18,
		final_errors,
		height=0.34,
		color=[TEN_METHOD_COLORS.get(label, f"C{index}") for index, label in enumerate(labels)],
		alpha=0.45,
		hatch="//",
		label="Final RMS",
	)
	axis.set_yticks(positions, labels=labels)
	axis.invert_yaxis()
	axis.set_xscale("log")
	axis.set(
		title="Trajectory accuracy against the numerical reference",
		xlabel="Periodic distance",
	)
	axis.grid(axis="x", which="both", alpha=0.25)
	axis.legend()
	return figure, axis


def plot_ten_method_accuracy_summary(
	summaries: Sequence[AccuracySummaryView],
) -> tuple[Figure, Axes]:
	"""Compare the ten established variants through the generic summary plot."""
	rows = tuple(summaries)
	if len(rows) != 10:
		raise ValueError("The accuracy summary plot requires exactly ten variants.")
	return plot_accuracy_summary(rows)


def plot_accuracy_runtime_tradeoff(
	summaries: Sequence[AccuracySummaryView],
) -> tuple[Figure, Axes]:
	"""Plot global RMS error against measured wall-clock runtime."""
	rows = tuple(summaries)
	if not rows:
		raise ValueError("At least one accuracy summary is required.")
	figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
	for index, row in enumerate(rows):
		if (
			not np.isfinite(row.runtime_seconds)
			or row.runtime_seconds <= 0.0
			or not np.isfinite(row.global_rms_distance)
			or row.global_rms_distance <= 0.0
		):
			raise ValueError("Runtime and global RMS error must be positive and finite.")
		color = TEN_METHOD_COLORS.get(row.method_name, f"C{index}")
		axis.scatter(
			row.runtime_seconds,
			row.global_rms_distance,
			s=55,
			color=color,
			zorder=3,
		)
		axis.annotate(
			TEN_METHOD_SHORT_LABELS.get(row.method_name, row.method_name),
			(row.runtime_seconds, row.global_rms_distance),
			xytext=(4, 4),
			textcoords="offset points",
			fontsize=7,
		)
	axis.set_xscale("log")
	axis.set_yscale("log")
	axis.set(
		title="Accuracy--runtime trade-off",
		xlabel="Runtime [s]",
		ylabel="Global RMS periodic distance",
	)
	axis.grid(which="both", alpha=0.25)
	return figure, axis


def plot_ten_method_accuracy_refinement(
	summaries: Sequence[StepAccuracySummaryView],
) -> tuple[Figure, np.ndarray]:
	"""Plot time-integrated RMS error against step for both method families."""
	rows = tuple(summaries)
	if not rows:
		raise ValueError("At least one step-accuracy summary is required.")
	labels = tuple(TEN_METHOD_COLORS)
	grouped: dict[str, dict[float, float]] = {label: {} for label in labels}
	for row in rows:
		step = float(row.integration_step)
		error = float(row.time_integrated_rms_distance)
		if (
			row.method_name not in grouped
			or not np.isfinite(step)
			or step <= 0.0
			or not np.isfinite(error)
			or error <= 0.0
		):
			raise ValueError("Refinement summaries contain an invalid method, step, or error.")
		if step in grouped[row.method_name]:
			raise ValueError("Each method may contain only one result per step.")
		grouped[row.method_name][step] = error
	step_values = tuple(
		sorted(
			{step for values in grouped.values() for step in values},
			reverse=True,
		)
	)
	if len(step_values) < 2 or any(
		set(values) != set(step_values) for values in grouped.values()
	):
		raise ValueError("All ten methods must share the same coarse-to-fine steps.")

	figure, axes = plt.subplots(
		1,
		2,
		figsize=(14, 6),
		sharex=True,
		constrained_layout=True,
	)
	for axis, family_name in zip(axes, ("ABBA", "BM4"), strict=True):
		for label in labels:
			if family_name not in label:
				continue
			linestyle = "--" if "Broyden" in label else "-"
			if "Midpoint" in label:
				linestyle = ":"
			axis.loglog(
				step_values,
				[grouped[label][step] for step in step_values],
				color=TEN_METHOD_COLORS[label],
				linestyle=linestyle,
				marker="o",
				markersize=5,
				label=label,
			)
		axis.invert_xaxis()
		axis.set_xticks(step_values, labels=[f"{step:g}" for step in step_values])
		axis.set(
			title=f"{family_name} step refinement",
			xlabel="Complete integration step $h$",
			ylabel="Time-integrated RMS periodic distance",
		)
		axis.grid(which="both", alpha=0.25)
		axis.legend(fontsize=7)
	return figure, axes


__all__ = [
	"AccuracySeriesView",
	"AccuracySummaryView",
	"StepAccuracySummaryView",
	"plot_accuracy_summary",
	"plot_accuracy_runtime_tradeoff",
	"plot_reference_trajectory_points",
	"plot_single_method_accuracy_refinement",
	"plot_ten_method_accuracy_over_time",
	"plot_ten_method_accuracy_refinement",
	"plot_ten_method_accuracy_summary",
	"plot_trajectory_accuracy_over_time",
]
