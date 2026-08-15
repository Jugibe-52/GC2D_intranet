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


def plot_ten_method_accuracy_over_time(
	times: np.ndarray,
	series: Mapping[str, AccuracySeriesView],
	*,
	reference_floor: float,
) -> tuple[Figure, np.ndarray]:
	"""Plot particle-RMS and maximum periodic error against the reference."""
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
		ylabel="RMS periodic distance",
	)
	axes[1].set(
		title="Maximum particle distance to the high-precision reference",
		xlabel="Time",
		ylabel="Maximum periodic distance",
	)
	return figure, axes


def plot_ten_method_accuracy_summary(
	summaries: Sequence[AccuracySummaryView],
) -> tuple[Figure, Axes]:
	"""Compare global and final RMS reference errors on a logarithmic axis."""
	rows = tuple(summaries)
	if len(rows) != 10:
		raise ValueError("The accuracy summary plot requires exactly ten variants.")
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


def plot_accuracy_runtime_tradeoff(
	summaries: Sequence[AccuracySummaryView],
) -> tuple[Figure, Axes]:
	"""Plot global RMS error against measured wall-clock runtime."""
	rows = tuple(summaries)
	if len(rows) != 10:
		raise ValueError("The trade-off plot requires exactly ten variants.")
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


__all__ = [
	"AccuracySeriesView",
	"AccuracySummaryView",
	"plot_accuracy_runtime_tradeoff",
	"plot_reference_trajectory_points",
	"plot_ten_method_accuracy_over_time",
	"plot_ten_method_accuracy_summary",
]
