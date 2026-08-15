"""Plots for explicit-Jacobian midpoint-BM4 symplecticity studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from simulation import Solution


class _SymplecticityRecord(Protocol):
	"""Scalar fields required by the averaged-defect plot."""

	@property
	def time(self) -> float:
		"""Observation time."""

	@property
	def mean_local_relative_defect(self) -> float:
		"""Arithmetic trajectory mean of local defects."""

	@property
	def std_local_relative_defect(self) -> float:
		"""Trajectory standard deviation of local defects."""

	@property
	def mean_accumulated_relative_defect(self) -> float:
		"""Arithmetic trajectory mean of accumulated defects."""

	@property
	def std_accumulated_relative_defect(self) -> float:
		"""Trajectory standard deviation of accumulated defects."""


def _positive(values: np.ndarray) -> np.ndarray:
	"""Replace exact zeros by a plotting-only positive floor."""
	result = np.asarray(values, dtype=float)
	if not np.all(np.isfinite(result)) or np.any(result < 0.0):
		raise ValueError("Symplecticity errors must be finite and non-negative.")
	positive = result[result > 0.0]
	floor = (
		float(np.min(positive)) / 10.0
		if positive.size
		else float(np.finfo(float).eps)
	)
	return np.where(result == 0.0, floor, result)


def plot_midpoint_bm4_symplecticity(
	records: Mapping[str, Sequence[_SymplecticityRecord]],
	*,
	labels: Sequence[str] | None = None,
) -> tuple[Figure, np.ndarray]:
	"""Plot mean local and accumulated defects for each integration step.

	The solid curves are arithmetic means over trajectories. Shaded regions show
	one standard deviation across trajectories and are descriptive dispersion,
	not uncertainty intervals.
	"""
	ordered_labels = tuple(records) if labels is None else tuple(labels)
	if not ordered_labels or any(label not in records for label in ordered_labels):
		raise ValueError("`labels` must select at least one available record series.")

	figure, axes = plt.subplots(
		2,
		1,
		figsize=(10, 8),
		sharex=True,
		constrained_layout=True,
	)
	fields = (
		(
			"mean_local_relative_defect",
			"std_local_relative_defect",
			"Mean local-step symplecticity error",
		),
		(
			"mean_accumulated_relative_defect",
			"std_accumulated_relative_defect",
			"Mean accumulated-flow symplecticity error",
		),
	)
	reference_times: np.ndarray | None = None
	for label in ordered_labels:
		rows = tuple(records[label])
		if not rows:
			raise ValueError(f"The record series for {label!r} is empty.")
		times = np.asarray([row.time for row in rows], dtype=float)
		if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
			raise ValueError("Diagnostic times must be finite and strictly increasing.")
		if reference_times is None:
			reference_times = times
		elif times.shape != reference_times.shape or not np.allclose(
			times,
			reference_times,
			rtol=0.0,
			atol=float(
				64.0
				* np.finfo(float).eps
				* max(1.0, float(np.max(np.abs(reference_times))))
			),
		):
			raise ValueError(
				"Compared symplecticity records must share one saved-time grid."
			)
		for axis, (mean_field, std_field, _title) in zip(
			axes,
			fields,
			strict=True,
		):
			means = np.asarray(
				[getattr(row, mean_field) for row in rows],
				dtype=float,
			)
			standard_deviations = np.asarray(
				[getattr(row, std_field) for row in rows],
				dtype=float,
			)
			_positive(standard_deviations)
			line = axis.semilogy(times, _positive(means), label=label)[0]
			lower = _positive(np.maximum(means - standard_deviations, 0.0))
			upper = _positive(means + standard_deviations)
			axis.fill_between(
				times,
				lower,
				upper,
				color=line.get_color(),
				alpha=0.15,
				linewidth=0.0,
			)

	for axis, (_, _, title) in zip(axes, fields, strict=True):
		axis.set(title=title, ylabel="Relative defect")
		axis.grid(which="both", alpha=0.25)
		axis.legend(title="Integration step")
	axes[-1].set_xlabel("Time")
	return figure, axes


def plot_midpoint_bm4_trajectories(
	solution: Solution,
	*,
	step_label: str | None = None,
) -> tuple[Figure, Axes]:
	"""Plot all planar trajectories from one midpoint-BM4 solution."""
	if not isinstance(solution, Solution):
		raise TypeError("`solution` must be a Solution instance.")
	x, y = solution.positions()
	if x.ndim != 2 or y.shape != x.shape:
		raise ValueError("The solution must contain sampled planar trajectories.")

	figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
	for particle, (x_values, y_values) in enumerate(zip(x, y, strict=True), start=1):
		line = axis.plot(
			x_values,
			y_values,
			linestyle="None",
			marker=".",
			markersize=5,
			label=f"Trajectory {particle}",
		)[0]
		axis.scatter(
			x_values[0],
			y_values[0],
			marker="o",
			s=28,
			color=line.get_color(),
			zorder=3,
		)
		axis.scatter(
			x_values[-1],
			y_values[-1],
			marker="x",
			s=34,
			color=line.get_color(),
			zorder=3,
		)
	title = "Midpoint-BM4 trajectories"
	if step_label is not None:
		title = f"{title} ({step_label})"
	axis.set(xlabel="$x$", ylabel="$y$", title=title)
	axis.set_aspect("equal", adjustable="datalim")
	axis.grid(alpha=0.25)
	axis.legend()
	return figure, axis


__all__ = [
	"plot_midpoint_bm4_symplecticity",
	"plot_midpoint_bm4_trajectories",
]
