"""Plots for averaged independent-trajectory symplecticity studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from solution import Solution

if TYPE_CHECKING:
	from studies.trajectory_symplecticity import TrajectorySymplecticityResult


class TrajectorySymplecticityRecordView(Protocol):
	"""Scalar record fields consumed by the comparison plot."""

	time: float
	mean_local_relative_defect: float
	std_local_relative_defect: float
	mean_accumulated_relative_defect: float
	std_accumulated_relative_defect: float


def _positive(values: np.ndarray) -> np.ndarray:
	"""Validate non-negative data and floor only exact zeros for log axes."""
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


def _symplecticity_series(
	records: Mapping[str, Sequence[TrajectorySymplecticityRecordView]],
	ordered_labels: tuple[str, ...], fields: tuple[tuple[str, str, str], ...],
) -> list[tuple[str, np.ndarray, tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]]]:
	"""Align record times and prepare non-negative means and uncertainty bands."""
	prepared: list[tuple[str, np.ndarray, tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]]] = []
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
			raise ValueError("Compared records must share one saved-time grid.")
		metrics: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
		for mean_field, std_field, _title in fields:
			means = np.asarray([getattr(row, mean_field) for row in rows], dtype=float)
			std = np.asarray([getattr(row, std_field) for row in rows], dtype=float)
			_positive(std)
			metrics.append((
				_positive(means),
				_positive(np.maximum(means - std, 0.0)),
				_positive(means + std),
			))
		prepared.append((label, times, tuple(metrics)))
	return prepared


def plot_trajectory_symplecticity(
	records: Mapping[str, Sequence[TrajectorySymplecticityRecordView]],
	*,
	labels: Sequence[str] | None = None,
	method_name: str,
) -> tuple[Figure, np.ndarray]:
	"""Plot mean local and accumulated errors for one numerical method."""
	ordered_labels = tuple(records) if labels is None else tuple(labels)
	if not ordered_labels or any(label not in records for label in ordered_labels):
		raise ValueError("`labels` must select at least one available series.")
	fields = (
		(
			"mean_local_relative_defect",
			"std_local_relative_defect",
			f"{method_name}: mean local-step symplecticity error",
		),
		(
			"mean_accumulated_relative_defect",
			"std_accumulated_relative_defect",
			f"{method_name}: mean accumulated-flow symplecticity error",
		),
	)
	prepared = _symplecticity_series(records, ordered_labels, fields)
	figure, axes = plt.subplots(
		2,
		1,
		figsize=(10, 8),
		sharex=True,
		constrained_layout=True,
	)
	for label, times, metrics in prepared:
		for axis, (means, lower, upper) in zip(axes, metrics, strict=True):
			line = axis.semilogy(times, means, marker="o", markersize=3, label=label)[0]
			axis.fill_between(times, lower, upper, color=line.get_color(), alpha=0.15, linewidth=0.0)
	for axis, (_, _, title) in zip(axes, fields, strict=True):
		axis.set(title=title, ylabel="Relative defect")
		axis.grid(which="both", alpha=0.25)
		axis.legend(title="Integration step")
	axes[-1].set_xlabel("Time")
	return figure, axes


def plot_gc_trajectory_points(
	solution: Solution,
	*,
	method_name: str,
	step_label: str | None = None,
) -> tuple[Figure, Axes]:
	"""Plot sampled planar trajectories as points without connecting lines."""
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
	title = f"{method_name} trajectories"
	if step_label is not None:
		title = f"{title} ({step_label})"
	axis.set(xlabel="$x$", ylabel="$y$", title=title)
	axis.set_aspect("equal", adjustable="datalim")
	axis.grid(alpha=0.25)
	axis.legend()
	return figure, axis


def plot_trajectory_symplecticity_result(
	result: TrajectorySymplecticityResult,
) -> tuple[Figure, np.ndarray]:
	"""Plot arithmetic-mean local and accumulated errors for a study result."""
	return plot_trajectory_symplecticity(
		cast(
			Mapping[str, Sequence[TrajectorySymplecticityRecordView]],
			result.records,
		),
		labels=tuple(step.label for step in result.steps),
		method_name=result.method_name,
	)


def plot_trajectory_symplecticity_trajectories(
	result: TrajectorySymplecticityResult,
	*,
	label: str | None = None,
) -> tuple[Figure, Axes]:
	"""Plot study trajectories as points, using the finest step by default."""
	selected = (
		min(result.steps, key=lambda step: step.value).label
		if label is None
		else label
	)
	if selected not in result.solutions:
		raise ValueError(f"Unknown integration-step label {selected!r}.")
	return plot_gc_trajectory_points(
		result.solutions[selected],
		method_name=result.method_name,
		step_label=selected,
	)


__all__ = [
	"TrajectorySymplecticityRecordView",
	"plot_gc_trajectory_points",
	"plot_trajectory_symplecticity",
	"plot_trajectory_symplecticity_result",
	"plot_trajectory_symplecticity_trajectories",
]
