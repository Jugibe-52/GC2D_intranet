"""Stacked recurrence-distance views for parallel BM4 trajectories."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from studies.bm4_parallel_recurrence import ParallelBM4RecurrenceResult


def plot_parallel_bm4_recurrence_stack(
	result: ParallelBM4RecurrenceResult,
	*,
	period: float,
	recurrence_tolerance_fraction: float = 0.01,
) -> tuple[Figure, np.ndarray]:
	"""Plot one vertically stacked distance-to-initial panel per trajectory."""
	if not isinstance(result, ParallelBM4RecurrenceResult):
		raise TypeError("`result` must be ParallelBM4RecurrenceResult.")
	cell_period = float(period)
	if not np.isfinite(cell_period) or cell_period <= 0.0:
		raise ValueError("`period` must be positive and finite.")
	tolerance_fraction = float(recurrence_tolerance_fraction)
	if (
		not np.isfinite(tolerance_fraction)
		or tolerance_fraction <= 0.0
		or tolerance_fraction >= 0.5
	):
		raise ValueError(
			"`recurrence_tolerance_fraction` must lie strictly between 0 and 0.5."
		)
	if not np.allclose(np.diff(result.times), 1.0):
		raise ValueError("The stacked recurrence plot requires one saved state per cycle.")
	distances = result.recurrence_distances(period=cell_period)
	particle_count = result.config.particle_count
	if distances.shape != (particle_count, result.times.size):
		raise ValueError("Recurrence distances do not match the campaign shape.")
	# Cycle zero is identically zero and would collapse every logarithmic axis.
	cycle_times = result.times[1:]
	display_floor = float(np.finfo(float).eps * cell_period)
	tolerance = tolerance_fraction * cell_period
	figure, axes = plt.subplots(
		particle_count,
		1,
		figsize=(14.0, 1.65 * particle_count),
		sharex=True,
		constrained_layout=True,
		squeeze=False,
	)
	axis_values = np.asarray(axes[:, 0], dtype=object)
	colors = plt.colormaps["viridis"](
		np.linspace(0.06, 0.94, particle_count)
	)
	for particle, axis in enumerate(axis_values):
		assert isinstance(axis, Axes)
		values = distances[particle, 1:]
		nearest_offset = int(np.argmin(values))
		nearest_cycle = int(round(float(cycle_times[nearest_offset])))
		nearest_distance = float(values[nearest_offset])
		nearest_display_distance = float(max(nearest_distance, display_floor))
		axis.semilogy(
			cycle_times,
			np.maximum(values, display_floor),
			color=colors[particle],
			linewidth=1.25,
			marker=".",
			markersize=2.5,
		)
		axis.axhline(tolerance, color="black", linestyle="--", linewidth=0.75)
		axis.scatter(
			np.asarray([nearest_cycle], dtype=float),
			np.asarray([nearest_display_distance], dtype=float),
			color="tab:red",
			s=18,
			zorder=4,
		)
		x_initial, y_initial = result.initial_positions[particle]
		axis.set_ylabel(f"{particle + 1:02d}", rotation=0, labelpad=16)
		axis.set_title(
			f"Trajectory {particle + 1:02d}: "
			f"initial=({x_initial:.3f}, {y_initial:.3f}); "
			f"nearest cycle={nearest_cycle}, distance={nearest_distance:.3e}",
			loc="left",
			fontsize=8,
			pad=2,
		)
		axis.grid(alpha=0.25)
	axis_values[-1].set_xlabel("Cycle boundary n (time t = n)")
	figure.supylabel("Minimum-image distance to each trajectory's initial position")
	figure.suptitle(
		"BM4 recurrence distance for 40 independently integrated trajectories\n"
		f"Dashed line: {100.0 * tolerance_fraction:g}% of the periodic-cell width; "
		"red point: closest later cycle",
		fontsize=13,
	)
	return figure, axis_values


__all__ = ["plot_parallel_bm4_recurrence_stack"]
