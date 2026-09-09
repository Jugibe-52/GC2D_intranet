"""Recurrence-distance and trajectory views for parallel BM4 campaigns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.colors import Normalize

from potential import Potential

if TYPE_CHECKING:
	from studies.bm4_parallel_recurrence import ParallelBM4RecurrenceResult


def _validate_result(result: object) -> None:
	"""Validate a recurrence result without creating a package import cycle."""
	from studies.bm4_parallel_recurrence import ParallelBM4RecurrenceResult

	if not isinstance(result, ParallelBM4RecurrenceResult):
		raise TypeError("`result` must be ParallelBM4RecurrenceResult.")


def _periodic_path(
	coordinates: np.ndarray,
	*,
	period: float,
) -> np.ndarray:
	"""Insert NaN breaks where a wrapped path crosses a periodic boundary."""
	if coordinates.shape[0] < 2:
		return coordinates
	jumps = np.any(np.abs(np.diff(coordinates, axis=0)) > 0.5 * period, axis=1)
	if not np.any(jumps):
		return coordinates
	path: list[np.ndarray] = [coordinates[0]]
	for index, point in enumerate(coordinates[1:]):
		if jumps[index]:
			path.append(np.full(2, np.nan))
		path.append(point)
	return np.asarray(path, dtype=float)


def animate_parallel_bm4_trajectories(
	potential: Potential,
	result: ParallelBM4RecurrenceResult,
	*,
	interval: int = 200,
	repeat: bool = True,
	potential_cmap: str = "RdBu_r",
	trajectory_cmap: str = "viridis",
) -> FuncAnimation:
	"""Animate all saved parallel trajectories inside the periodic cell."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	_validate_result(result)
	if isinstance(interval, (bool, np.bool_)) or int(interval) <= 0:
		raise ValueError("`interval` must be a positive integer.")
	if not isinstance(repeat, (bool, np.bool_)):
		raise TypeError("`repeat` must be a boolean.")

	grid = potential.grid
	period = float(grid.period)
	particle_count = result.config.particle_count
	times = np.asarray(result.times, dtype=float)
	positions = np.asarray(result.positions, dtype=float)
	wrapped_x = grid.xmin + (positions[:, 0] - grid.xmin) % period
	wrapped_y = grid.ymin + (positions[:, 1] - grid.ymin) % period
	fields = np.asarray(potential.evaluate_grid(times), dtype=float)
	if fields.shape != (*grid.shape, times.size):
		raise ValueError("Potential evaluation returned an unexpected animation shape.")

	trajectory_norm = Normalize(vmin=1.0, vmax=float(particle_count))
	trajectory_colormap = plt.colormaps[trajectory_cmap]
	trajectory_numbers = np.arange(1, particle_count + 1, dtype=float)
	trajectory_colors = trajectory_colormap(trajectory_norm(trajectory_numbers))
	field_minimum = float(np.min(fields))
	field_maximum = float(np.max(fields))

	figure, axis = plt.subplots(figsize=(9.5, 7.5), constrained_layout=True)
	image = axis.imshow(
		fields[:, :, 0].T,
		origin="lower",
		extent=(grid.xmin, grid.xmin + period, grid.ymin, grid.ymin + period),
		cmap=potential_cmap,
		vmin=field_minimum,
		vmax=field_maximum,
		aspect="equal",
	)
	paths = LineCollection(
		[],
		colors=trajectory_colors,
		linewidths=1.2,
		alpha=0.85,
		zorder=3,
	)
	axis.add_collection(paths)
	current_points = axis.scatter(
		wrapped_x[:, 0],
		wrapped_y[:, 0],
		c=trajectory_numbers,
		cmap=trajectory_colormap,
		norm=trajectory_norm,
		s=48,
		edgecolor="white",
		linewidth=0.6,
		zorder=5,
	)
	axis.scatter(
		wrapped_x[:, 0],
		wrapped_y[:, 0],
		s=58,
		facecolor="none",
		edgecolor="black",
		linewidth=0.65,
		zorder=4,
	)
	axis.set(
		xlabel="$x$",
		ylabel="$y$",
		xlim=(grid.xmin, grid.xmin + period),
		ylim=(grid.ymin, grid.ymin + period),
	)
	figure.colorbar(image, ax=axis, label="Effective potential")
	figure.colorbar(current_points, ax=axis, label="Trajectory")

	def update(frame: int) -> tuple[Any, ...]:
		"""Update the potential, wrapped histories, and current positions."""
		image.set_data(fields[:, :, frame].T)
		paths.set_segments(
			[
				_periodic_path(
					np.column_stack(
						(wrapped_x[particle, : frame + 1], wrapped_y[particle, : frame + 1])
					),
					period=period,
				)
				for particle in range(particle_count)
			]
		)
		current_points.set_offsets(
			np.column_stack((wrapped_x[:, frame], wrapped_y[:, frame]))
		)
		axis.set_title(
			f"Parallel BM4 trajectories at saved time t = {times[frame]:g}"
		)
		return image, paths, current_points, axis.title

	return FuncAnimation(
		figure,
		update,
		frames=times.size,
		interval=int(interval),
		blit=False,
		repeat=bool(repeat),
	)


def plot_parallel_bm4_recurrence_stack(
	result: ParallelBM4RecurrenceResult,
	*,
	period: float,
	recurrence_tolerance_fraction: float = 0.01,
) -> tuple[Figure, np.ndarray]:
	"""Plot one vertically stacked distance-to-initial panel per trajectory."""
	_validate_result(result)
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
		f"BM4 recurrence distance for {particle_count} independently integrated "
		"trajectories\n"
		f"Dashed line: {100.0 * tolerance_fraction:g}% of the periodic-cell width; "
		"red point: closest later cycle",
		fontsize=13,
	)
	return figure, axis_values


__all__ = [
	"animate_parallel_bm4_trajectories",
	"plot_parallel_bm4_recurrence_stack",
]
