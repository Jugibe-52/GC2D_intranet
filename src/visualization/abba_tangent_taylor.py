"""Trajectory-drift plots for the tangent-Taylor ABBA methods."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from studies.abba_tangent_taylor_comparison import ABBATangentTaylorComparisonResult

from .particles import _field_normalization, _frame_indices


def _validated_result(
	result: ABBATangentTaylorComparisonResult,
	particle_index: int,
) -> int:
	"""Validate one aligned result and return the selected particle index."""
	if not isinstance(result, ABBATangentTaylorComparisonResult):
		raise TypeError("`result` must be an ABBATangentTaylorComparisonResult.")
	if isinstance(particle_index, (bool, np.bool_)) or not isinstance(
		particle_index,
		(int, np.integer),
	):
		raise TypeError("`particle_index` must be an integer.")
	particle_count = result.particle_distances.shape[0]
	index = int(particle_index)
	if not 0 <= index < particle_count:
		raise IndexError("`particle_index` is outside the compared trajectory range.")
	return index


def plot_tangent_taylor_trajectory_comparison(
	result: ABBATangentTaylorComparisonResult,
	*,
	particle_index: int = 0,
) -> tuple[Figure, np.ndarray]:
	"""Plot one orbit pair and the periodic distance accumulated over time."""
	index = _validated_result(result, particle_index)
	base_x, base_y = result.base_solution.positions()
	tangent_x, tangent_y = result.tangent_solution.positions()
	times = np.asarray(result.base_solution.t)
	figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
	axes[0].plot(
		base_x[index],
		base_y[index],
		label=result.base_method_name,
		linewidth=2.2,
	)
	axes[0].plot(
		tangent_x[index],
		tangent_y[index],
		label=result.tangent_method_name,
		linestyle="--",
		linewidth=1.8,
	)
	axes[0].scatter(
		[base_x[index, 0]],
		[base_y[index, 0]],
		marker="o",
		color="black",
		label="initial point",
		zorder=4,
	)
	axes[0].set(
		title=f"Physical trajectory of particle {index}",
		xlabel="$x$",
		ylabel="$y$",
		aspect="equal",
	)
	axes[0].grid(alpha=0.25)
	axes[0].legend(fontsize="small")

	particle_drift = result.particle_distances[index]
	rms_drift = result.rms_distance
	maximum_drift = result.max_distance
	# The aligned initial states have exactly zero separation. NaN omits that one
	# point from the logarithmic axis without replacing it by an artificial floor.
	axes[1].semilogy(
		times,
		np.where(particle_drift > 0.0, particle_drift, np.nan),
		label=f"particle {index}",
	)
	axes[1].semilogy(
		times,
		np.where(rms_drift > 0.0, rms_drift, np.nan),
		label="RMS over particles",
		linestyle="--",
	)
	axes[1].semilogy(
		times,
		np.where(maximum_drift > 0.0, maximum_drift, np.nan),
		label="maximum over particles",
		linestyle=":",
	)
	axes[1].set(
		title="Periodic trajectory drift from the original method",
		xlabel="$t$",
		ylabel="minimum-image distance",
	)
	axes[1].grid(which="both", alpha=0.25)
	axes[1].legend(fontsize="small")
	return figure, axes


def plot_tangent_taylor_component_comparison(
	result: ABBATangentTaylorComparisonResult,
	*,
	particle_index: int = 0,
) -> tuple[Figure, np.ndarray]:
	"""Plot coordinate histories and their signed minimum-image displacement."""
	index = _validated_result(result, particle_index)
	times = np.asarray(result.base_solution.t)
	base_x, base_y = result.base_solution.positions()
	tangent_x, tangent_y = result.tangent_solution.positions()
	delta_x, delta_y = result.periodic_displacement_components
	figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
	axes[0].plot(times, base_x[index], label=f"{result.base_method_name}: x")
	axes[0].plot(times, tangent_x[index], linestyle="--", label="tangent-Taylor: x")
	axes[0].plot(times, base_y[index], label=f"{result.base_method_name}: y")
	axes[0].plot(times, tangent_y[index], linestyle="--", label="tangent-Taylor: y")
	axes[0].set(
		title=f"Coordinate histories of particle {index}",
		ylabel="coordinate",
	)
	axes[0].grid(alpha=0.25)
	axes[0].legend(fontsize="small", ncol=2)
	axes[1].plot(times, delta_x[index], label=r"$\Delta x$")
	axes[1].plot(times, delta_y[index], label=r"$\Delta y$")
	axes[1].set(
		title="Signed tangent-Taylor minus original displacement",
		xlabel="$t$",
		ylabel="minimum-image displacement",
	)
	axes[1].axhline(0.0, color="black", linewidth=0.8)
	axes[1].grid(alpha=0.25)
	axes[1].legend()
	return figure, axes


def animate_tangent_taylor_particle_evolution(
	result: ABBATangentTaylorComparisonResult,
	*,
	frames: int | None = None,
	interval: int = 80,
	repeat: bool = True,
	cmap: str = "RdBu_r",
	**imshow_kwargs: Any,
) -> FuncAnimation:
	"""Animate original and tangent-Taylor particles with accumulated paths."""
	if not isinstance(result, ABBATangentTaylorComparisonResult):
		raise TypeError("`result` must be an ABBATangentTaylorComparisonResult.")
	if isinstance(interval, (bool, np.bool_)) or int(interval) <= 0:
		raise ValueError("`interval` must be a positive integer.")
	times = np.asarray(result.base_solution.t, dtype=float)
	indices = _frame_indices(times.size, frames)
	frame_times = times[indices]
	fields = np.asarray(result.potential.evaluate(frame_times), dtype=float)
	if fields.shape != (*result.potential.grid.shape, indices.size):
		raise ValueError("Potential evaluation returned an unexpected animation shape.")
	base_x, base_y = result.base_solution.positions()
	tangent_x, tangent_y = result.tangent_solution.positions()
	particle_count = base_x.shape[0]
	grid = result.potential.grid

	figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
	image = axis.imshow(
		fields[:, :, 0].T,
		origin="lower",
		extent=(grid.xmin, grid.xmin + grid.period, grid.ymin, grid.ymin + grid.period),
		aspect="equal",
		cmap=cmap,
		norm=_field_normalization(fields),
		**imshow_kwargs,
	)
	base_paths = LineCollection([], colors="tab:blue", linewidths=1.7, alpha=0.8)
	tangent_paths = LineCollection(
		[],
		colors="tab:orange",
		linewidths=1.5,
		linestyles="dashed",
		alpha=0.85,
	)
	connectors = LineCollection(
		[],
		colors="white",
		linewidths=0.8,
		linestyles="dotted",
		alpha=0.8,
	)
	axis.add_collection(base_paths)
	axis.add_collection(tangent_paths)
	axis.add_collection(connectors)
	base_markers = axis.scatter(
		[],
		[],
		s=34,
		marker="o",
		color="tab:blue",
		edgecolor="white",
		linewidth=0.5,
		zorder=7,
	)
	tangent_markers = axis.scatter(
		[],
		[],
		s=38,
		marker="x",
		color="tab:orange",
		linewidth=1.4,
		zorder=8,
	)
	axis.scatter(
		base_x[:, 0],
		base_y[:, 0],
		s=22,
		facecolor="none",
		edgecolor="black",
		linewidth=0.8,
		zorder=6,
	)
	legend_handles = (
		Line2D(
			[0],
			[0],
			color="tab:blue",
			marker="o",
			label=result.base_method_name,
		),
		Line2D(
			[0],
			[0],
			color="tab:orange",
			linestyle="--",
			marker="x",
			label=result.tangent_method_name,
		),
		Line2D(
			[0],
			[0],
			color="white",
			linestyle=":",
			label="current pair separation",
		),
	)
	axis.set(
		xlabel="$x$",
		ylabel="$y$",
		xlim=(grid.xmin, grid.xmin + grid.period),
		ylim=(grid.ymin, grid.ymin + grid.period),
	)
	axis.legend(handles=legend_handles, loc="upper right", fontsize="small")
	figure.colorbar(image, ax=axis, label="Potential")

	def update(frame: int) -> tuple[Any, ...]:
		"""Move particles, extend both path families, and show pair separation."""
		sample_index = int(indices[frame])
		image.set_data(fields[:, :, frame].T)
		base_paths.set_segments(
			[
				np.column_stack(
					(base_x[particle, : sample_index + 1], base_y[particle, : sample_index + 1])
				)
				for particle in range(particle_count)
			]
		)
		tangent_paths.set_segments(
			[
				np.column_stack(
					(
						tangent_x[particle, : sample_index + 1],
						tangent_y[particle, : sample_index + 1],
					)
				)
				for particle in range(particle_count)
			]
		)
		base_points = np.column_stack(
			(base_x[:, sample_index], base_y[:, sample_index])
		)
		tangent_points = np.column_stack(
			(tangent_x[:, sample_index], tangent_y[:, sample_index])
		)
		base_markers.set_offsets(base_points)
		tangent_markers.set_offsets(tangent_points)
		connectors.set_segments(
			[np.vstack((base_points[i], tangent_points[i])) for i in range(particle_count)]
		)
		axis.set_title(
			f"Particle evolution at t = {times[sample_index]:.3f}; "
			f"max drift = {result.max_distance[sample_index]:.3e}"
		)
		return (
			image,
			base_paths,
			tangent_paths,
			connectors,
			base_markers,
			tangent_markers,
		)

	return FuncAnimation(
		figure,
		update,
		frames=indices.size,
		interval=int(interval),
		blit=False,
		repeat=repeat,
	)


__all__ = [
	"animate_tangent_taylor_particle_evolution",
	"plot_tangent_taylor_component_comparison",
	"plot_tangent_taylor_trajectory_comparison",
]
