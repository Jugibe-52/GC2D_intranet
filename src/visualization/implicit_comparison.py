"""Trajectory animations and nonlinear-work plots for implicit methods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from potential import Potential
from simulation import Solution

from .particles import _field_normalization, _frame_indices


IMPLICIT_METHOD_COLORS: Mapping[str, str] = {
	"ImplicitABBA1": "tab:blue",
	"ImplicitABBA2": "tab:orange",
	"BM4Implicit1": "tab:green",
	"BM4Implicit2": "tab:red",
}
_METHOD_LINESTYLES = ("solid", "dashed", "dashdot", "dotted")
_METHOD_LINEWIDTHS = (3.0, 2.35, 1.7, 1.05)
_METHOD_MARKERS = ("o", "s", "^", "D")
_METHOD_MARKER_SIZES = (42, 32, 23, 15)


def _validated_solutions(
	solutions: Mapping[str, Solution],
) -> tuple[tuple[str, ...], np.ndarray, int]:
	"""Validate an aligned non-empty set of planar particle solutions."""
	if not solutions:
		raise ValueError("At least one labeled solution is required.")
	labels = tuple(solutions)
	reference_times: np.ndarray | None = None
	particle_count: int | None = None
	for label, solution in solutions.items():
		if not isinstance(label, str) or not label:
			raise ValueError("Solution labels must be non-empty strings.")
		if not isinstance(solution, Solution):
			raise TypeError("Every comparison value must be a Solution.")
		times = np.asarray(solution.t, dtype=float)
		x, y = solution.positions()
		if x.shape != y.shape or x.ndim != 2 or x.shape[1] != times.size:
			raise ValueError("Every solution must contain aligned planar trajectories.")
		if reference_times is None:
			reference_times = times
			particle_count = x.shape[0]
		elif not np.array_equal(times, reference_times) or x.shape[0] != particle_count:
			raise ValueError(
				"All compared solutions must share times and particle count."
			)
	assert reference_times is not None and particle_count is not None
	return labels, reference_times, particle_count


def plot_implicit_method_iterations(
	solutions: Mapping[str, Solution],
) -> tuple[Figure, np.ndarray]:
	"""Compare nonlinear corrections, residual evaluations, and convergence."""
	labels, times, _ = _validated_solutions(solutions)
	figure, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
	for index, label in enumerate(labels):
		solution = solutions[label]
		iterations = np.asarray(
			solution.diagnostics["nonlinear_iterations"], dtype=int
		)
		residual_evaluations = np.asarray(
			solution.diagnostics["residual_evaluations"], dtype=int
		)
		residuals = np.asarray(
			solution.diagnostics["nonlinear_residual_norms"], dtype=float
		)
		tolerances = np.asarray(
			solution.diagnostics["nonlinear_tolerances"], dtype=float
		)
		expected_shape = (times.size - 1,)
		if any(
			value.shape != expected_shape
			for value in (iterations, residual_evaluations, residuals, tolerances)
		):
			raise ValueError(
				"Per-step diagnostics require one saved state at every grid node."
			)
		color = IMPLICIT_METHOD_COLORS.get(label, f"C{index}")
		axes[0].step(
			times[1:], iterations, where="mid", color=color, label=label
		)
		axes[1].step(
			times[1:], residual_evaluations, where="mid", color=color, label=label
		)
		axes[2].semilogy(
			times[1:],
			np.maximum(residuals / tolerances, np.finfo(float).tiny),
			color=color,
			label=label,
		)

	axes[0].set(title="Nonlinear iterations at each new step", ylabel="iterations")
	axes[1].set(
		title="Explicit residual evaluations at each new step",
		ylabel="evaluations",
	)
	axes[2].axhline(1.0, color="black", linestyle="--", label="acceptance limit")
	axes[2].set(
		title="Final residual relative to the effective tolerance",
		xlabel="$t_{n+1}$",
		ylabel="residual / tolerance",
	)
	for axis in axes[:2]:
		axis.yaxis.set_major_locator(MaxNLocator(integer=True))
	for axis in axes:
		axis.grid(alpha=0.25)
		axis.legend(fontsize="small", ncol=2)
	return figure, axes


def plot_implicit_trajectory_differences(
	potential: Potential,
	solutions: Mapping[str, Solution],
) -> tuple[Figure, Axes]:
	"""Plot the mean periodic trajectory distance between every method pair."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	labels, _, _ = _validated_solutions(solutions)
	if len(labels) != 4:
		raise ValueError("The distance matrix requires exactly four solutions.")

	period = float(potential.grid.period)
	positions = {label: solutions[label].positions() for label in labels}
	# A minimum-image displacement measures physical separation across periodic edges.
	distance_matrix = np.zeros((len(labels), len(labels)), dtype=float)
	for first_index, first_label in enumerate(labels):
		first_x, first_y = positions[first_label]
		for second_index in range(first_index + 1, len(labels)):
			second_x, second_y = positions[labels[second_index]]
			delta_x = (second_x - first_x + 0.5 * period) % period - 0.5 * period
			delta_y = (second_y - first_y + 0.5 * period) % period - 0.5 * period
			mean_distance = float(np.mean(np.hypot(delta_x, delta_y)))
			distance_matrix[first_index, second_index] = mean_distance
			distance_matrix[second_index, first_index] = mean_distance

	positive_distances = distance_matrix[distance_matrix > 0.0]
	plot_floor = float(np.finfo(float).eps * max(period, 1.0))
	if positive_distances.size:
		minimum_positive = float(np.min(positive_distances))
		vmin = minimum_positive if minimum_positive > plot_floor else plot_floor
	else:
		vmin = plot_floor
	maximum_distance = float(np.max(distance_matrix))
	vmax = maximum_distance if maximum_distance > vmin else vmin * 10.0
	plot_values = np.maximum(distance_matrix, vmin)

	figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
	image = axis.imshow(
		plot_values,
		cmap="viridis",
		norm=LogNorm(vmin=vmin, vmax=vmax),
	)
	axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=30, ha="left")
	axis.set_yticks(np.arange(len(labels)), labels=labels)
	axis.xaxis.tick_top()
	axis.xaxis.set_label_position("top")
	axis.set(
		title="Mean periodic distance between implicit trajectories",
		xlabel="compared method",
		ylabel="reference method",
	)
	# The separators expose the two ABBA and two BM4 formulation blocks.
	axis.axhline(1.5, color="white", linewidth=2.0)
	axis.axvline(1.5, color="white", linewidth=2.0)
	for row in range(len(labels)):
		for column in range(len(labels)):
			normalized = float(image.norm(plot_values[row, column]))
			axis.text(
				column,
				row,
				f"{distance_matrix[row, column]:.3e}",
				ha="center",
				va="center",
				color="white" if normalized < 0.45 else "black",
				fontsize="small",
			)
	figure.colorbar(image, ax=axis, label="mean periodic particle distance")
	return figure, axis


def animate_implicit_method_trajectories(
	potential: Potential,
	solutions: Mapping[str, Solution],
	*,
	frames: int | None = None,
	interval: int = 80,
	repeat: bool = True,
	cmap: str = "RdBu_r",
	**imshow_kwargs: Any,
) -> FuncAnimation:
	"""Animate aligned particle trajectories with one color per method."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if isinstance(interval, (bool, np.bool_)) or int(interval) <= 0:
		raise ValueError("`interval` must be a positive integer.")
	labels, times, particle_count = _validated_solutions(solutions)
	indices = _frame_indices(times.size, frames)
	frame_times = times[indices]
	fields = np.asarray(potential.evaluate(frame_times), dtype=float)
	if fields.shape != (*potential.grid.shape, indices.size):
		raise ValueError("Potential evaluation returned an unexpected animation shape.")
	positions = {label: solutions[label].positions() for label in labels}

	grid = potential.grid
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
	collections: dict[str, LineCollection] = {}
	markers: dict[str, Any] = {}
	legend_handles: list[Line2D] = []
	for index, label in enumerate(labels):
		color = IMPLICIT_METHOD_COLORS.get(label, f"C{index}")
		collection = LineCollection(
			[],
			colors=color,
			linewidths=_METHOD_LINEWIDTHS[index % len(_METHOD_LINEWIDTHS)],
			linestyles=_METHOD_LINESTYLES[index % len(_METHOD_LINESTYLES)],
			alpha=0.9,
			zorder=2 + index,
		)
		axis.add_collection(collection)
		collections[label] = collection
		markers[label] = axis.scatter(
			[],
			[],
			s=_METHOD_MARKER_SIZES[index % len(_METHOD_MARKER_SIZES)],
			marker=_METHOD_MARKERS[index % len(_METHOD_MARKERS)],
			color=color,
			edgecolor="white",
			linewidth=0.35,
			zorder=7 + index,
		)
		legend_handles.append(
			Line2D(
				[0],
				[0],
				color=color,
				linestyle=_METHOD_LINESTYLES[index % len(_METHOD_LINESTYLES)],
				marker=_METHOD_MARKERS[index % len(_METHOD_MARKERS)],
				label=label,
			)
		)
	initial_x, initial_y = positions[labels[0]]
	axis.scatter(
		initial_x[:, 0],
		initial_y[:, 0],
		s=18,
		facecolor="none",
		edgecolor="black",
		linewidth=0.7,
		label=f"{particle_count} initial conditions",
		zorder=4,
	)
	legend_handles.append(
		Line2D(
			[0],
			[0],
			marker="o",
			linestyle="none",
			markerfacecolor="none",
			markeredgecolor="black",
			label=f"{particle_count} initial conditions",
		)
	)
	axis.set(
		xlabel="x",
		ylabel="y",
		xlim=(grid.xmin, grid.xmin + grid.period),
		ylim=(grid.ymin, grid.ymin + grid.period),
	)
	axis.legend(handles=legend_handles, loc="upper right", fontsize="small")
	figure.colorbar(image, ax=axis, label="Effective potential")

	def update(frame: int) -> tuple[Any, ...]:
		"""Update the field, accumulated paths, and markers for each method."""
		sample_index = int(indices[frame])
		image.set_data(fields[:, :, frame].T)
		artists: list[Any] = [image]
		for label in labels:
			x, y = positions[label]
			segments = [
				np.column_stack(
					(x[particle, : sample_index + 1], y[particle, : sample_index + 1])
				)
				for particle in range(particle_count)
			]
			collections[label].set_segments(segments)
			markers[label].set_offsets(
				np.column_stack((x[:, sample_index], y[:, sample_index]))
			)
			artists.extend((collections[label], markers[label]))
		axis.set_title(
			"Four implicit methods with a common step "
			f"at t = {times[sample_index]:.3f}"
		)
		return tuple(artists)

	return FuncAnimation(
		figure,
		update,
		frames=indices.size,
		interval=int(interval),
		blit=False,
		repeat=repeat,
	)


__all__ = [
	"IMPLICIT_METHOD_COLORS",
	"animate_implicit_method_trajectories",
	"plot_implicit_method_iterations",
	"plot_implicit_trajectory_differences",
]
