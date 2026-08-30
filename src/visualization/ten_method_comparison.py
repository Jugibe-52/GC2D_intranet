"""Plots and animation for the aligned ten-method trajectory comparison."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from potential import Potential
from simulation import Solution

from .particles import _field_normalization, _frame_indices


TEN_METHOD_COLORS: Mapping[str, str] = {
	"Midpoint ABBA": "tab:purple",
	"Midpoint BM4": "tab:brown",
	"ABBA2 reduced (Newton)": "tab:blue",
	"ABBA2 reduced (Broyden)": "cornflowerblue",
	"ABBA2 simultaneous (Newton)": "tab:orange",
	"ABBA2 simultaneous (Broyden)": "goldenrod",
	"BM4 implicit 1 (Newton)": "tab:green",
	"BM4 implicit 1 (Broyden)": "yellowgreen",
	"BM4 implicit 2 (Newton)": "tab:red",
	"BM4 implicit 2 (Broyden)": "lightcoral",
}
TEN_METHOD_SHORT_LABELS: Mapping[str, str] = {
	"Midpoint ABBA": "ABBA\nmidpoint",
	"Midpoint BM4": "BM4\nmidpoint",
	"ABBA2 reduced (Newton)": "ABBA2 reduced\nNewton",
	"ABBA2 reduced (Broyden)": "ABBA2 reduced\nBroyden",
	"ABBA2 simultaneous (Newton)": "ABBA2 simultaneous\nNewton",
	"ABBA2 simultaneous (Broyden)": "ABBA2 simultaneous\nBroyden",
	"BM4 implicit 1 (Newton)": "BM4 1\nNewton",
	"BM4 implicit 1 (Broyden)": "BM4 1\nBroyden",
	"BM4 implicit 2 (Newton)": "BM4 2\nNewton",
	"BM4 implicit 2 (Broyden)": "BM4 2\nBroyden",
}


def _validated_solutions(
	solutions: Mapping[str, Solution],
	*,
	expected_count: int | None = None,
) -> tuple[tuple[str, ...], np.ndarray, int]:
	"""Validate an aligned non-empty collection of planar solutions."""
	if not solutions:
		raise ValueError("At least one labeled solution is required.")
	labels = tuple(solutions)
	if expected_count is not None and len(labels) != expected_count:
		raise ValueError(f"The plot requires exactly {expected_count} solutions.")
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
			raise ValueError("All solutions must share times and particle count.")
	assert reference_times is not None and particle_count is not None
	return labels, reference_times, particle_count


def _mean_periodic_distance_matrix(
	potential: Potential,
	solutions: Mapping[str, Solution],
) -> tuple[tuple[str, ...], np.ndarray]:
	"""Return the symmetric mean minimum-image distance matrix."""
	labels, _, _ = _validated_solutions(solutions, expected_count=10)
	period = float(potential.grid.period)
	positions = {label: solutions[label].positions() for label in labels}
	matrix = np.zeros((len(labels), len(labels)), dtype=float)
	for first_index, first_label in enumerate(labels):
		first_x, first_y = positions[first_label]
		for second_index in range(first_index + 1, len(labels)):
			second_x, second_y = positions[labels[second_index]]
			delta_x = (second_x - first_x + 0.5 * period) % period - 0.5 * period
			delta_y = (second_y - first_y + 0.5 * period) % period - 0.5 * period
			mean_distance = float(np.mean(np.hypot(delta_x, delta_y)))
			matrix[first_index, second_index] = mean_distance
			matrix[second_index, first_index] = mean_distance
	return labels, matrix


def plot_ten_method_trajectory_differences(
	potential: Potential,
	solutions: Mapping[str, Solution],
) -> tuple[Figure, Axes]:
	"""Plot the 10 x 10 mean periodic trajectory-distance matrix."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	labels, distance_matrix = _mean_periodic_distance_matrix(potential, solutions)
	plot_floor = float(np.finfo(float).eps * max(potential.grid.period, 1.0))
	positive = distance_matrix[distance_matrix > 0.0]
	vmin = max(plot_floor, float(np.min(positive))) if positive.size else plot_floor
	vmax = max(float(np.max(distance_matrix)), 10.0 * vmin)
	plot_values = np.maximum(distance_matrix, vmin)

	figure, axis = plt.subplots(figsize=(12, 10), constrained_layout=True)
	image = axis.imshow(
		plot_values,
		cmap="viridis",
		norm=LogNorm(vmin=vmin, vmax=vmax),
	)
	short_labels = [TEN_METHOD_SHORT_LABELS.get(label, label) for label in labels]
	axis.set_xticks(np.arange(len(labels)), labels=short_labels)
	axis.set_yticks(np.arange(len(labels)), labels=short_labels)
	axis.xaxis.tick_top()
	axis.xaxis.set_label_position("top")
	axis.set(
		title="Mean periodic distance between ten trajectory variants",
		xlabel="compared variant",
		ylabel="reference variant",
	)
	# Separate midpoint, implicit ABBA, and implicit BM4 families.
	for boundary in (1.5, 5.5):
		axis.axhline(boundary, color="white", linewidth=2.0)
		axis.axvline(boundary, color="white", linewidth=2.0)
	for row in range(len(labels)):
		for column in range(len(labels)):
			normalized = float(image.norm(plot_values[row, column]))
			axis.text(
				column,
				row,
				f"{distance_matrix[row, column]:.1e}",
				ha="center",
				va="center",
				color="white" if normalized < 0.48 else "black",
				fontsize=7,
			)
	figure.colorbar(image, ax=axis, label="mean periodic particle distance")
	return figure, axis


def plot_ten_method_nonlinear_work(
	solutions: Mapping[str, Solution],
) -> tuple[Figure, np.ndarray]:
	"""Compare nonlinear corrections and residual work for eight variants."""
	labels, times, _ = _validated_solutions(solutions, expected_count=8)
	figure, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
	for index, label in enumerate(labels):
		solution = solutions[label]
		iterations = np.asarray(solution.diagnostics["nonlinear_iterations"], dtype=int)
		residual_evaluations = np.asarray(
			solution.diagnostics["residual_evaluations"], dtype=int
		)
		residuals = np.asarray(
			solution.diagnostics["nonlinear_residual_norms"], dtype=float
		)
		tolerances = np.asarray(solution.diagnostics["nonlinear_tolerances"], dtype=float)
		expected_shape = (times.size - 1,)
		if any(
			value.shape != expected_shape
			for value in (iterations, residual_evaluations, residuals, tolerances)
		):
			raise ValueError("Implicit diagnostics must align with every saved step.")
		color = TEN_METHOD_COLORS.get(label, f"C{index}")
		linestyle = "--" if "Broyden" in label else "-"
		axes[0].step(
			times[1:], iterations, where="mid", color=color, linestyle=linestyle, label=label
		)
		axes[1].step(
			times[1:], residual_evaluations, where="mid", color=color,
			linestyle=linestyle, label=label
		)
		axes[2].semilogy(
			times[1:],
			np.maximum(residuals / tolerances, np.finfo(float).tiny),
			color=color,
			linestyle=linestyle,
			label=label,
		)
	axis_titles = (
		("Nonlinear corrections at each complete step", "iterations"),
		("Explicit residual evaluations at each complete step", "evaluations"),
	)
	for axis, (title, ylabel) in zip(axes[:2], axis_titles, strict=True):
		axis.set(title=title, ylabel=ylabel)
		axis.yaxis.set_major_locator(MaxNLocator(integer=True))
	axes[2].axhline(1.0, color="black", linestyle=":", label="acceptance limit")
	axes[2].set(
		title="Final residual relative to the effective tolerance",
		xlabel="$t_{n+1}$",
		ylabel="residual / tolerance",
	)
	for axis in axes:
		axis.grid(alpha=0.25)
		axis.legend(fontsize=8, ncol=2)
	return figure, axes


def plot_ten_method_runtimes(
	runtimes: Mapping[str, float],
) -> tuple[Figure, Axes]:
	"""Plot wall-clock integration times for all ten variants."""
	if len(runtimes) != 10:
		raise ValueError("The runtime plot requires exactly ten variants.")
	labels = tuple(runtimes)
	values = np.asarray([runtimes[label] for label in labels], dtype=float)
	if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
		raise ValueError("Every runtime must be positive and finite.")
	figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
	positions = np.arange(len(labels))
	axis.barh(
		positions,
		values,
		color=[TEN_METHOD_COLORS.get(label, f"C{index}") for index, label in enumerate(labels)],
	)
	axis.set_yticks(positions, labels=labels)
	axis.invert_yaxis()
	axis.set(
		title="Wall-clock integration time on the common problem",
		xlabel="runtime [s]",
	)
	axis.grid(axis="x", alpha=0.25)
	return figure, axis


def animate_trajectory_points(
	potential: Potential,
	solutions: Mapping[str, Solution],
	*,
	frames: int | None = None,
	interval: int = 80,
	repeat: bool = True,
	cmap: str = "RdBu_r",
	**imshow_kwargs: Any,
) -> FuncAnimation:
	"""Animate accumulated sampled points for aligned labeled solutions."""
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
	figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
	image = axis.imshow(
		fields[:, :, 0].T,
		origin="lower",
		extent=(grid.xmin, grid.xmin + grid.period, grid.ymin, grid.ymin + grid.period),
		aspect="equal",
		cmap=cmap,
		norm=_field_normalization(fields),
		**imshow_kwargs,
	)
	point_clouds: dict[str, Any] = {}
	current_markers: dict[str, Any] = {}
	legend_handles: list[Line2D] = []
	for index, label in enumerate(labels):
		color = TEN_METHOD_COLORS.get(label, f"C{index}")
		point_clouds[label] = axis.scatter(
			[], [], s=7, marker=".", color=color, alpha=0.38, zorder=2 + index
		)
		current_markers[label] = axis.scatter(
			[], [], s=23, marker="o", color=color, edgecolor="white",
			linewidth=0.3, zorder=20 + index
		)
		legend_handles.append(
			Line2D([0], [0], color=color, marker=".", linestyle="none", label=label)
		)
	initial_x, initial_y = positions[labels[0]]
	axis.scatter(
		initial_x[:, 0],
		initial_y[:, 0],
		s=24,
		facecolor="none",
		edgecolor="black",
		linewidth=0.7,
		zorder=35,
	)
	legend_handles.append(
		Line2D(
			[0], [0], marker="o", linestyle="none", markerfacecolor="none",
			markeredgecolor="black", label=f"{particle_count} initial conditions"
		)
	)
	axis.set(
		xlabel="x",
		ylabel="y",
		xlim=(grid.xmin, grid.xmin + grid.period),
		ylim=(grid.ymin, grid.ymin + grid.period),
	)
	axis.legend(handles=legend_handles, loc="upper right", fontsize=7, ncol=2)
	figure.colorbar(image, ax=axis, label="Effective potential")

	def update(frame: int) -> tuple[Any, ...]:
		"""Update the field, accumulated point clouds, and current positions."""
		sample_index = int(indices[frame])
		image.set_data(fields[:, :, frame].T)
		artists: list[Any] = [image]
		for label in labels:
			x, y = positions[label]
			point_clouds[label].set_offsets(
				np.column_stack(
					(x[:, : sample_index + 1].ravel(), y[:, : sample_index + 1].ravel())
				)
			)
			current_markers[label].set_offsets(
				np.column_stack((x[:, sample_index], y[:, sample_index]))
			)
			artists.extend((point_clouds[label], current_markers[label]))
		axis.set_title(f"Trajectory variants at t = {times[sample_index]:.3f}")
		return tuple(artists)

	return FuncAnimation(
		figure,
		update,
		frames=indices.size,
		interval=int(interval),
		blit=False,
		repeat=repeat,
	)


def animate_ten_method_trajectory_points(
	potential: Potential,
	solutions: Mapping[str, Solution],
	*,
	frames: int | None = None,
	interval: int = 80,
	repeat: bool = True,
	cmap: str = "RdBu_r",
	**imshow_kwargs: Any,
) -> FuncAnimation:
	"""Animate accumulated sampled points for exactly ten aligned variants."""
	_validated_solutions(solutions, expected_count=10)
	return animate_trajectory_points(
		potential,
		solutions,
		frames=frames,
		interval=interval,
		repeat=repeat,
		cmap=cmap,
		**imshow_kwargs,
	)


__all__ = [
	"TEN_METHOD_COLORS",
	"TEN_METHOD_SHORT_LABELS",
	"animate_ten_method_trajectory_points",
	"animate_trajectory_points",
	"plot_ten_method_nonlinear_work",
	"plot_ten_method_runtimes",
	"plot_ten_method_trajectory_differences",
]
