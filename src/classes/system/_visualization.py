"""Private visualizations that combine systems, trajectories and solutions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from classes.potential import Potential
from classes.trajectory import Area

from .solution import Solution


ProjectedDiagnostics = tuple[np.ndarray, np.ndarray, np.ndarray]


def _validated_solution_series(
	trajectory: Area,
	solutions: Sequence[Solution],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
	"""Validate solutions sampled at common times for direct comparison."""
	if not solutions:
		raise ValueError("At least one solution is required for an area animation.")
	initial_state = trajectory.state
	assert initial_state is not None
	reference_times: np.ndarray | None = None
	state_series: list[np.ndarray] = []
	for solution in solutions:
		if not isinstance(solution, Solution):
			raise TypeError("Every animated solution must be a Solution instance.")
		times = np.asarray(solution.t, dtype=float)
		states = np.asarray(solution.y, dtype=float)
		if (
			times.ndim != 1
			or times.size < 2
			or states.ndim != 2
			or states.shape != (initial_state.size, times.size)
		):
			raise ValueError(
				"Each solution must contain at least two times and matching Area states."
			)
		if (
			not np.all(np.isfinite(times))
			or not np.all(np.isfinite(states))
			or np.any(np.diff(times) <= 0)
		):
			raise ValueError(
				"Solutions must contain finite states and strictly increasing times."
			)
		if reference_times is None:
			reference_times = times
		else:
			time_scale = max(1.0, float(np.max(np.abs(reference_times))))
			tolerance = float(32 * np.finfo(float).eps * time_scale)
			if times.shape != reference_times.shape or not np.allclose(
				times,
				reference_times,
				rtol=0.0,
				atol=tolerance,
			):
				raise ValueError(
					"Compared solutions must be saved at the same times."
				)
		state_series.append(states)
	assert reference_times is not None
	return reference_times, tuple(state_series)


def _validated_labels(labels: Sequence[str], series_count: int) -> tuple[str, ...]:
	"""Require one distinct, non-empty display label per solution."""
	values = tuple(labels)
	if (
		len(values) != series_count
		or any(not isinstance(label, str) or not label.strip() for label in values)
		or len(set(values)) != len(values)
	):
		raise ValueError("Each solution requires a distinct, non-empty label.")
	return values


def _validated_relative_diagnostics(
	diagnostic_times: np.ndarray | None,
	relative_symplecticity_errors: np.ndarray | None,
	relative_copy_separations: np.ndarray | None,
) -> ProjectedDiagnostics | None:
	"""Validate synchronized, non-negative projected-trajectory diagnostics."""
	provided = (
		diagnostic_times,
		relative_symplecticity_errors,
		relative_copy_separations,
	)
	if all(value is None for value in provided):
		return None
	if any(value is None for value in provided):
		raise ValueError(
			"`diagnostic_times`, `relative_symplecticity_errors` and "
			"`relative_copy_separations` must be provided together."
		)
	assert diagnostic_times is not None
	assert relative_symplecticity_errors is not None
	assert relative_copy_separations is not None
	times = np.asarray(diagnostic_times, dtype=float)
	symplecticity_errors = np.asarray(relative_symplecticity_errors, dtype=float)
	copy_separations = np.asarray(relative_copy_separations, dtype=float)
	if (
		times.ndim != 1
		or times.size < 1
		or symplecticity_errors.shape != times.shape
		or copy_separations.shape != times.shape
	):
		raise ValueError(
			"Projected diagnostics must be one-dimensional arrays of equal length."
		)
	if (
		not np.all(np.isfinite(times))
		or not np.all(np.isfinite(symplecticity_errors))
		or not np.all(np.isfinite(copy_separations))
		or np.any(np.diff(times) <= 0)
	):
		raise ValueError(
			"Projected diagnostics must be finite and have strictly increasing times."
		)
	if np.any(symplecticity_errors < 0) or np.any(copy_separations < 0):
		raise ValueError("Relative projected diagnostics must be non-negative.")
	return times, symplecticity_errors, copy_separations


def _validated_diagnostic_series(
	diagnostic_times: Sequence[np.ndarray | None],
	relative_symplecticity_errors: Sequence[np.ndarray | None],
	relative_copy_separations: Sequence[np.ndarray | None],
	series_count: int,
) -> tuple[ProjectedDiagnostics, ...] | None:
	"""Validate either complete diagnostics for every series or none at all."""
	if not (
		len(diagnostic_times)
		== len(relative_symplecticity_errors)
		== len(relative_copy_separations)
		== series_count
	):
		raise ValueError("Diagnostic sequences must match the number of solutions.")
	validated = tuple(
		_validated_relative_diagnostics(times, defects, separations)
		for times, defects, separations in zip(
			diagnostic_times,
			relative_symplecticity_errors,
			relative_copy_separations,
			strict=True,
		)
	)
	if all(item is None for item in validated):
		return None
	if any(item is None for item in validated):
		raise ValueError("Projected diagnostics are required for every comparison.")
	return tuple(item for item in validated if item is not None)


def _positive_log_limits(values: Sequence[np.ndarray]) -> tuple[float, float]:
	"""Return stable global log limits, including diagnostics that begin at zero."""
	combined = np.concatenate(tuple(values))
	positive = combined[combined > 0]
	if positive.size:
		lower = max(float(np.min(positive)) / 2, float(np.finfo(float).tiny))
		upper = max(float(np.max(positive)) * 2, lower * 10)
	else:
		lower = float(np.finfo(float).eps)
		upper = 10 * lower
	return lower, upper


def _linear_limits(values: Sequence[np.ndarray]) -> tuple[float, float]:
	"""Return padded global limits for several signed linear diagnostics."""
	lower = min(float(np.min(value)) for value in values)
	upper = max(float(np.max(value)) for value in values)
	span = upper - lower
	padding = float(
		0.05 * span
		if span > 0
		else 0.05 * abs(lower) or float(np.finfo(float).eps)
	)
	return lower - padding, upper + padding


def animate_gc_area(
	potential: Potential,
	trajectory: Area,
	solutions: Sequence[Solution],
	*,
	labels: Sequence[str],
	frames: int | None,
	interval: int,
	cmap: str,
	repeat: bool,
	diagnostic_times: Sequence[np.ndarray | None],
	relative_symplecticity_errors: Sequence[np.ndarray | None],
	relative_copy_separations: Sequence[np.ndarray | None],
	pcolormesh_kwargs: dict[str, Any],
) -> FuncAnimation:
	"""Render one or more transported areas and synchronized diagnostics.

	All compared solutions share saved times. Their contours and scalar histories
	use a common color per integration step. Optional projected diagnostics add
	logarithmic panels for normalized symplecticity error and normalized separation
	of the two internal extended-phase-space trajectories.
	"""
	times, state_series = _validated_solution_series(trajectory, solutions)
	series_count = len(state_series)
	series_labels = _validated_labels(labels, series_count)
	if frames is not None and (
		isinstance(frames, (bool, np.bool_))
		or not isinstance(frames, (int, np.integer))
		or frames < 2
	):
		raise ValueError("`frames` must be None or an integer of at least 2.")
	if (
		isinstance(interval, (bool, np.bool_))
		or not isinstance(interval, (int, np.integer))
		or interval <= 0
	):
		raise ValueError("`interval` must be a positive integer.")
	diagnostics = _validated_diagnostic_series(
		diagnostic_times,
		relative_symplecticity_errors,
		relative_copy_separations,
		series_count,
	)

	# Each array has one signed polygon area per common saved time. Normalizing by
	# the positive initial magnitude preserves the direction of each area change.
	area_values = tuple(
		trajectory.calculate_area(states, period=potential.grid.period)
		for states in state_series
	)
	initial_areas = tuple(float(values[0]) for values in area_values)
	if any(initial_area == 0.0 for initial_area in initial_areas):
		raise ValueError("Every initial area must be non-zero.")
	relative_area_errors = tuple(
		(values - initial_area) / abs(initial_area)
		for values, initial_area in zip(area_values, initial_areas, strict=True)
	)

	# Subsampling changes only displayed frames. Every diagnostic retains its full
	# temporal resolution and is revealed up to the current animation time.
	frame_count = times.size if frames is None else min(int(frames), times.size)
	frame_indices = np.linspace(0, times.size - 1, frame_count, dtype=int)
	frame_times = times[frame_indices]
	fields = [potential.evaluate(time) for time in frame_times]
	vmin = min(float(np.min(field)) for field in fields)
	vmax = max(float(np.max(field)) for field in fields)
	if vmin < 0 < vmax:
		norm: mcolors.Normalize = mcolors.TwoSlopeNorm(
			vmin=vmin,
			vcenter=0.0,
			vmax=vmax,
		)
	elif np.isclose(vmin, vmax):
		delta = abs(vmin) * 0.01 or 1.0
		norm = mcolors.Normalize(vmin=vmin - delta, vmax=vmax + delta)
	else:
		norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

	# Roughly twenty arrows per axis retain field direction without hiding the
	# compared contours. One global scale makes magnitudes comparable over time.
	quiver_stride = max(1, int(np.ceil(max(potential.grid.shape) / 20)))
	quiver_x, quiver_y = np.meshgrid(
		potential.grid.x[::quiver_stride],
		potential.grid.y[::quiver_stride],
		indexing="ij",
	)
	electric_fields = [
		potential.electric_field(time, quiver_x, quiver_y) for time in frame_times
	]
	max_magnitude = max(
		float(np.max(np.hypot(field_x, field_y)))
		for field_x, field_y in electric_fields
	)
	arrow_length = 0.75 * quiver_stride * min(potential.grid.dx, potential.grid.dy)
	quiver_scale = (
		None if np.isclose(max_magnitude, 0.0) else max_magnitude / arrow_length
	)

	if diagnostics is None:
		fig, (ax_field, ax_area) = plt.subplots(
			1,
			2,
			figsize=(12, 5),
			constrained_layout=True,
		)
		ax_symplecticity = None
		ax_separation = None
	else:
		fig = plt.figure(figsize=(14, 9), constrained_layout=True)
		grid = fig.add_gridspec(3, 2, width_ratios=(1.2, 1.0))
		ax_field = fig.add_subplot(grid[:, 0])
		ax_area = fig.add_subplot(grid[0, 1])
		ax_symplecticity = fig.add_subplot(grid[1, 1])
		ax_separation = fig.add_subplot(grid[2, 1])

	mesh = ax_field.pcolormesh(
		potential.grid.x,
		potential.grid.y,
		fields[0].T,
		shading="auto",
		cmap=cmap,
		norm=norm,
		**pcolormesh_kwargs,
	)
	fig.colorbar(mesh, ax=ax_field, label=r"$\phi_{\mathrm{eff}}$")
	quiver = ax_field.quiver(
		quiver_x,
		quiver_y,
		*electric_fields[0],
		color="black",
		angles="xy",
		scale_units="xy",
		scale=quiver_scale,
		width=0.003,
	)
	comparison = series_count > 1
	shared_colors = tuple(f"C{index}" for index in range(series_count))
	contour_colors = shared_colors if comparison else ("lime",)
	area_colors = shared_colors if comparison else ("tab:red",)
	symplecticity_colors = shared_colors if comparison else ("tab:purple",)
	separation_colors = shared_colors if comparison else ("tab:blue",)

	contours = tuple(
		ax_field.plot(
			[],
			[],
			color=color,
			linewidth=2.0,
			label=label if comparison else None,
		)[0]
		for color, label in zip(contour_colors, series_labels, strict=True)
	)
	ax_field.set(
		xlabel="x",
		ylabel="y",
		aspect="equal",
		xlim=(potential.grid.xmin, potential.grid.xmin + potential.grid.period),
		ylim=(potential.grid.ymin, potential.grid.ymin + potential.grid.period),
	)
	if comparison:
		ax_field.legend(title="Integration step", loc="upper right")

	area_lines = tuple(
		ax_area.plot(
			[],
			[],
			color=color,
			linewidth=1.6,
			label=label if comparison else None,
		)[0]
		for color, label in zip(area_colors, series_labels, strict=True)
	)
	area_markers = tuple(
		ax_area.plot(
			[],
			[],
			"o",
			color=color if comparison else "black",
			markeredgecolor="black",
			markersize=5,
		)[0]
		for color in area_colors
	)
	ax_area.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
	ax_area.set(
		xlabel="t",
		ylabel=r"$\varepsilon_A(t)=(A(t)-A(0))/|A(0)|$",
		title="Relative area error evolution",
		xlim=(float(times[0]), float(times[-1])),
		ylim=_linear_limits(relative_area_errors),
	)
	ax_area.grid(alpha=0.25)
	if comparison:
		ax_area.legend(loc="best", fontsize="small")

	symplecticity_lines: tuple[Any, ...] = ()
	symplecticity_markers: tuple[Any, ...] = ()
	separation_lines: tuple[Any, ...] = ()
	separation_markers: tuple[Any, ...] = ()
	diagnostic_plot_times: tuple[np.ndarray, ...] | None = None
	relative_symplecticity_values: tuple[np.ndarray, ...] | None = None
	relative_copy_values: tuple[np.ndarray, ...] | None = None
	plot_symplecticity_values: tuple[np.ndarray, ...] | None = None
	plot_copy_values: tuple[np.ndarray, ...] | None = None
	if diagnostics is not None:
		diagnostic_plot_times = tuple(item[0] for item in diagnostics)
		relative_symplecticity_values = tuple(item[1] for item in diagnostics)
		relative_copy_values = tuple(item[2] for item in diagnostics)
		symplecticity_limits = _positive_log_limits(relative_symplecticity_values)
		separation_limits = _positive_log_limits(relative_copy_values)
		plot_symplecticity_values = tuple(
			np.maximum(values, symplecticity_limits[0])
			for values in relative_symplecticity_values
		)
		plot_copy_values = tuple(
			np.maximum(values, separation_limits[0])
			for values in relative_copy_values
		)
		assert ax_symplecticity is not None
		assert ax_separation is not None
		symplecticity_lines = tuple(
			ax_symplecticity.plot(
				[],
				[],
				color=color,
				linewidth=1.6,
				label=label if comparison else None,
			)[0]
			for color, label in zip(
				symplecticity_colors,
				series_labels,
				strict=True,
			)
		)
		symplecticity_markers = tuple(
			ax_symplecticity.plot(
				[],
				[],
				"o",
				color=color if comparison else "black",
				markeredgecolor="black",
				markersize=5,
			)[0]
			for color in symplecticity_colors
		)
		ax_symplecticity.set(
			xlabel="t",
			ylabel=r"$\|DG^T\Omega DG-\Omega\|_F/\|\Omega\|_F$",
			title="Relative symplecticity error of projected trajectories",
			xlim=(float(times[0]), float(times[-1])),
			ylim=symplecticity_limits,
			yscale="log",
		)
		ax_symplecticity.grid(alpha=0.25)
		separation_lines = tuple(
			ax_separation.plot(
				[],
				[],
				color=color,
				linewidth=1.6,
				label=label if comparison else None,
			)[0]
			for color, label in zip(
				separation_colors,
				series_labels,
				strict=True,
			)
		)
		separation_markers = tuple(
			ax_separation.plot(
				[],
				[],
				"o",
				color=color if comparison else "black",
				markeredgecolor="black",
				markersize=5,
			)[0]
			for color in separation_colors
		)
		ax_separation.set(
			xlabel="t",
			ylabel=r"$\|z_1-z_2\|_2/\|(z_1+z_2)/2\|_2$",
			title="Relative separation of internal trajectories",
			xlim=(float(times[0]), float(times[-1])),
			ylim=separation_limits,
			yscale="log",
		)
		ax_separation.grid(alpha=0.25)
		if comparison:
			ax_symplecticity.legend(loc="best", fontsize="small")
			ax_separation.legend(loc="best", fontsize="small")

	# Every component block has shape (boundary vertices, saved times). Periodic
	# wrapping is applied independently so each numerical contour remains legible.
	component_series = tuple(trajectory.split(states) for states in state_series)
	period = potential.grid.period
	xmin = potential.grid.xmin
	ymin = potential.grid.ymin

	def wrapped_contour(
		series_index: int,
		solution_index: int,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Wrap one contour and break artificial edges crossing the periodic cell."""
		components = component_series[series_index]
		x = ((components.x[:, solution_index] - xmin) % period) + xmin
		y = ((components.y[:, solution_index] - ymin) % period) + ymin
		plot_x = [float(x[0])]
		plot_y = [float(y[0])]
		for vertex in range(x.size):
			next_vertex = (vertex + 1) % x.size
			if (
				abs(x[next_vertex] - x[vertex]) > period / 2
				or abs(y[next_vertex] - y[vertex]) > period / 2
			):
				plot_x.append(np.nan)
				plot_y.append(np.nan)
			plot_x.append(float(x[next_vertex]))
			plot_y.append(float(y[next_vertex]))
		return np.asarray(plot_x), np.asarray(plot_y)

	def update(index: int) -> tuple[Any, ...]:
		"""Reveal every trajectory and diagnostic up to one common saved time."""
		solution_index = int(frame_indices[index])
		mesh.set_array(fields[index].T)
		quiver.set_UVC(*electric_fields[index])
		for series_index, contour in enumerate(contours):
			contour.set_data(*wrapped_contour(series_index, solution_index))
		for area_line, area_marker, relative_error in zip(
			area_lines,
			area_markers,
			relative_area_errors,
			strict=True,
		):
			area_line.set_data(
				times[: solution_index + 1],
				relative_error[: solution_index + 1],
			)
			area_marker.set_data(
				[times[solution_index]],
				[relative_error[solution_index]],
			)
		if comparison:
			ax_field.set_title(
				rf"GC contour comparison, $t={times[solution_index]:.3f}$"
			)
		else:
			running_max_error = float(
				np.max(np.abs(relative_area_errors[0][: solution_index + 1]))
			)
			ax_area.set_title(
				"Relative area error evolution\n"
				+ rf"$\varepsilon_A={relative_area_errors[0][solution_index]:.3e}$, "
				+ rf"$\max_{{s\leq t}}|\varepsilon_A(s)|={running_max_error:.3e}$"
			)
			ax_field.set_title(
				rf"Effective GC potential, $t={times[solution_index]:.3f}$"
				+ "\n"
				+ rf"$A={area_values[0][solution_index]:.6g}$, "
				+ rf"$\varepsilon_A={relative_area_errors[0][solution_index]:.3e}$"
			)

		artists: list[Any] = [mesh, quiver, *contours]
		for area_line, area_marker in zip(area_lines, area_markers, strict=True):
			artists.extend((area_line, area_marker))
		artists.append(ax_field.title)
		if diagnostics is not None:
			assert diagnostic_plot_times is not None
			assert relative_symplecticity_values is not None
			assert relative_copy_values is not None
			assert plot_symplecticity_values is not None
			assert plot_copy_values is not None
			assert ax_symplecticity is not None
			assert ax_separation is not None
			current_time = float(times[solution_index])
			time_tolerance = 32 * np.finfo(float).eps * max(1.0, abs(current_time))
			for series_index in range(series_count):
				diagnostic_stop = int(
					np.searchsorted(
						diagnostic_plot_times[series_index],
						current_time + time_tolerance,
						side="right",
					)
				)
				symplecticity_line = symplecticity_lines[series_index]
				symplecticity_marker = symplecticity_markers[series_index]
				separation_line = separation_lines[series_index]
				separation_marker = separation_markers[series_index]
				if diagnostic_stop:
					symplecticity_line.set_data(
						diagnostic_plot_times[series_index][:diagnostic_stop],
						plot_symplecticity_values[series_index][:diagnostic_stop],
					)
					symplecticity_marker.set_data(
						[
							diagnostic_plot_times[series_index][
								diagnostic_stop - 1
							]
						],
						[
							plot_symplecticity_values[series_index][
								diagnostic_stop - 1
							]
						],
					)
					separation_line.set_data(
						diagnostic_plot_times[series_index][:diagnostic_stop],
						plot_copy_values[series_index][:diagnostic_stop],
					)
					separation_marker.set_data(
						[
							diagnostic_plot_times[series_index][
								diagnostic_stop - 1
							]
						],
						[
							plot_copy_values[series_index][diagnostic_stop - 1]
						],
					)
				else:
					symplecticity_line.set_data([], [])
					symplecticity_marker.set_data([], [])
					separation_line.set_data([], [])
					separation_marker.set_data([], [])
				if not comparison and diagnostic_stop:
					current_symplecticity = float(
						relative_symplecticity_values[0][diagnostic_stop - 1]
					)
					current_separation = float(
						relative_copy_values[0][diagnostic_stop - 1]
					)
					ax_symplecticity.set_title(
						"Relative symplecticity error of the projected trajectory\n"
						+ rf"$\varepsilon_\Omega={current_symplecticity:.3e}$, "
						+ rf"$\max={np.max(relative_symplecticity_values[0][:diagnostic_stop]):.3e}$"
					)
					ax_separation.set_title(
						"Relative separation of internal trajectories\n"
						+ rf"$\delta_z={current_separation:.3e}$, "
						+ rf"$\max={np.max(relative_copy_values[0][:diagnostic_stop]):.3e}$"
					)
			for line, marker in zip(
				symplecticity_lines,
				symplecticity_markers,
				strict=True,
			):
				artists.extend((line, marker))
			for line, marker in zip(
				separation_lines,
				separation_markers,
				strict=True,
			):
				artists.extend((line, marker))
			artists.extend((ax_symplecticity.title, ax_separation.title))
		return tuple(artists)

	update(0)
	animation = FuncAnimation(
		fig,
		update,
		frames=frame_count,
		interval=int(interval),
		blit=False,
		repeat=repeat,
	)
	plt.close(fig)
	return animation


__all__: list[str] = []
