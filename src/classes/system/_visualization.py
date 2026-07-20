"""Private visualizations that combine systems, trajectories and solutions."""

from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from classes.potential import Potential
from classes.trajectory import Area

from .solution import Solution


def _validated_relative_diagnostics(
	diagnostic_times: np.ndarray | None,
	relative_symplecticity_errors: np.ndarray | None,
	relative_copy_separations: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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

	# The None case was rejected above; these arrays contain one diagnostic sample
	# per complete projected integration step, independent of solution saving.
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


def _positive_log_series(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
	"""Replace exact zeros for log display and return stable global limits."""
	positive = values[values > 0]
	if positive.size:
		lower = max(float(np.min(positive)) / 2, float(np.finfo(float).tiny))
		upper = max(float(np.max(positive)) * 2, lower * 10)
	else:
		# A fully exact diagnostic still needs a finite visible log-scale range.
		lower = float(np.finfo(float).eps)
		upper = 10 * lower
	return np.maximum(values, lower), (lower, upper)


def animate_gc_area(
	potential: Potential,
	trajectory: Area,
	solution: Solution,
	*,
	frames: int | None,
	interval: int,
	cmap: str,
	repeat: bool,
	diagnostic_times: np.ndarray | None,
	relative_symplecticity_errors: np.ndarray | None,
	relative_copy_separations: np.ndarray | None,
	pcolormesh_kwargs: dict[str, Any],
) -> FuncAnimation:
	"""Render a transported area and synchronized relative diagnostics.

	The field and electric arrows share the left panel. The right panel follows
	the signed change relative to the initial polygon area. When projected
	diagnostics are supplied, two additional log panels follow the relative
	symplecticity error and relative separation of the internal trajectories.
	``frames`` is the maximum number of displayed solution samples (all are used
	when it is ``None``), while ``interval`` is the display delay in milliseconds.
	"""
	# ``times`` indexes saved columns and ``states`` keeps the Area's [x, y]
	# block layout: their expected shapes are (saved_times,) and (2N, saved_times).
	times = np.asarray(solution.t, dtype=float)
	states = np.asarray(solution.y, dtype=float)
	initial_state = trajectory.state
	# ``SystemGC.animate_area`` guarantees an Area with an initialized state;
	# keeping the assertion here also narrows the type for static analysis.
	assert initial_state is not None
	if (
		times.ndim != 1
		or times.size < 2
		or states.ndim != 2
		or states.shape != (initial_state.size, times.size)
	):
		raise ValueError(
			"`solution` must contain at least two times and matching Area states."
		)
	if (
		not np.all(np.isfinite(times))
		or not np.all(np.isfinite(states))
		or np.any(np.diff(times) <= 0)
	):
		raise ValueError(
			"The solution must contain finite states and strictly increasing times."
		)
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
	diagnostics = _validated_relative_diagnostics(
		diagnostic_times,
		relative_symplecticity_errors,
		relative_copy_separations,
	)

	# These arrays contain one scalar diagnostic per saved time. Area is signed
	# according to contour orientation; the denominator below is positive so the
	# relative error retains the direction of its change.
	area_values = trajectory.calculate_area(states, period=potential.grid.period)
	initial_area = float(area_values[0])
	if initial_area == 0.0:
		raise ValueError("The initial area must be non-zero.")
	relative_error = (area_values - initial_area) / abs(initial_area)

	# ``frame_indices`` maps animation frames back to columns of the full saved
	# solution; subsampling changes only the visualization, never the diagnostics.
	frame_count = times.size if frames is None else min(int(frames), times.size)
	frame_indices = np.linspace(0, times.size - 1, frame_count, dtype=int)
	frame_times = times[frame_indices]
	# Precomputing all displayed fields gives every frame the same color limits
	# and keeps animation callbacks free of repeated spline evaluations. Every
	# entry in ``fields`` has the potential grid's (nx, ny) convention.
	fields = [potential.evaluate(time) for time in frame_times]
	vmin = min(float(np.min(field)) for field in fields)
	vmax = max(float(np.max(field)) for field in fields)
	if vmin < 0 < vmax:
		# A zero-centered diverging scale makes opposite potential signs visually
		# comparable even when their extrema have different magnitudes.
		norm: mcolors.Normalize = mcolors.TwoSlopeNorm(
			vmin=vmin,
			vcenter=0.0,
			vmax=vmax,
		)
	elif np.isclose(vmin, vmax):
		# Matplotlib needs a non-degenerate range for a spatially constant field.
		delta = abs(vmin) * 0.01 or 1.0
		norm = mcolors.Normalize(vmin=vmin - delta, vmax=vmax + delta)
	else:
		norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

	# Limit arrows to roughly twenty samples per axis: the scalar field retains
	# full resolution while the vector overlay stays legible. ``quiver_x`` and
	# ``quiver_y`` are the sparse physical coordinates where vectors are sampled.
	quiver_stride = max(1, int(np.ceil(max(potential.grid.shape) / 20)))
	quiver_x, quiver_y = np.meshgrid(
		potential.grid.x[::quiver_stride],
		potential.grid.y[::quiver_stride],
		indexing="ij",
	)
	# Each tuple is (Ex, Ey) on the sparse quiver grid for one displayed time.
	electric_fields = [
		potential.electric_field(time, quiver_x, quiver_y) for time in frame_times
	]
	max_magnitude = max(
		float(np.max(np.hypot(field_x, field_y)))
		for field_x, field_y in electric_fields
	)
	# ``arrow_length`` is a visual target based on the spacing between sampled
	# arrows, not another physical field magnitude.
	arrow_length = 0.75 * quiver_stride * min(potential.grid.dx, potential.grid.dy)
	# A global scale preserves relative arrow magnitudes across time. ``None``
	# lets Matplotlib handle the special case of an identically zero field.
	quiver_scale = (
		None if np.isclose(max_magnitude, 0.0) else max_magnitude / arrow_length
	)

	if diagnostics is None:
		fig, (ax_field, ax_error) = plt.subplots(
			1,
			2,
			figsize=(12, 5),
			constrained_layout=True,
		)
		ax_symplecticity = None
		ax_separation = None
	else:
		# The field spans the full height while the three scalar histories share the
		# right column. This keeps the transported contour spatially legible.
		fig = plt.figure(figsize=(14, 9), constrained_layout=True)
		grid = fig.add_gridspec(3, 2, width_ratios=(1.2, 1.0))
		ax_field = fig.add_subplot(grid[:, 0])
		ax_error = fig.add_subplot(grid[0, 1])
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
	contour = ax_field.plot([], [], color="lime", linewidth=2.0)[0]
	ax_field.set(
		xlabel="x",
		ylabel="y",
		aspect="equal",
		xlim=(potential.grid.xmin, potential.grid.xmin + potential.grid.period),
		ylim=(potential.grid.ymin, potential.grid.ymin + potential.grid.period),
	)

	error_line = ax_error.plot([], [], color="tab:red", linewidth=1.6)[0]
	error_marker = ax_error.plot([], [], "o", color="black", markersize=5)[0]
	ax_error.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
	# These extrema describe the entire diagnostic, including frames skipped by
	# visual subsampling, so the error axis stays fixed throughout the animation.
	error_min = float(np.min(relative_error))
	error_max = float(np.max(relative_error))
	error_span = error_max - error_min
	# Pad both constant and varying traces so the line is never clipped against
	# a degenerate y-axis.
	error_padding = (
		0.05 * error_span
		if error_span > 0
		else 0.05 * abs(error_min) or np.finfo(float).eps
	)
	ax_error.set(
		xlabel="t",
		ylabel=r"$\varepsilon_A(t)=(A(t)-A(0))/|A(0)|$",
		title="Error relativo del área",
		xlim=(float(times[0]), float(times[-1])),
		ylim=(error_min - error_padding, error_max + error_padding),
	)
	ax_error.grid(alpha=0.25)

	symplecticity_line = None
	symplecticity_marker = None
	separation_line = None
	separation_marker = None
	diagnostic_plot_times = None
	plot_symplecticity_errors = None
	plot_copy_separations = None
	if diagnostics is not None:
		(
			diagnostic_plot_times,
			relative_symplecticity_values,
			relative_copy_values,
		) = diagnostics
		plot_symplecticity_errors, symplecticity_limits = _positive_log_series(
			relative_symplecticity_values
		)
		plot_copy_separations, separation_limits = _positive_log_series(
			relative_copy_values
		)
		assert ax_symplecticity is not None
		assert ax_separation is not None
		symplecticity_line = ax_symplecticity.plot(
			[], [], color="tab:purple", linewidth=1.6
		)[0]
		symplecticity_marker = ax_symplecticity.plot(
			[], [], "o", color="black", markersize=5
		)[0]
		ax_symplecticity.set(
			xlabel="t",
			ylabel=r"$\|DG^T\Omega DG-\Omega\|_F/\|\Omega\|_F$",
			title="Error simpléctico relativo de la trayectoria proyectada",
			xlim=(float(times[0]), float(times[-1])),
			ylim=symplecticity_limits,
			yscale="log",
		)
		ax_symplecticity.grid(alpha=0.25)
		separation_line = ax_separation.plot(
			[], [], color="tab:blue", linewidth=1.6
		)[0]
		separation_marker = ax_separation.plot(
			[], [], "o", color="black", markersize=5
		)[0]
		ax_separation.set(
			xlabel="t",
			ylabel=r"$\|z_1-z_2\|_2/\|(z_1+z_2)/2\|_2$",
			title="Separación relativa de las trayectorias internas",
			xlim=(float(times[0]), float(times[-1])),
			ylim=separation_limits,
			yscale="log",
		)
		ax_separation.grid(alpha=0.25)

	# The component blocks each have shape (boundary_vertices, saved_times).
	components = trajectory.split(states)
	# ``period``, ``xmin`` and ``ymin`` define the displayed periodic cell used
	# to choose an image for each boundary vertex.
	period = potential.grid.period
	xmin = potential.grid.xmin
	ymin = potential.grid.ymin

	def wrapped_contour(solution_index: int) -> tuple[np.ndarray, np.ndarray]:
		"""Wrap one contour into the cell and break lines crossing its boundary."""
		x = ((components.x[:, solution_index] - xmin) % period) + xmin
		y = ((components.y[:, solution_index] - ymin) % period) + ymin
		plot_x = [float(x[0])]
		plot_y = [float(y[0])]
		for vertex in range(x.size):
			next_vertex = (vertex + 1) % x.size
			# A wrapped edge can otherwise draw a long, artificial segment across
			# the periodic cell. NaNs ask Matplotlib to lift the pen at that edge.
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
		"""Update every artist from the same saved-solution index."""
		solution_index = int(frame_indices[index])
		# pcolormesh expects its first displayed dimension along y, hence the
		# transpose of fields stored with the project's (x, y) convention.
		mesh.set_array(fields[index].T)
		quiver.set_UVC(*electric_fields[index])
		contour.set_data(*wrapped_contour(solution_index))
		error_line.set_data(
			times[: solution_index + 1],
			relative_error[: solution_index + 1],
		)
		error_marker.set_data(
			[times[solution_index]],
			[relative_error[solution_index]],
		)
		# The right panel emphasizes both the instantaneous error and its worst
		# excursion so far; this makes convergence loss visible while the contour
		# deforms in the left panel.
		running_max_error = float(
			np.max(np.abs(relative_error[: solution_index + 1]))
		)
		ax_error.set_title(
			"Evolución del error relativo del área\n"
			+ rf"$\varepsilon_A={relative_error[solution_index]:.3e}$, "
			+ rf"$\max_{{s\leq t}}|\varepsilon_A(s)|={running_max_error:.3e}$"
		)
		ax_field.set_title(
			rf"Potencial efectivo GC, $t={times[solution_index]:.3f}$"
			+ "\n"
			+ rf"$A={area_values[solution_index]:.6g}$, "
			+ rf"$\varepsilon_A={relative_error[solution_index]:.3e}$"
		)
		artists: list[Any] = [
			mesh,
			quiver,
			contour,
			error_line,
			error_marker,
			ax_field.title,
		]
		if diagnostics is not None:
			assert diagnostic_plot_times is not None
			assert plot_symplecticity_errors is not None
			assert plot_copy_separations is not None
			assert symplecticity_line is not None
			assert symplecticity_marker is not None
			assert separation_line is not None
			assert separation_marker is not None
			assert ax_symplecticity is not None
			assert ax_separation is not None
			current_time = float(times[solution_index])
			time_tolerance = 32 * np.finfo(float).eps * max(1.0, abs(current_time))
			diagnostic_stop = int(
				np.searchsorted(
					diagnostic_plot_times,
					current_time + time_tolerance,
					side="right",
				)
			)
			if diagnostic_stop:
				symplecticity_line.set_data(
					diagnostic_plot_times[:diagnostic_stop],
					plot_symplecticity_errors[:diagnostic_stop],
				)
				symplecticity_marker.set_data(
					[diagnostic_plot_times[diagnostic_stop - 1]],
					[plot_symplecticity_errors[diagnostic_stop - 1]],
				)
				separation_line.set_data(
					diagnostic_plot_times[:diagnostic_stop],
					plot_copy_separations[:diagnostic_stop],
				)
				separation_marker.set_data(
					[diagnostic_plot_times[diagnostic_stop - 1]],
					[plot_copy_separations[diagnostic_stop - 1]],
				)
				relative_symplecticity_values = diagnostics[1]
				relative_copy_values = diagnostics[2]
				current_symplecticity = float(
					relative_symplecticity_values[diagnostic_stop - 1]
				)
				current_separation = float(relative_copy_values[diagnostic_stop - 1])
				ax_symplecticity.set_title(
					"Error simpléctico relativo de la trayectoria proyectada\n"
					+ rf"$\varepsilon_\Omega={current_symplecticity:.3e}$, "
					+ rf"$\max={np.max(relative_symplecticity_values[:diagnostic_stop]):.3e}$"
				)
				ax_separation.set_title(
					"Separación relativa de las trayectorias internas\n"
					+ rf"$\delta_z={current_separation:.3e}$, "
					+ rf"$\max={np.max(relative_copy_values[:diagnostic_stop]):.3e}$"
				)
			artists.extend(
				[
					symplecticity_line,
					symplecticity_marker,
					separation_line,
					separation_marker,
					ax_symplecticity.title,
					ax_separation.title,
				]
			)
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
