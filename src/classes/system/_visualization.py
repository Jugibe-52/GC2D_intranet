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


def animate_gc_area(
	potential: Potential,
	trajectory: Area,
	solution: Solution,
	*,
	frames: int | None,
	interval: int,
	cmap: str,
	repeat: bool,
	pcolormesh_kwargs: dict[str, Any],
) -> FuncAnimation:
	"""Render a transported area over a GC effective potential."""
	times = np.asarray(solution.t, dtype=float)
	states = np.asarray(solution.y, dtype=float)
	initial_state = trajectory.state
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

	area_values = trajectory.calculate_area(states, period=potential.grid.period)
	initial_area = float(area_values[0])
	if initial_area == 0.0:
		raise ValueError("The initial area must be non-zero.")
	relative_error = (area_values - initial_area) / abs(initial_area)

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

	fig, (ax_field, ax_error) = plt.subplots(
		1,
		2,
		figsize=(12, 5),
		constrained_layout=True,
	)
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
	error_min = float(np.min(relative_error))
	error_max = float(np.max(relative_error))
	error_span = error_max - error_min
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

	components = trajectory.split(states)
	period = potential.grid.period
	xmin = potential.grid.xmin
	ymin = potential.grid.ymin

	def wrapped_contour(solution_index: int) -> tuple[np.ndarray, np.ndarray]:
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
		solution_index = int(frame_indices[index])
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
		ax_field.set_title(
			rf"Potencial efectivo GC, $t={times[solution_index]:.3f}$"
			+ "\n"
			+ rf"$A={area_values[solution_index]:.6g}$, "
			+ rf"$\varepsilon_A={relative_error[solution_index]:.3e}$"
		)
		return mesh, quiver, contour, error_line, error_marker, ax_field.title

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
