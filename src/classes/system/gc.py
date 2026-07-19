"""Guiding-centre system."""

from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from classes.potential import Potential
from classes.trajectory import Area, TrajectoryGC

from ._integration import solve_gc
from .solution import Solution
from .system import System


class SystemGC(System):
	"""Guiding-centre dynamics over a gyroaveraged potential."""

	trajectory: TrajectoryGC

	def __init__(self, potential: Potential, trajectory: TrajectoryGC) -> None:
		if not isinstance(trajectory, TrajectoryGC):
			raise TypeError("SystemGC requires a TrajectoryGC instance.")
		super().__init__(potential, trajectory)
		self.effective_potential = potential.gyroaverage(trajectory.rho)

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		x, y = self.trajectory.positions(state)
		ex, ey = self.effective_potential.electric_field(t, x, y)
		return np.concatenate((ey, -ex))

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		x, y = self.trajectory.positions(state)
		return self.effective_potential.evaluate(t, x, y)

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		x, y = self.trajectory.positions(state)
		return -self.effective_potential.evaluate(t, x, y, dt=1)

	def animate_area(
		self,
		solution: Solution,
		*,
		frames: int | None = 120,
		interval: int = 50,
		cmap: str = "RdBu_r",
		repeat: bool = True,
		**pcolormesh_kwargs: Any,
	) -> FuncAnimation:
		"""Animate an area over the effective potential and its relative error."""
		if not isinstance(self.trajectory, Area):
			raise TypeError("`animate_area` requires an Area trajectory.")
		if not isinstance(solution, Solution):
			raise TypeError("`solution` must be a Solution instance.")

		times = np.asarray(solution.t, dtype=float)
		states = np.asarray(solution.y, dtype=float)
		initial_state = self.trajectory.state
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

		area_values = self.trajectory.calculate_area(
			states,
			period=self.potential.grid.period,
		)
		initial_area = float(area_values[0])
		if initial_area == 0.0:
			raise ValueError("The initial area must be non-zero.")
		relative_error = (area_values - initial_area) / abs(initial_area)

		frame_count = times.size if frames is None else min(int(frames), times.size)
		frame_indices = np.linspace(0, times.size - 1, frame_count, dtype=int)
		frame_times = times[frame_indices]
		fields = [self.effective_potential.evaluate(time) for time in frame_times]
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

		quiver_stride = max(1, int(np.ceil(max(self.potential.grid.shape) / 20)))
		quiver_x, quiver_y = np.meshgrid(
			self.potential.grid.x[::quiver_stride],
			self.potential.grid.y[::quiver_stride],
			indexing="ij",
		)
		electric_fields = [
			self.effective_potential.electric_field(time, quiver_x, quiver_y)
			for time in frame_times
		]
		max_magnitude = max(
			float(np.max(np.hypot(field_x, field_y)))
			for field_x, field_y in electric_fields
		)
		arrow_length = (
			0.75
			* quiver_stride
			* min(self.potential.grid.dx, self.potential.grid.dy)
		)
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
			self.potential.grid.x,
			self.potential.grid.y,
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
			xlim=(self.potential.grid.xmin, self.potential.grid.xmin + self.potential.grid.period),
			ylim=(self.potential.grid.ymin, self.potential.grid.ymin + self.potential.grid.period),
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

		x_all, y_all = self.trajectory.positions(states)
		period = self.potential.grid.period
		xmin = self.potential.grid.xmin
		ymin = self.potential.grid.ymin

		def wrapped_contour(solution_index: int) -> tuple[np.ndarray, np.ndarray]:
			x = ((x_all[:, solution_index] - xmin) % period) + xmin
			y = ((y_all[:, solution_index] - ymin) % period) + ymin
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

	def _integrate(
		self,
		state: np.ndarray,
		*,
		step: float,
		t_span: tuple[float, float],
		n_save_step: int,
		check_energy: bool,
		progress: bool,
	) -> Solution:
		return solve_gc(
			self,
			state,
			step=step,
			t_span=t_span,
			n_save_step=n_save_step,
			check_energy=check_energy,
			progress=progress,
		)


__all__ = ["SystemGC"]
