# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Research diagnostics for composed potential/trajectory systems."""

import logging
from typing import Any, Sequence, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

from contracts import Array

from .fc import SystemFC
from .gc import SystemGC
from .system import System

logger = logging.getLogger(__name__)


class SystemResearch:
	"""Analyse numerical-method properties without owning domain dynamics.

	The object wraps an existing :class:`System`.  Simulations still
	use that system directly; this class only prepares and visualises numerical
	diagnostics and potential fields.
	"""

	def __init__(self, system: System) -> None:
		if not isinstance(system, System):
			raise TypeError('`system` must be a System instance.')
		self.system = system

	@staticmethod
	def _split_augmented_state(
		state: Array,
		state_dimension: int,
	) -> tuple[tuple[Array, ...], Array]:
		"""Separate a phase-space state from its tangent-map matrix."""
		state_array = np.asarray(state)
		component_count = state_dimension + state_dimension**2
		if state_array.ndim == 0 or state_array.shape[0] % component_count != 0:
			raise ValueError(
				f"The first state dimension must be divisible by {component_count} "
				f"for a {state_dimension}-dimensional tangent map."
			)
		components = tuple(np.split(state_array, component_count, axis=0))
		phase_state = components[:state_dimension]
		jacobian = np.asarray(components[state_dimension:]).reshape(
			(state_dimension, state_dimension, -1)
		)
		return phase_state, jacobian

	def y_dot_lyap(self, t: float, state: Array) -> Array:
		"""Return the trajectory and tangent-map derivatives."""
		if isinstance(self.system, SystemGC):
			return self._guiding_center_y_dot_lyap(t, state)
		if isinstance(self.system, SystemFC):
			return self._full_cyclotron_y_dot_lyap(t, state)
		raise TypeError(f"Unsupported system class: {type(self.system).__name__}.")

	def _guiding_center_y_dot_lyap(self, t: float, state: Array) -> Array:
		(x, y), jacobian = self._split_augmented_state(state, state_dimension=2)
		phase_state = np.concatenate((x, y))
		d2psi_dx2 = self.system.psi(t, x, y, dx=2)
		d2psi_dxdy = self.system.psi(t, x, y, dx=1, dy=1)
		d2psi_dy2 = self.system.psi(t, x, y, dy=2)
		linearization = np.zeros_like(jacobian)
		linearization[0, 0], linearization[0, 1] = -d2psi_dxdy, -d2psi_dy2
		linearization[1, 0], linearization[1, 1] = d2psi_dx2, d2psi_dxdy
		jacobian_dot = np.einsum("ijm,jkm->ikm", linearization, jacobian)
		return np.concatenate((self.system.vector_field(t, phase_state), jacobian_dot.reshape(-1)))

	def _full_cyclotron_y_dot_lyap(self, t: float, state: Array) -> Array:
		(x, y, vx, vy), jacobian = self._split_augmented_state(state, state_dimension=4)
		phase_state = np.concatenate((x, y, vx, vy))
		system = cast(SystemFC, self.system)
		d2phi_dx2 = -system.electric_scale * system.phi(t, x, y, dx=2)
		d2phi_dxdy = -system.electric_scale * system.phi(t, x, y, dx=1, dy=1)
		d2phi_dy2 = -system.electric_scale * system.phi(t, x, y, dy=2)
		linearization = np.zeros_like(jacobian)
		linearization[0, 2] = system.velocity_scale
		linearization[1, 3] = system.velocity_scale
		linearization[2, 3] = system.larmor_frequency
		linearization[3, 2] = -system.larmor_frequency
		linearization[2, 0], linearization[2, 1] = d2phi_dx2, d2phi_dxdy
		linearization[3, 0], linearization[3, 1] = d2phi_dxdy, d2phi_dy2
		jacobian_dot = np.einsum("ijm,jkm->ikm", linearization, jacobian)
		return np.concatenate((system.vector_field(t, phase_state), jacobian_dot.reshape(-1)))

	@staticmethod
	def _comparison_norm(*fields: Array) -> mcolors.Normalize:
		"""Return one colour normalization shared by diagnostic field plots."""
		vmin = min(float(np.nanmin(field)) for field in fields)
		vmax = max(float(np.nanmax(field)) for field in fields)
		if vmin < 0 < vmax:
			return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
		if np.isclose(vmin, vmax):
			delta = abs(vmin) * 0.01 or 1.0
			return mcolors.Normalize(vmin=vmin - delta, vmax=vmax + delta)
		return mcolors.Normalize(vmin=vmin, vmax=vmax)

	@staticmethod
	def _validate_step(step: int) -> None:
		if not isinstance(step, (int, np.integer)) or step < 1:
			raise ValueError('`step` must be a positive integer.')

	def plot_phi_psi(
		self,
		t: float = 0.0,
		*,
		contours: int | Sequence[float] | None = 12,
		cmap: str = 'RdBu_r',
		show: bool = True,
		**pcolormesh_kwargs: Any,
	) -> tuple[Figure, np.ndarray]:
		"""Plot the physical potential phi beside the effective potential psi."""
		system = self.system
		phi, psi = system.phi(t), system.psi(t)
		norm = self._comparison_norm(phi, psi)
		fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
		for ax, field, label in zip(axes, (phi, psi), (r'$\phi$', r'$\psi$')):
			mesh = ax.pcolormesh(
				system.grid.x,
				system.grid.y,
				field.T,
				shading='auto',
				cmap=cmap,
				norm=norm,
				**pcolormesh_kwargs,
			)
			if contours is not None:
				ax.contour(
					system.grid.x,
					system.grid.y,
					field.T,
					levels=contours,
					colors='k',
					linewidths=0.45,
					alpha=0.55,
				)
			ax.set(xlabel='x', ylabel='y', title=rf'{label}, $t={t:.3f}$', aspect='equal')
		fig.colorbar(mesh, ax=axes, label='potential')
		if show:
			plt.show()
		return fig, axes

	def plot_electric_phi_psi(
		self,
		t: float = 0.0,
		*,
		step: int = 4,
		contours: int | Sequence[float] | None = 12,
		cmap: str = 'RdBu_r',
		show: bool = True,
		**pcolormesh_kwargs: Any,
	) -> tuple[Figure, np.ndarray]:
		"""Plot phi and psi together with their electric fields."""
		self._validate_step(step)
		system = self.system
		phi, psi = system.phi(t), system.psi(t)
		x_mesh, y_mesh = np.meshgrid(system.grid.x, system.grid.y, indexing='ij')
		electric_fields = (
			system.electric_field(t, x_mesh, y_mesh, effective=False),
			system.electric_field(t, x_mesh, y_mesh),
		)
		max_magnitude = max(float(np.nanmax(np.hypot(ex, ey))) for ex, ey in electric_fields)
		scale = None if np.isclose(max_magnitude, 0.0) else max_magnitude / (2 * min(system.grid.dx, system.grid.dy))
		norm = self._comparison_norm(phi, psi)
		fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
		for ax, potential, (ex, ey), symbol in zip(
			axes,
			(phi, psi),
			electric_fields,
			(r'\phi', r'\psi'),
		):
			mesh = ax.pcolormesh(
				system.grid.x,
				system.grid.y,
				potential.T,
				shading='auto',
				cmap=cmap,
				norm=norm,
				**pcolormesh_kwargs,
			)
			if contours is not None:
				ax.contour(
					system.grid.x,
					system.grid.y,
					potential.T,
					levels=contours,
					colors='k',
					linewidths=0.45,
					alpha=0.55,
				)
			ax.quiver(
				x_mesh[::step, ::step],
				y_mesh[::step, ::step],
				ex[::step, ::step],
				ey[::step, ::step],
				color='black',
				angles='xy',
				scale_units='xy',
				scale=scale,
				width=0.003,
			)
			ax.set(
				xlabel='x',
				ylabel='y',
				title=rf'${symbol}$ and $\mathbf{{E}}=-\nabla {symbol}$, $t={t:.3f}$',
				aspect='equal',
			)
		fig.colorbar(mesh, ax=axes, label='potential')
		if show:
			plt.show()
		return fig, axes

	def plot_psi(
		self,
		t: float = 0.0,
		*,
		contours: int | Sequence[float] | None = 12,
		cmap: str = 'RdBu_r',
		show: bool = True,
		**pcolormesh_kwargs: Any,
	) -> tuple[Figure, Axes]:
		"""Plot the complete effective potential psi at time ``t``."""
		system = self.system
		psi = system.psi(t)
		norm = self._comparison_norm(psi)
		fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
		mesh = ax.pcolormesh(
			system.grid.x,
			system.grid.y,
			psi.T,
			shading='auto',
			cmap=cmap,
			norm=norm,
			**pcolormesh_kwargs,
		)
		if contours is not None:
			ax.contour(
				system.grid.x,
				system.grid.y,
				psi.T,
				levels=contours,
				colors='k',
				linewidths=0.45,
				alpha=0.55,
			)
		fig.colorbar(mesh, ax=ax, label=r'$\psi$')
		ax.set(xlabel='x', ylabel='y', title=rf'Effective potential $\psi$, $t={t:.3f}$', aspect='equal')
		if show:
			plt.show()
		return fig, ax

	def animate_phi_psi(
		self,
		*,
		t_max: float = 2 * np.pi,
		frames: int = 120,
		interval: int = 50,
		cmap: str = 'RdBu_r',
		repeat: bool = True,
		**pcolormesh_kwargs: Any,
	) -> FuncAnimation:
		"""Animate phi and psi side by side."""
		if frames < 2:
			raise ValueError('`frames` must be at least 2.')
		if t_max <= 0:
			raise ValueError('`t_max` must be positive.')
		system = self.system
		times = np.linspace(0.0, t_max, frames, endpoint=False)
		phi_fields = [system.phi(t) for t in times]
		psi_fields = [system.psi(t) for t in times]
		norm = self._comparison_norm(*phi_fields, *psi_fields)
		fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
		meshes = [
			ax.pcolormesh(
				system.grid.x,
				system.grid.y,
				field.T,
				shading='auto',
				cmap=cmap,
				norm=norm,
				**pcolormesh_kwargs,
			)
			for ax, field in zip(axes, (phi_fields[0], psi_fields[0]))
		]
		for ax in axes:
			ax.set(xlabel='x', ylabel='y', aspect='equal')
		fig.colorbar(meshes[0], ax=axes, label='potential')

		def update(index: int) -> tuple[Any, ...]:
			for mesh, field, ax, label in zip(
				meshes,
				(phi_fields[index], psi_fields[index]),
				axes,
				(r'$\phi$', r'$\psi$'),
			):
				mesh.set_array(field.T)
				ax.set_title(rf'{label}, $t={times[index]:.3f}$')
			return *meshes, *(ax.title for ax in axes)

		update(0)
		animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False, repeat=repeat)
		plt.close(fig)
		return animation

	def animate_psi(
		self,
		*,
		t_max: float = 2 * np.pi,
		frames: int = 120,
		interval: int = 50,
		cmap: str = 'RdBu_r',
		repeat: bool = True,
		**pcolormesh_kwargs: Any,
	) -> FuncAnimation:
		"""Animate the complete effective potential psi."""
		animate = getattr(self.system.effective_potential, "animate", None)
		if not callable(animate):
			raise TypeError("This potential representation does not provide animation support.")
		return cast(FuncAnimation, animate(
			t_max=t_max,
			frames=frames,
			interval=interval,
			cmap=cmap,
			repeat=repeat,
			title=r'Effective potential $\psi$',
			**pcolormesh_kwargs,
		))

	def animate_electric_phi_psi(
		self,
		*,
		t_max: float = 2 * np.pi,
		frames: int = 120,
		interval: int = 50,
		step: int = 4,
		cmap: str = 'RdBu_r',
		repeat: bool = True,
		**pcolormesh_kwargs: Any,
	) -> FuncAnimation:
		"""Animate phi, psi, and their corresponding electric fields."""
		if frames < 2:
			raise ValueError('`frames` must be at least 2.')
		if t_max <= 0:
			raise ValueError('`t_max` must be positive.')
		self._validate_step(step)
		system = self.system
		times = np.linspace(0.0, t_max, frames, endpoint=False)
		x_mesh, y_mesh = np.meshgrid(system.grid.x, system.grid.y, indexing='ij')
		phi_fields = [system.phi(t) for t in times]
		psi_fields = [system.psi(t) for t in times]
		phi_electric = [system.electric_field(t, x_mesh, y_mesh, effective=False) for t in times]
		psi_electric = [system.electric_field(t, x_mesh, y_mesh) for t in times]
		max_magnitude = max(
			float(np.nanmax(np.hypot(ex, ey)))
			for fields in (phi_electric, psi_electric)
			for ex, ey in fields
		)
		scale = None if np.isclose(max_magnitude, 0.0) else max_magnitude / (2 * min(system.grid.dx, system.grid.dy))
		norm = self._comparison_norm(*phi_fields, *psi_fields)
		fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
		meshes = [
			ax.pcolormesh(
				system.grid.x,
				system.grid.y,
				field.T,
				shading='auto',
				cmap=cmap,
				norm=norm,
				**pcolormesh_kwargs,
			)
			for ax, field in zip(axes, (phi_fields[0], psi_fields[0]))
		]
		quivers = [
			ax.quiver(
				x_mesh[::step, ::step],
				y_mesh[::step, ::step],
				ex[::step, ::step],
				ey[::step, ::step],
				color='black',
				angles='xy',
				scale_units='xy',
				scale=scale,
				width=0.003,
			)
			for ax, (ex, ey) in zip(axes, (phi_electric[0], psi_electric[0]))
		]
		for ax in axes:
			ax.set(xlabel='x', ylabel='y', aspect='equal')
		fig.colorbar(meshes[0], ax=axes, label='potential')

		def update(index: int) -> tuple[Any, ...]:
			for mesh, quiver, potential, (ex, ey), ax, symbol in zip(
				meshes,
				quivers,
				(phi_fields[index], psi_fields[index]),
				(phi_electric[index], psi_electric[index]),
				axes,
				(r'\phi', r'\psi'),
			):
				mesh.set_array(potential.T)
				quiver.set_UVC(ex[::step, ::step], ey[::step, ::step])
				ax.set_title(rf'${symbol}$ and $\mathbf{{E}}=-\nabla {symbol}$, $t={times[index]:.3f}$')
			return *meshes, *quivers, *(ax.title for ax in axes)

		update(0)
		animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False, repeat=repeat)
		plt.close(fig)
		return animation

	def animate_electric_psi(
		self,
		*,
		t_max: float = 2 * np.pi,
		frames: int = 120,
		interval: int = 50,
		step: int = 4,
		cmap: str = 'RdBu_r',
		repeat: bool = True,
		**pcolormesh_kwargs: Any,
	) -> FuncAnimation:
		"""Animate psi together with its electric field."""
		if frames < 2:
			raise ValueError('`frames` must be at least 2.')
		if t_max <= 0:
			raise ValueError('`t_max` must be positive.')
		self._validate_step(step)
		system = self.system
		times = np.linspace(0.0, t_max, frames, endpoint=False)
		x_mesh, y_mesh = np.meshgrid(system.grid.x, system.grid.y, indexing='ij')
		psi_fields = [system.psi(t) for t in times]
		electric_fields = [system.electric_field(t, x_mesh, y_mesh) for t in times]
		max_magnitude = max(float(np.nanmax(np.hypot(ex, ey))) for ex, ey in electric_fields)
		scale = None if np.isclose(max_magnitude, 0.0) else max_magnitude / (2 * min(system.grid.dx, system.grid.dy))
		norm = self._comparison_norm(*psi_fields)
		fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
		mesh = ax.pcolormesh(
			system.grid.x,
			system.grid.y,
			psi_fields[0].T,
			shading='auto',
			cmap=cmap,
			norm=norm,
			**pcolormesh_kwargs,
		)
		ex0, ey0 = electric_fields[0]
		quiver = ax.quiver(
			x_mesh[::step, ::step],
			y_mesh[::step, ::step],
			ex0[::step, ::step],
			ey0[::step, ::step],
			color='black',
			angles='xy',
			scale_units='xy',
			scale=scale,
			width=0.003,
		)
		fig.colorbar(mesh, ax=ax, label=r'$\psi$')
		ax.set(xlabel='x', ylabel='y', aspect='equal')

		def update(index: int) -> tuple[Any, ...]:
			ex, ey = electric_fields[index]
			mesh.set_array(psi_fields[index].T)
			quiver.set_UVC(ex[::step, ::step], ey[::step, ::step])
			ax.set_title(rf'$\psi$ and $\mathbf{{E}}=-\nabla\psi$, $t={times[index]:.3f}$')
			return mesh, quiver, ax.title

		update(0)
		animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False, repeat=repeat)
		plt.close(fig)
		return animation

	def animate_electric_psi_trajectories(
		self,
		solution: Any,
		*,
		frames: int | None = None,
		frame_stride: int = 1,
		interval: int = 50,
		step: int = 4,
		cmap: str = 'RdBu_r',
		repeat: bool = True,
		**pcolormesh_kwargs: Any,
	) -> FuncAnimation:
		"""Animate trajectories over psi and its electric field."""
		self._validate_step(step)
		if not isinstance(frame_stride, (int, np.integer)) or frame_stride < 1:
			raise ValueError('`frame_stride` must be a positive integer.')
		if frames is not None and frames < 2:
			raise ValueError('`frames` must be at least 2.')

		times_all = np.asarray(solution.t, dtype=float)
		states = np.asarray(solution.y)
		if times_all.ndim != 1 or states.ndim != 2 or states.shape[1] != times_all.size:
			raise ValueError('`solution` must provide t with shape (n_times,) and y with shape (n_state, n_times).')
		if times_all.size < 2:
			raise ValueError('`solution` must contain at least two time samples.')

		frame_indices = np.arange(0, times_all.size, frame_stride, dtype=int)
		if frame_indices[-1] != times_all.size - 1:
			frame_indices = np.append(frame_indices, times_all.size - 1)
		if frames is not None and frames < frame_indices.size:
			selection = np.linspace(0, frame_indices.size - 1, frames, dtype=int)
			frame_indices = frame_indices[np.unique(selection)]
		times = times_all[frame_indices]

		system = self.system
		x_positions, y_positions = system.get_positions(states)
		x_wrapped, y_wrapped = system.grid.wrap_or_clip(x_positions, y_positions)
		x_lines, y_lines = x_wrapped.copy(), y_wrapped.copy()
		if system.grid.period is not None:
			crosses_boundary = (
				(np.abs(np.diff(x_wrapped, axis=1)) > system.grid.period / 2)
				| (np.abs(np.diff(y_wrapped, axis=1)) > system.grid.period / 2)
			)
			x_lines[:, 1:][crosses_boundary] = np.nan
			y_lines[:, 1:][crosses_boundary] = np.nan

		x_mesh, y_mesh = np.meshgrid(system.grid.x, system.grid.y, indexing='ij')
		psi_fields = [system.psi(t) for t in times]
		electric_fields = [system.electric_field(t, x_mesh, y_mesh) for t in times]
		max_magnitude = max(float(np.nanmax(np.hypot(ex, ey))) for ex, ey in electric_fields)
		scale = None if np.isclose(max_magnitude, 0.0) else max_magnitude / (2 * min(system.grid.dx, system.grid.dy))
		norm = self._comparison_norm(*psi_fields)

		fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
		mesh = ax.pcolormesh(
			system.grid.x,
			system.grid.y,
			psi_fields[0].T,
			shading='auto',
			cmap=cmap,
			norm=norm,
			**pcolormesh_kwargs,
		)
		ex0, ey0 = electric_fields[0]
		quiver = ax.quiver(
			x_mesh[::step, ::step],
			y_mesh[::step, ::step],
			ex0[::step, ::step],
			ey0[::step, ::step],
			color='black',
			angles='xy',
			scale_units='xy',
			scale=scale,
			width=0.003,
		)
		lines = [
			ax.plot([], [], color='green', lw=1.5, label=f'trajectory {index + 1}')[0]
			for index in range(x_wrapped.shape[0])
		]
		markers = [
			ax.plot([], [], marker='o', color='green', markersize=4)[0]
			for _ in range(x_wrapped.shape[0])
		]
		fig.colorbar(mesh, ax=ax, label=r'$\psi$')
		if x_wrapped.shape[0] > 1:
			ax.legend(loc='upper right')
		ax.set(xlabel='x', ylabel='y', aspect='equal')

		def update(index: int) -> tuple[Any, ...]:
			current = int(frame_indices[index])
			ex, ey = electric_fields[index]
			mesh.set_array(psi_fields[index].T)
			quiver.set_UVC(ex[::step, ::step], ey[::step, ::step])
			for line, marker, x_path, y_path, x_display, y_display in zip(
				lines,
				markers,
				x_wrapped,
				y_wrapped,
				x_lines,
				y_lines,
			):
				line.set_data(x_display[:current + 1], y_display[:current + 1])
				marker.set_data([x_path[current]], [y_path[current]])
			ax.set_title(rf'$\psi$, $\mathbf{{E}}=-\nabla\psi$, $t={times[index]:.3f}$')
			return mesh, quiver, *lines, *markers, ax.title

		update(0)
		animation = FuncAnimation(
			fig,
			update,
			frames=times.size,
			interval=interval,
			blit=False,
			repeat=repeat,
		)
		plt.close(fig)
		return animation

	def _unwrap_polygon_coordinates(
		self,
		x_vertices: Array,
		y_vertices: Array,
	) -> tuple[Array, Array]:
		"""Put consecutive periodic polygon vertices in one local image."""
		x_unwrapped = np.asarray(x_vertices, dtype=float).copy()
		y_unwrapped = np.asarray(y_vertices, dtype=float).copy()
		if self.system.grid.period is None:
			return x_unwrapped, y_unwrapped
		period = self.system.grid.period
		for vertex in range(1, x_unwrapped.shape[0]):
			delta_x = x_vertices[vertex] - x_vertices[vertex - 1]
			delta_y = y_vertices[vertex] - y_vertices[vertex - 1]
			delta_x -= period * np.round(delta_x / period)
			delta_y -= period * np.round(delta_y / period)
			x_unwrapped[vertex] = x_unwrapped[vertex - 1] + delta_x
			y_unwrapped[vertex] = y_unwrapped[vertex - 1] + delta_y
		return x_unwrapped, y_unwrapped

	def guiding_center_area_element(
		self,
		solution: Any,
		trajectory_indices: Sequence[int] = (0, 1, 2),
	) -> Array:
		"""Return the oriented local area :math:`dX\\wedge dY` along a GC solution.

		The three selected trajectories represent a reference point, the endpoint
		of an initial ``dX`` displacement, and the endpoint of an initial ``dY``
		displacement.  At every stored time the oriented area is the determinant
		of those two transported displacement vectors.  For a periodic potential,
		displacements use the minimum-image convention; the three trajectories
		therefore need to remain close compared with half the spatial period.
		"""
		if self.system.trajectory.kind != 'gc':
			raise ValueError('Position-space dX wedge dY conservation is defined here only for guiding-center trajectories.')
		states = np.asarray(solution.y)
		if states.ndim != 2:
			raise ValueError('`solution.y` must have shape (n_state, n_times).')
		if states.shape[0] % 2 != 0:
			raise ValueError('A guiding-center solution must contain equally many x and y rows.')
		if len(trajectory_indices) != 3:
			raise ValueError('`trajectory_indices` must contain reference, dX, and dY trajectory indices.')
		if any(not isinstance(index, (int, np.integer)) for index in trajectory_indices):
			raise ValueError('`trajectory_indices` must contain integers.')
		reference, endpoint_x, endpoint_y = (int(index) for index in trajectory_indices)
		if len({reference, endpoint_x, endpoint_y}) != 3:
			raise ValueError('`trajectory_indices` must contain three distinct trajectories.')
		n_trajectories = states.shape[0] // 2
		if min(reference, endpoint_x, endpoint_y) < 0 or max(reference, endpoint_x, endpoint_y) >= n_trajectories:
			raise ValueError(
				f'`trajectory_indices` must lie between 0 and {n_trajectories - 1}.'
			)

		x_all, y_all = self.system.get_positions(states)
		dx_x = x_all[endpoint_x] - x_all[reference]
		dx_y = y_all[endpoint_x] - y_all[reference]
		dy_x = x_all[endpoint_y] - x_all[reference]
		dy_y = y_all[endpoint_y] - y_all[reference]
		if self.system.grid.period is not None:
			period = self.system.grid.period
			dx_x -= period * np.round(dx_x / period)
			dx_y -= period * np.round(dx_y / period)
			dy_x -= period * np.round(dy_x / period)
			dy_y -= period * np.round(dy_y / period)
		return np.asarray(dx_x * dy_y - dx_y * dy_x)

	def guiding_center_polygon_area(
		self,
		solution: Any,
		trajectory_indices: Sequence[int] | None = None,
	) -> Array:
		"""Return the signed area enclosed by transported GC boundary points."""
		if self.system.trajectory.kind != 'gc':
			raise ValueError('Position-space polygon area is defined here only for guiding-center trajectories.')
		states = np.asarray(solution.y)
		if states.ndim != 2 or states.shape[0] % 2 != 0:
			raise ValueError('A guiding-center solution must contain equally many x and y rows.')
		n_trajectories = states.shape[0] // 2
		if trajectory_indices is None:
			indices = tuple(range(n_trajectories))
		else:
			if any(not isinstance(index, (int, np.integer)) for index in trajectory_indices):
				raise ValueError('`trajectory_indices` must contain integers.')
			indices = tuple(int(index) for index in trajectory_indices)
		if len(indices) < 4 or len(set(indices)) != len(indices):
			raise ValueError('A polygon requires at least four distinct trajectory indices.')
		if min(indices) < 0 or max(indices) >= n_trajectories:
			raise ValueError(f'`trajectory_indices` must lie between 0 and {n_trajectories - 1}.')

		x_all, y_all = self.system.get_positions(states)
		x_vertices, y_vertices = self._unwrap_polygon_coordinates(
			x_all[np.asarray(indices)],
			y_all[np.asarray(indices)],
		)
		area = 0.5 * np.sum(
			x_vertices * np.roll(y_vertices, -1, axis=0)
			- y_vertices * np.roll(x_vertices, -1, axis=0),
			axis=0,
		)
		return np.asarray(area)

	def animate_electric_psi_area_conservation(
		self,
		solution: Any,
		*,
		trajectory_indices: Sequence[int] | None = None,
		frames: int | None = None,
		frame_stride: int = 1,
		interval: int = 50,
		step: int = 4,
		cmap: str = 'RdBu_r',
		repeat: bool = True,
		**pcolormesh_kwargs: Any,
	) -> FuncAnimation:
		"""Animate GC trajectories and conservation of :math:`dX\\wedge dY`.

		With three selected trajectories, the indices represent a reference point
		and the endpoints of two infinitesimal displacements.  With four or more,
		they represent counter-clockwise samples of a finite polygon boundary.  If
		``trajectory_indices`` is omitted, every trajectory is used.  The left
		panel transports the area over :math:`\\psi` and its electric field; the
		right panel shows ``(area - area[0]) / abs(area[0])``.

		For a differential element, use small initial displacements.  For a finite
		region, sample each boundary edge with several trajectories so the polygon
		can follow nonlinear curvature.
		"""
		self._validate_step(step)
		times_all = np.asarray(solution.t, dtype=float)
		states_all = np.asarray(solution.y)
		if times_all.ndim != 1 or states_all.ndim != 2 or states_all.shape[1] != times_all.size:
			raise ValueError('`solution` must provide t with shape (n_times,) and y with shape (n_state, n_times).')
		if times_all.size < 2:
			raise ValueError('`solution` must contain at least two time samples.')
		if not np.all(np.isfinite(times_all)) or np.any(np.diff(times_all) <= 0):
			raise ValueError('`solution.t` must contain finite, strictly increasing times.')
		if frames is not None and frames < 2:
			raise ValueError('`frames` must be at least 2.')
		if not isinstance(frame_stride, (int, np.integer)) or frame_stride < 1:
			raise ValueError('`frame_stride` must be a positive integer.')

		if states_all.shape[0] % 2 != 0:
			raise ValueError('A guiding-center solution must contain equally many x and y rows.')
		n_trajectories = states_all.shape[0] // 2
		if trajectory_indices is None:
			indices = tuple(range(n_trajectories))
		else:
			if any(not isinstance(index, (int, np.integer)) for index in trajectory_indices):
				raise ValueError('`trajectory_indices` must contain integers.')
			indices = tuple(int(index) for index in trajectory_indices)
		if len(indices) == 3:
			area_mode = 'element'
			area = self.guiding_center_area_element(solution, indices)
		elif len(indices) >= 4:
			area_mode = 'polygon'
			area = self.guiding_center_polygon_area(solution, indices)
		else:
			raise ValueError('Select three trajectories for dX wedge dY or at least four for a finite area.')
		if not np.all(np.isfinite(area)):
			raise ValueError('The transported area contains non-finite values.')
		initial_area = float(area[0])
		if initial_area == 0.0:
			raise ValueError('The initial dX wedge dY area must be non-zero.')
		relative_error = (area - initial_area) / abs(initial_area)

		frame_indices = np.arange(0, times_all.size, frame_stride, dtype=int)
		if frame_indices[-1] != times_all.size - 1:
			frame_indices = np.append(frame_indices, times_all.size - 1)
		if frames is not None and frames < frame_indices.size:
			selected = np.linspace(0, frame_indices.size - 1, frames, dtype=int)
			frame_indices = frame_indices[np.unique(selected)]
		times = times_all[frame_indices]

		x_raw, y_raw = self.system.get_positions(states_all)
		x_all, y_all = self.system.grid.wrap_or_clip(x_raw, y_raw)
		line_x, line_y = x_all.copy(), y_all.copy()
		if self.system.grid.period is not None:
			crosses_boundary = (
				(np.abs(np.diff(x_all, axis=1)) > self.system.grid.period / 2)
				| (np.abs(np.diff(y_all, axis=1)) > self.system.grid.period / 2)
			)
			line_x[:, 1:][crosses_boundary] = np.nan
			line_y[:, 1:][crosses_boundary] = np.nan

		reference = indices[0]
		displacements: list[tuple[Array, Array]] = []
		polygon_x: Array | None = None
		polygon_y: Array | None = None
		if area_mode == 'element':
			for endpoint in indices[1:]:
				delta_x = x_raw[endpoint] - x_raw[reference]
				delta_y = y_raw[endpoint] - y_raw[reference]
				if self.system.grid.period is not None:
					period = self.system.grid.period
					delta_x -= period * np.round(delta_x / period)
					delta_y -= period * np.round(delta_y / period)
				displacements.append((delta_x, delta_y))
		else:
			polygon_x, polygon_y = self._unwrap_polygon_coordinates(
				x_raw[np.asarray(indices)],
				y_raw[np.asarray(indices)],
			)
			# Keep the local polygon next to the wrapped reference trajectory.
			polygon_x += x_all[reference] - polygon_x[0]
			polygon_y += y_all[reference] - polygon_y[0]

		X, Y = np.meshgrid(self.system.grid.x, self.system.grid.y, indexing='ij')
		psi_fields = [self.system.psi(t) for t in times]
		electric_fields = [self.system.electric_field(t, X, Y) for t in times]
		max_magnitude = max(float(np.nanmax(np.hypot(Ex, Ey))) for Ex, Ey in electric_fields)
		scale = None if np.isclose(max_magnitude, 0.0) else max_magnitude / (2 * min(self.system.grid.dx, self.system.grid.dy))
		norm = self._comparison_norm(*psi_fields)

		fig, (ax_field, ax_area) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
		mesh = ax_field.pcolormesh(
			self.system.grid.x,
			self.system.grid.y,
			psi_fields[0].T,
			shading='auto',
			cmap=cmap,
			norm=norm,
			**pcolormesh_kwargs,
		)
		Ex0, Ey0 = electric_fields[0]
		quiver = ax_field.quiver(
			X[::step, ::step],
			Y[::step, ::step],
			Ex0[::step, ::step],
			Ey0[::step, ::step],
			color='black',
			angles='xy',
			scale_units='xy',
			scale=scale,
			width=0.003,
		)
		trail_indices = indices if len(indices) <= 4 else ()
		colors = ('black', 'tab:orange', 'tab:green', 'tab:purple')
		labels = (
			('reference', r'$dX$ endpoint', r'$dY$ endpoint')
			if area_mode == 'element'
			else tuple(f'vertex {number + 1}' for number in range(len(trail_indices)))
		)
		trajectory_lines = [
			ax_field.plot([], [], color=color, lw=1.2, label=label)[0]
			for color, label in zip(colors, labels)
		]
		trajectory_markers = [
			ax_field.plot([], [], marker='o', color=color, markersize=4)[0]
			for color in colors[:len(trail_indices)]
		]
		displacement_arrows: list[FancyArrowPatch] = []
		if area_mode == 'element':
			# Keep dX and dY distinguishable even when their geometries overlap.
			for color, linestyle in zip(('tab:orange', 'tab:purple'), ('--', ':')):
				arrow = FancyArrowPatch(
					(0.0, 0.0),
					(0.0, 0.0),
					arrowstyle='-|>',
					mutation_scale=12,
					color=color,
					linestyle=linestyle,
					linewidth=2.0,
					zorder=6,
				)
				ax_field.add_patch(arrow)
				displacement_arrows.append(arrow)
		area_patch = ax_field.fill([], [], facecolor='lime', edgecolor='darkgreen', alpha=0.3)[0]
		boundary_markers = ax_field.plot(
			[], [], ls='none', marker='.', color='darkgreen', markersize=4,
		)[0]
		fig.colorbar(mesh, ax=ax_field, label=r'$\psi$')
		if trail_indices:
			ax_field.legend(loc='upper right')
		ax_field.set(xlabel='x', ylabel='y', aspect='equal')

		ax_area.axhline(0.0, color='0.5', ls='--', lw=1)
		area_line = ax_area.plot([], [], color='tab:blue', lw=1.8)[0]
		area_marker = ax_area.plot([], [], marker='o', color='tab:red', markersize=5)[0]
		area_text = ax_area.text(
			0.03,
			0.97,
			'',
			transform=ax_area.transAxes,
			va='top',
			bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8},
		)
		error_scale = float(np.max(np.abs(relative_error)))
		error_limit = 1.1 * error_scale if error_scale > 0 else 1e-12
		ax_area.set_xlim(times_all[0], times_all[-1])
		ax_area.set_ylim(-error_limit, error_limit)
		ax_area.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
		ax_area.set(
			xlabel='t',
			ylabel=r'$(A-A_0)/|A_0|$',
			title=r'Guiding-center area conservation',
		)

		def update(index: int) -> tuple[Any, ...]:
			current = int(frame_indices[index])
			Ex, Ey = electric_fields[index]
			mesh.set_array(psi_fields[index].T)
			quiver.set_UVC(Ex[::step, ::step], Ey[::step, ::step])
			for line, marker, trajectory_index in zip(trajectory_lines, trajectory_markers, trail_indices):
				line.set_data(
					line_x[trajectory_index, :current + 1],
					line_y[trajectory_index, :current + 1],
				)
				marker.set_data([x_all[trajectory_index, current]], [y_all[trajectory_index, current]])

			if area_mode == 'element':
				ref_x = x_all[reference, current]
				ref_y = y_all[reference, current]
				dx_x, dx_y = (component[current] for component in displacements[0])
				dy_x, dy_y = (component[current] for component in displacements[1])
				for arrow, (end_x, end_y) in zip(
					displacement_arrows,
					((ref_x + dx_x, ref_y + dx_y), (ref_x + dy_x, ref_y + dy_y)),
				):
					arrow.set_positions((ref_x, ref_y), (end_x, end_y))
				vertices = np.asarray([
					(ref_x, ref_y),
					(ref_x + dx_x, ref_y + dx_y),
					(ref_x + dx_x + dy_x, ref_y + dx_y + dy_y),
					(ref_x + dy_x, ref_y + dy_y),
					(ref_x, ref_y),
				])
			else:
				if polygon_x is None or polygon_y is None:
					raise RuntimeError('Polygon coordinates were not prepared.')
				vertices = np.column_stack((polygon_x[:, current], polygon_y[:, current]))
				vertices = np.vstack((vertices, vertices[0]))
			area_patch.set_xy(vertices)
			boundary_markers.set_data(vertices[:-1, 0], vertices[:-1, 1])

			area_line.set_data(times_all[:current + 1], relative_error[:current + 1])
			area_marker.set_data([times_all[current]], [relative_error[current]])
			area_text.set_text(
				rf'$A={area[current]:.6g}$' '\n'
				rf'$A/A_0={area[current] / initial_area:.6g}$'
			)
			ax_field.set_title(rf'$\psi$, $\mathbf{{E}}=-\nabla\psi$, $t={times_all[current]:.3f}$')
			return (
				mesh,
				quiver,
				*trajectory_lines,
				*trajectory_markers,
				*displacement_arrows,
				area_patch,
				boundary_markers,
				area_line,
				area_marker,
				area_text,
				ax_field.title,
			)

		update(0)
		animation = FuncAnimation(
			fig,
			update,
			frames=times.size,
			interval=interval,
			blit=False,
			repeat=repeat,
		)
		plt.close(fig)
		logger.info(
			"GC area-conservation animation ready: trajectories=%s frames=%d "
			"initial_area=%g max_relative_error=%g",
			indices,
			times.size,
			initial_area,
			error_scale,
		)
		return animation

__all__ = ['SystemResearch']
