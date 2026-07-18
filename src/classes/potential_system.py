#
# BSD 2-Clause License
#
# Copyright (c) 2023, Cristel Chandre
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import logging
import time
from typing import Any, Literal, Sequence

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pyhamsys import HamSys

from contracts import TrajectoryParams
from .grid import Grid
from .potential import Array, Potential, PotentialFields, PotentialInterpolators, real_imag

logger = logging.getLogger(__name__)

class PotentialSystem(HamSys):
	"""Particle dynamics over explicit physical and effective potentials."""

	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for turbulent potentials'

	@property
	def grid(self) -> Grid:
		"""Grid shared by the physical potential :math:`\\phi` and effective :math:`\\psi`."""
		return self.effective_potential.grid

	@property
	def fields(self) -> PotentialFields:
		"""Fields of the effective guiding-center potential :math:`\\psi`."""
		return self.effective_potential.fields

	@property
	def interpolators(self) -> PotentialInterpolators:
		"""Interpolators of the effective guiding-center potential :math:`\\psi`."""
		return self.effective_potential.interpolators

	@property
	def kinterp(self) -> int:
		"""Spline order of the effective guiding-center potential :math:`\\psi`."""
		return self.effective_potential.kinterp
		
	def __init__(self, potential: Potential, traj: TrajectoryParams) -> None:
		HamSys.__init__(self, ndof=1.5 if traj["type"]=='gc' else 2.5)
		self.traj = traj.copy()
		self.rho = self.traj.get("rho", 0)
		self.eta = self.traj.get("eta", 0)
		# Keep phi and psi as named, independent objects so their roles are explicit.
		self.physical_potential = Potential(
			grid=potential.grid,
			fields=potential.fields.copy(),
			k=potential.kinterp,
		)
		if min(potential.kinterp * potential.grid.dx, potential.kinterp * potential.grid.dy) < self.rho:
			raise ValueError(
				f"Interpolation order {potential.kinterp} is too low for rho = {self.rho}. "
				"Increase k or decrease rho."
			)
		self.effective_potential = (
			self.physical_potential.gyroaveraged(self.rho)
			if self.rho != 0
			else Potential(
				grid=potential.grid,
				fields=potential.fields.copy(),
				k=potential.kinterp,
			)
		)
		if self.traj["type"] == 'fo':
			self.v_fo = self.rho / (2 * np.abs(self.eta))
			self.phi_fo = np.sign(self.eta) / self.rho
			self.omlar = 1 / (2 * self.eta)

	def initial_conditions(
		self,
		n_traj: int,
		x: Array | None = None,
		y: Array | None = None,
		type: Literal['random', 'fixed'] = 'fixed',
	) -> Array:
		x, y = self.grid.x if x is None else x, self.grid.y if y is None else y
		if type == 'random':
			np.random.seed(int(time.time()))
			x0 = (x[-1] - x[0]) * np.random.rand(n_traj) + x[0]
			y0 = (y[-1] - y[0]) * np.random.rand(n_traj) + y[0]
			z0 = np.concatenate((x0, y0), axis=None)
		elif type == 'fixed':
			n_traj = int(np.sqrt(n_traj))**2
			x0 = np.linspace(x[0], x[-1], int(np.sqrt(n_traj)), endpoint=False)
			y0 = np.linspace(y[0], y[-1], int(np.sqrt(n_traj)), endpoint=False)
			x0, y0 = np.meshgrid(x0, y0, indexing='ij')
			z0 = np.concatenate((x0.flatten(), y0.flatten()), axis=None)
		else:
			raise ValueError("`type` must be either 'random' or 'fixed'.")
		if self.traj["type"] == 'fo':
			np.random.seed(int(time.time()))
			phi_perp = 2 * np.pi * np.random.rand(n_traj)
			z0 = np.concatenate((z0, np.cos(phi_perp), np.sin(phi_perp)), axis=None)
		return np.asarray(z0)
	
	def get_positions(self, z: Array) -> tuple[Array, Array]:
		x, y = np.split(z if self.traj["type"] == 'gc' else np.split(z, 2)[0], 2)
		return x, y
	
	def get_velocities(self, z: Array) -> tuple[Array, Array] | None:
		if self.traj["type"] == 'gc':
			return None
		vx, vy = np.split(np.split(z, 2)[1], 2)
		return vx, vy

	def psi(
		self,
		t: float,
		x: Array | None = None,
		y: Array | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> Array:
		"""Evaluate the effective guiding-center potential :math:`\\psi`.

		For now this is the leading-order gyroaveraged potential.  Keeping this
		separate name makes the GC equations explicit and leaves a single place
		to add the eta-dependent corrections later.
		"""
		return self.effective_potential.field_at_time(t, x, y, dx=dx, dy=dy, dt=dt)

	def phi(
		self,
		t: float,
		x: Array | None = None,
		y: Array | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> Array:
		"""Evaluate the physical, ungyroaveraged potential :math:`\\phi`."""
		return self.physical_potential.field_at_time(t, x, y, dx=dx, dy=dy, dt=dt)

	def field_at_time(
		self,
		t: float,
		x: Array | None = None,
		y: Array | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> Array:
		"""Evaluate :math:`\\psi`; retained as a convenience alias."""
		return self.psi(t, x, y, dx=dx, dy=dy, dt=dt)

	def electric_field(
		self,
		t: float,
		x: Array | None = None,
		y: Array | None = None,
		*,
		effective: bool = True,
	) -> tuple[Array, Array]:
		"""Evaluate :math:`\\mathbf{E}=-\\nabla\\psi` or :math:`-\\nabla\\phi` and log it."""
		if x is None and y is None:
			x, y = np.meshgrid(self.grid.x, self.grid.y, indexing='ij')
		elif x is None or y is None:
			raise ValueError("`x` and `y` must be provided together.")
		potential = self.psi if effective else self.phi
		field_name = "generalized (psi)" if effective else "physical (phi)"
		logger.debug(
			"Calculating %s electric field at t=%g (position shape=%s)",
			field_name,
			t,
			np.shape(x),
		)
		return -potential(t, x, y, dx=1), -potential(t, x, y, dy=1)

	@staticmethod
	def _comparison_norm(*fields: Array) -> mcolors.Normalize:
		"""Return one colour normalization shared by phi and psi plots."""
		vmin = min(float(np.nanmin(field)) for field in fields)
		vmax = max(float(np.nanmax(field)) for field in fields)
		if vmin < 0 < vmax:
			return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
		if np.isclose(vmin, vmax):
			delta = abs(vmin) * 0.01 or 1.0
			return mcolors.Normalize(vmin=vmin - delta, vmax=vmax + delta)
		return mcolors.Normalize(vmin=vmin, vmax=vmax)

	def plot_phi_psi(
		self,
		t: float = 0.0,
		*,
		contours: int | Sequence[float] | None = 12,
		cmap: str = 'RdBu_r',
		show: bool = True,
		**pcolormesh_kwargs: Any,
	) -> tuple[Figure, np.ndarray]:
		"""Plot the physical :math:`\\phi` (left) and effective :math:`\\psi` (right)."""
		phi_t, psi_t = self.phi(t), self.psi(t)
		norm = self._comparison_norm(phi_t, psi_t)
		fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
		for ax, field, label in zip(axes, (phi_t, psi_t), (r'$\phi$', r'$\psi$')):
			mesh = ax.pcolormesh(
				self.grid.x, self.grid.y, field.T, shading='auto', cmap=cmap, norm=norm, **pcolormesh_kwargs,
			)
			if contours is not None:
				ax.contour(self.grid.x, self.grid.y, field.T, levels=contours, colors='k', linewidths=0.45, alpha=0.55)
			ax.set(xlabel='x', ylabel='y', title=rf'{label}, $t={t:.3f}$', aspect='equal')
		fig.colorbar(mesh, ax=axes, label='potential')
		if show:
			plt.show()
		return fig, axes

	@staticmethod
	def _validate_step(step: int) -> None:
		if not isinstance(step, (int, np.integer)) or step < 1:
			raise ValueError('`step` must be a positive integer.')

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
		"""Plot :math:`\\phi` and :math:`\\psi` with their electric fields.

		``step`` selects every nth grid point for the arrow mesh; the potential
		background always remains at the full resolution.
		"""
		self._validate_step(step)
		phi_t, psi_t = self.phi(t), self.psi(t)
		X, Y = np.meshgrid(self.grid.x, self.grid.y, indexing='ij')
		fields = (
			self.electric_field(t, X, Y, effective=False),
			self.electric_field(t, X, Y),
		)
		max_magnitude = max(float(np.nanmax(np.hypot(Ex, Ey))) for Ex, Ey in fields)
		scale = None if np.isclose(max_magnitude, 0.0) else max_magnitude / (2 * min(self.grid.dx, self.grid.dy))
		norm = self._comparison_norm(phi_t, psi_t)
		fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
		for ax, potential, (Ex, Ey), symbol in zip(axes, (phi_t, psi_t), fields, (r'\phi', r'\psi')):
			mesh = ax.pcolormesh(
				self.grid.x, self.grid.y, potential.T, shading='auto', cmap=cmap, norm=norm, **pcolormesh_kwargs,
			)
			if contours is not None:
				ax.contour(self.grid.x, self.grid.y, potential.T, levels=contours, colors='k', linewidths=0.45, alpha=0.55)
			ax.quiver(
				X[::step, ::step], Y[::step, ::step], Ex[::step, ::step], Ey[::step, ::step],
				color='black', angles='xy', scale_units='xy', scale=scale, width=0.003,
			)
			ax.set(xlabel='x', ylabel='y', title=rf'${symbol}$ and $\mathbf{{E}}=-\nabla {symbol}$, $t={t:.3f}$', aspect='equal')
		fig.colorbar(mesh, ax=axes, label='potential')
		if show:
			plt.show()
		return fig, axes

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
		"""Animate :math:`\\phi` (left) and :math:`\\psi` (right) side by side."""
		if frames < 2:
			raise ValueError('`frames` must be at least 2.')
		if t_max <= 0:
			raise ValueError('`t_max` must be positive.')
		times = np.linspace(0.0, t_max, frames, endpoint=False)
		phi_fields = [self.phi(t) for t in times]
		psi_fields = [self.psi(t) for t in times]
		norm = self._comparison_norm(*phi_fields, *psi_fields)
		fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
		meshes = [
			ax.pcolormesh(self.grid.x, self.grid.y, field.T, shading='auto', cmap=cmap, norm=norm, **pcolormesh_kwargs)
			for ax, field in zip(axes, (phi_fields[0], psi_fields[0]))
		]
		for ax, label in zip(axes, (r'$\phi$', r'$\psi$')):
			ax.set(xlabel='x', ylabel='y', aspect='equal')
		fig.colorbar(meshes[0], ax=axes, label='potential')

		def update(index: int) -> tuple[Any, ...]:
			for mesh, field, ax, label in zip(meshes, (phi_fields[index], psi_fields[index]), axes, (r'$\phi$', r'$\psi$')):
				mesh.set_array(field.T)
				ax.set_title(rf'{label}, $t={times[index]:.3f}$')
			return *meshes, *(ax.title for ax in axes)

		update(0)
		animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False, repeat=repeat)
		plt.close(fig)
		return animation

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
		"""Animate :math:`\\phi`, :math:`\\psi`, and their electric-field arrows."""
		if frames < 2:
			raise ValueError('`frames` must be at least 2.')
		if t_max <= 0:
			raise ValueError('`t_max` must be positive.')
		self._validate_step(step)
		times = np.linspace(0.0, t_max, frames, endpoint=False)
		X, Y = np.meshgrid(self.grid.x, self.grid.y, indexing='ij')
		phi_fields, psi_fields = [self.phi(t) for t in times], [self.psi(t) for t in times]
		phi_electric = [self.electric_field(t, X, Y, effective=False) for t in times]
		psi_electric = [self.electric_field(t, X, Y) for t in times]
		max_magnitude = max(
			float(np.nanmax(np.hypot(Ex, Ey)))
			for fields in (phi_electric, psi_electric)
			for Ex, Ey in fields
		)
		scale = None if np.isclose(max_magnitude, 0.0) else max_magnitude / (2 * min(self.grid.dx, self.grid.dy))
		norm = self._comparison_norm(*phi_fields, *psi_fields)
		fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
		meshes = [
			ax.pcolormesh(self.grid.x, self.grid.y, field.T, shading='auto', cmap=cmap, norm=norm, **pcolormesh_kwargs)
			for ax, field in zip(axes, (phi_fields[0], psi_fields[0]))
		]
		quivers = [
			ax.quiver(X[::step, ::step], Y[::step, ::step], Ex[::step, ::step], Ey[::step, ::step],
				color='black', angles='xy', scale_units='xy', scale=scale, width=0.003)
			for ax, (Ex, Ey) in zip(axes, (phi_electric[0], psi_electric[0]))
		]
		for ax in axes:
			ax.set(xlabel='x', ylabel='y', aspect='equal')
		fig.colorbar(meshes[0], ax=axes, label='potential')

		def update(index: int) -> tuple[Any, ...]:
			for mesh, quiver, potential, (Ex, Ey), ax, symbol in zip(
				meshes, quivers, (phi_fields[index], psi_fields[index]),
				(phi_electric[index], psi_electric[index]), axes, (r'\phi', r'\psi'),
			):
				mesh.set_array(potential.T)
				quiver.set_UVC(Ex[::step, ::step], Ey[::step, ::step])
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
		"""Animate :math:`\\psi` with arrows for :math:`\\mathbf{E}=-\\nabla\\psi`."""
		if frames < 2:
			raise ValueError('`frames` must be at least 2.')
		if t_max <= 0:
			raise ValueError('`t_max` must be positive.')
		self._validate_step(step)
		times = np.linspace(0.0, t_max, frames, endpoint=False)
		X, Y = np.meshgrid(self.grid.x, self.grid.y, indexing='ij')
		psi_fields = [self.psi(t) for t in times]
		electric_fields = [self.electric_field(t, X, Y) for t in times]
		max_magnitude = max(float(np.nanmax(np.hypot(Ex, Ey))) for Ex, Ey in electric_fields)
		scale = None if np.isclose(max_magnitude, 0.0) else max_magnitude / (2 * min(self.grid.dx, self.grid.dy))
		norm = self._comparison_norm(*psi_fields)
		fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
		mesh = ax.pcolormesh(self.grid.x, self.grid.y, psi_fields[0].T, shading='auto', cmap=cmap, norm=norm, **pcolormesh_kwargs)
		Ex0, Ey0 = electric_fields[0]
		quiver = ax.quiver(
			X[::step, ::step], Y[::step, ::step], Ex0[::step, ::step], Ey0[::step, ::step],
			color='black', angles='xy', scale_units='xy', scale=scale, width=0.003,
		)
		fig.colorbar(mesh, ax=ax, label=r'$\psi$')
		ax.set(xlabel='x', ylabel='y', aspect='equal')

		def update(index: int) -> tuple[Any, ...]:
			Ex, Ey = electric_fields[index]
			mesh.set_array(psi_fields[index].T)
			quiver.set_UVC(Ex[::step, ::step], Ey[::step, ::step])
			ax.set_title(rf'$\psi$ and $\mathbf{{E}}=-\nabla \psi$, $t={times[index]:.3f}$')
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
		"""Animate :math:`\\psi`, its electric field, and one or more trajectories.

		``solution`` must expose the ``t`` and ``y`` arrays returned by the
		integrator. ``frame_stride`` keeps one stored frame every n samples while
		preserving the first and last times. Trajectories are wrapped onto the
		plotted periodic domain.
		"""
		self._validate_step(step)
		times_all = np.asarray(solution.t, dtype=float)
		states_all = np.asarray(solution.y)
		if times_all.ndim != 1 or states_all.ndim != 2 or states_all.shape[1] != times_all.size:
			raise ValueError('`solution` must provide t with shape (n_times,) and y with shape (n_state, n_times).')
		if times_all.size < 2:
			raise ValueError('`solution` must contain at least two time samples.')
		if frames is not None and frames < 2:
			raise ValueError('`frames` must be at least 2.')
		if not isinstance(frame_stride, (int, np.integer)) or frame_stride < 1:
			raise ValueError('`frame_stride` must be a positive integer.')
		logger.info(
			"Preparing trajectory animation: solution_shape=%s samples=%d time_range=[%g, %g]",
			states_all.shape,
			times_all.size,
			times_all[0],
			times_all[-1],
		)
		frame_indices = np.arange(0, times_all.size, frame_stride, dtype=int)
		if frame_indices[-1] != times_all.size - 1:
			frame_indices = np.append(frame_indices, times_all.size - 1)
		if frames is not None and frames < frame_indices.size:
			selected = np.linspace(0, frame_indices.size - 1, frames, dtype=int)
			frame_indices = frame_indices[np.unique(selected)]
		times = times_all[frame_indices]
		x_all, y_all = self.get_positions(states_all)
		x_all, y_all = self.grid.wrap_or_clip(x_all, y_all)
		logger.info(
			"Animation data selected: trajectories=%d frames=%d stride=%d "
			"x_range=[%g, %g] y_range=[%g, %g]",
			x_all.shape[0],
			times.size,
			frame_stride,
			float(np.nanmin(x_all)),
			float(np.nanmax(x_all)),
			float(np.nanmin(y_all)),
			float(np.nanmax(y_all)),
		)
		line_x, line_y = x_all.copy(), y_all.copy()
		if self.grid.period is not None:
			crosses_boundary = (
				(np.abs(np.diff(x_all, axis=1)) > self.grid.period / 2)
				| (np.abs(np.diff(y_all, axis=1)) > self.grid.period / 2)
			)
			# A NaN breaks a Matplotlib line.  It avoids drawing a false segment
			# across the periodic square when a trajectory re-enters on the other side.
			line_x[:, 1:][crosses_boundary] = np.nan
			line_y[:, 1:][crosses_boundary] = np.nan
		X, Y = np.meshgrid(self.grid.x, self.grid.y, indexing='ij')
		logger.info(
			"Field source used by the animation: psi is evaluated from PotentialSystem.psi(t) "
			"(the effective/gyroaveraged potential stored in the system), not from solution.y; "
			"E=(Ex, Ey) is calculated by PotentialSystem.electric_field(t, X, Y) as "
			"(-dpsi/dx, -dpsi/dy). Evaluations use solution.t[frame_indices] on the system "
			"grid X,Y with shape=%s, x_range=[%g, %g], y_range=[%g, %g]; "
			"the quiver displays every %d grid points",
			X.shape,
			self.grid.xmin,
			self.grid.xmax,
			self.grid.ymin,
			self.grid.ymax,
			step,
		)
		psi_fields = [self.psi(t) for t in times]
		electric_fields = [self.electric_field(t, X, Y) for t in times]
		n_trajectories = x_all.shape[0]
		x_rows = f"0:{n_trajectories}"
		y_rows = f"{n_trajectories}:{2 * n_trajectories}"
		logger.info(
			"Solution values used by the animation: frame_indices=%s; solution.t[frame_indices]=%s; "
			"solution.y[%s, frame_indices] -> x=%s; solution.y[%s, frame_indices] -> y=%s",
			np.array2string(frame_indices),
			np.array2string(times, precision=6),
			x_rows,
			np.array2string(states_all[:n_trajectories, frame_indices], precision=6),
			y_rows,
			np.array2string(states_all[n_trajectories:2 * n_trajectories, frame_indices], precision=6),
		)
		max_magnitude = max(float(np.nanmax(np.hypot(Ex, Ey))) for Ex, Ey in electric_fields)
		scale = None if np.isclose(max_magnitude, 0.0) else max_magnitude / (2 * min(self.grid.dx, self.grid.dy))
		norm = self._comparison_norm(*psi_fields)
		fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
		mesh = ax.pcolormesh(self.grid.x, self.grid.y, psi_fields[0].T, shading='auto', cmap=cmap, norm=norm, **pcolormesh_kwargs)
		Ex0, Ey0 = electric_fields[0]
		quiver = ax.quiver(
			X[::step, ::step], Y[::step, ::step], Ex0[::step, ::step], Ey0[::step, ::step],
			color='black', angles='xy', scale_units='xy', scale=scale, width=0.003,
		)
		lines = [ax.plot([], [], color='green', lw=1.5, label=f'trajectory {index + 1}')[0] for index in range(x_all.shape[0])]
		markers = [ax.plot([], [], marker='o', color='green', markersize=4)[0] for _ in range(x_all.shape[0])]
		fig.colorbar(mesh, ax=ax, label=r'$\psi$')
		if x_all.shape[0] > 1:
			ax.legend(loc='upper right')
		ax.set(xlabel='x', ylabel='y', aspect='equal')

		def update(index: int) -> tuple[Any, ...]:
			current = frame_indices[index]
			Ex, Ey = electric_fields[index]
			mesh.set_array(psi_fields[index].T)
			quiver.set_UVC(Ex[::step, ::step], Ey[::step, ::step])
			for trajectory, marker, x_path, y_path, display_x, display_y in zip(lines, markers, x_all, y_all, line_x, line_y):
				trajectory.set_data(display_x[:current + 1], display_y[:current + 1])
				marker.set_data([x_path[current]], [y_path[current]])
			ax.set_title(rf'$\psi$, $\mathbf{{E}}=-\nabla \psi$, $t={times[index]:.3f}$')
			return mesh, quiver, *lines, *markers, ax.title

		update(0)
		animation = FuncAnimation(fig, update, frames=times.size, interval=interval, blit=False, repeat=repeat)
		plt.close(fig)
		logger.info(
			"Trajectory animation ready in memory: frames=%d interval_ms=%d repeat=%s; "
			"render or save the returned FuncAnimation to produce the output",
			times.size,
			interval,
			repeat,
		)
		return animation


	def plot_psi(
		self,
		t: float = 0.0,
		*,
		contours: int | Sequence[float] | None = 12,
		cmap: str = 'RdBu_r',
		show: bool = True,
		**pcolormesh_kwargs: Any,
	) -> tuple[Figure, Axes]:
		"""Plot the complete effective potential :math:`\\psi` at time ``t``."""
		psi_t = self.psi(t)
		vmin, vmax = float(np.nanmin(psi_t)), float(np.nanmax(psi_t))
		if vmin < 0 < vmax:
			norm: mcolors.Normalize = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
		elif np.isclose(vmin, vmax):
			delta = abs(vmin) * 0.01 or 1.0
			norm = mcolors.Normalize(vmin=vmin - delta, vmax=vmax + delta)
		else:
			norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
		fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
		mesh = ax.pcolormesh(
			self.grid.x,
			self.grid.y,
			psi_t.T,
			shading='auto',
			cmap=cmap,
			norm=norm,
			**pcolormesh_kwargs,
		)
		if contours is not None:
			ax.contour(self.grid.x, self.grid.y, psi_t.T, levels=contours, colors='k', linewidths=0.45, alpha=0.55)
		fig.colorbar(mesh, ax=ax, label=r'$\psi$')
		ax.set(xlabel='x', ylabel='y', title=rf'Effective potential $\psi$, $t={t:.3f}$', aspect='equal')
		if show:
			plt.show()
		return fig, ax

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
		"""Animate the complete effective potential :math:`\\psi`."""
		return self.effective_potential.animate(
			t_max=t_max,
			frames=frames,
			interval=interval,
			cmap=cmap,
			repeat=repeat,
			title=r'Effective potential $\psi$',
			**pcolormesh_kwargs,
		)

	def hamiltonian(self, t: float | Array, z: Array) -> Array:
		x, y = self.get_positions(z)
		if self.traj["type"] == 'gc':
			return np.asarray(self.psi(t, x, y))
		else:
			phi_t = self.phi(t, x, y)
			velocities = self.get_velocities(z)
			if velocities is None:
				raise RuntimeError("Full-orbit velocities are unavailable.")
			vx, vy = velocities
			return np.asarray(
				self.rho / (4 * np.abs(self.eta)) * (vx**2 + vy**2)
				+ phi_t * np.sign(self.eta) / self.rho
			)

	def y_dot(self, t: float, z: Array, output: Literal['full', 'reduced'] = 'full') -> Array:
		x, y = self.get_positions(z)
		if self.traj["type"] == 'gc':
			ex_t, ey_t = self.electric_field(t, x, y)
			return np.concatenate((ey_t, -ex_t), axis=None)
		ex_t, ey_t = self.electric_field(t, x, y, effective=False)
		if output == 'reduced':
			return np.concatenate((ey_t, -ex_t), axis=None)
		else:
			velocities = self.get_velocities(z)
			if velocities is None:
				raise RuntimeError("Full-orbit velocities are unavailable.")
			vx, vy = velocities
			return np.concatenate((vx * self.v_fo, vy * self.v_fo, ex_t * self.phi_fo\
						   + vy * self.omlar, ey_t * self.phi_fo - vx * self.omlar), axis=None)

	def y_dot_lyap(self, t: float, z: Array) -> Array:
		if self.traj["type"] == 'fo':
			x, y, vx, vy, *jacobian_parts = np.split(z, 20)
			z = np.concatenate((x, y, vx, vy), axis=None)
			jacobian = np.array(jacobian_parts).reshape((4, 4, -1))
		else:
			x, y, *jacobian_parts = np.split(z, 6)
			z = np.concatenate((x, y), axis=None)
			jacobian = np.array(jacobian_parts).reshape((2, 2, -1))
		z_dot = self.y_dot(t, z)
		if self.traj["type"] == 'gc':
			d2psidx2_t = self.psi(t, x, y, dx=2)
			d2psidxdy_t = self.psi(t, x, y, dx=1, dy=1)
			d2psidy2_t = self.psi(t, x, y, dy=2)
		else:
			d2phidx2_t = self.phi(t, x, y, dx=2)
			d2phidxdy_t = self.phi(t, x, y, dx=1, dy=1)
			d2phidy2_t = self.phi(t, x, y, dy=2)
		A = np.zeros_like(jacobian)
		if self.traj["type"] == 'fo':
			d2phidx2_t *= -self.phi_fo
			d2phidxdy_t *= -self.phi_fo
			d2phidy2_t *= -self.phi_fo
			A[0, 2, :], A[1, 3, :] = self.v_fo * np.ones_like(x), self.v_fo * np.ones_like(x)
			A[2, 3, :], A[3, 2, :] = self.omlar * np.ones_like(x), -self.omlar * np.ones_like(x)
			A[2, 0, :], A[2, 1, :] = d2phidx2_t, d2phidxdy_t
			A[3, 0, :], A[3, 1, :] = d2phidxdy_t, d2phidy2_t
		if self.traj["type"] == 'gc':
			A[0, 0, :], A[0, 1, :] = -d2psidxdy_t, -d2psidy2_t
			A[1, 0, :], A[1, 1, :] = d2psidx2_t, d2psidxdy_t
		J_dot = np.einsum('ijm,jkm->ikm', A, jacobian)
		return np.concatenate((z_dot, J_dot.reshape(-1)), axis=None)

	def k_dot(self, t: float, z: Array) -> float | Array:
		x, y = self.get_positions(z)
		if self.traj["type"] == 'gc':
			return -float(np.sum(self.psi(t, x, y, dt=1)))
		dphidt_t = -float(np.sum(self.phi(t, x, y, dt=1)))
		if self.traj["type"] == 'fo':
			dphidt_t *= -self.phi_fo
		return dphidt_t

	def chi(self, h: float, t: float, z: Array) -> Array:
		x, y, vx, vy = np.split(z, 4)
		exp_ = np.exp(-1j * h * self.omlar)
		x, y = real_imag(x + 1j * y + 1j * self.rho * np.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		pot = np.split(self.y_dot(t, np.concatenate((x, y), axis=None), output='reduced'), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * self.phi_fo)
		return np.concatenate((x, y, vx, vy), axis=None)
	
	def chi_star(self, h: float, t: float, z: Array) -> Array:
		x, y, vx, vy = np.split(z, 4)
		pot = np.split(self.y_dot(t, np.concatenate((x, y), axis=None), output='reduced'), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * self.phi_fo)
		exp_ = np.exp(-1j * h * self.omlar)
		x, y = real_imag(x + 1j * y + 1j * self.rho * np.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		return np.concatenate((x, y, vx, vy), axis=None)
	
	def fo2gc(self, z: Array) -> tuple[Array, Array]:
		x, y, vx, vy = np.split(z, 4)
		v = vy + 1j * vx
		theta, rho = np.pi + np.angle(v), self.rho * np.abs(v)
		return x - rho * np.cos(theta), y + rho * np.sin(theta)
