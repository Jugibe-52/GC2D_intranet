# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Electrostatic potential used by the development notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.fft import fft2, fftfreq, ifft2
from scipy.interpolate import RectBivariateSpline
from scipy.special import jv

from .grid import Grid


@dataclass(frozen=True, slots=True)
class _SplineDomain:
	x: np.ndarray
	y: np.ndarray
	padding: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _Spline:
	real: RectBivariateSpline
	imag: RectBivariateSpline

	def evaluate(
		self,
		x: np.ndarray,
		y: np.ndarray,
		*,
		dx: int = 0,
		dy: int = 0,
	) -> np.ndarray:
		return np.asarray(
			self.real.ev(x, y, dx=dx, dy=dy)
			+ 1j * self.imag.ev(x, y, dx=dx, dy=dy)
		)


def _spline_domain(grid: Grid, interpolation_order: int) -> _SplineDomain:
	"""Extend the grid so splines remain valid at both boundaries."""
	margin = interpolation_order + 1
	padding = (margin, margin + 1)
	left, right = padding
	x = np.pad(
		grid.x,
		padding,
		mode="linear_ramp",
		end_values=(grid.xmin - left * grid.dx, grid.xmax + right * grid.dx),
	)
	y = np.pad(
		grid.y,
		padding,
		mode="linear_ramp",
		end_values=(grid.ymin - left * grid.dy, grid.ymax + right * grid.dy),
	)
	return _SplineDomain(np.asarray(x), np.asarray(y), padding)


def _build_spline(
	grid: Grid,
	coefficient: np.ndarray,
	interpolation_order: int,
) -> _Spline:
	domain = _spline_domain(grid, interpolation_order)
	padded = np.pad(
		coefficient,
		(domain.padding, domain.padding),
		mode="wrap",
	)
	return _Spline(
		RectBivariateSpline(
			domain.x,
			domain.y,
			padded.real,
			kx=interpolation_order,
			ky=interpolation_order,
		),
		RectBivariateSpline(
			domain.x,
			domain.y,
			padded.imag,
			kx=interpolation_order,
			ky=interpolation_order,
		),
	)


def _colour_scale(field: np.ndarray) -> mcolors.Normalize:
	vmin = float(np.nanmin(field))
	vmax = float(np.nanmax(field))
	if vmin < 0 < vmax:
		return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
	if np.isclose(vmin, vmax):
		delta = abs(vmin) * 0.01 or 1.0
		return mcolors.Normalize(vmin=vmin - delta, vmax=vmax + delta)
	return mcolors.Normalize(vmin=vmin, vmax=vmax)


class Potential:
	"""One time-dependent mode interpolated on a regular spatial grid.

	The physical field is reconstructed as
	``2 * real(coefficient * exp(-1j * t))``.  The development notebooks create
	this representation with :meth:`random`.
	"""

	def __init__(
		self,
		grid: Grid,
		coefficient: np.ndarray,
		*,
		interpolation_order: int = 3,
	) -> None:
		if not isinstance(grid, Grid):
			raise TypeError("`grid` must be a Grid instance.")
		field = np.asarray(coefficient, dtype=np.complex128)
		if field.shape != grid.shape:
			raise ValueError(
				f"The coefficient shape is {field.shape}; expected {grid.shape}."
			)
		if not np.all(np.isfinite(field)):
			raise ValueError("The potential coefficient must contain finite values.")
		if (
			isinstance(interpolation_order, (bool, np.bool_))
			or not isinstance(interpolation_order, (int, np.integer))
			or not 2 <= int(interpolation_order) <= 5
		):
			raise ValueError("`interpolation_order` must be an integer from 2 to 5.")
		self.grid = grid
		self.interpolation_order = int(interpolation_order)
		self._coefficient = field.copy()
		self._spline = _build_spline(
			self.grid,
			self._coefficient,
			self.interpolation_order,
		)

	@classmethod
	def random(
		cls,
		*,
		A: float,
		M: int,
		nx: int,
		ny: int,
		seed: int = 27,
		interpolation_order: int = 3,
	) -> Potential:
		"""Create the reproducible periodic potential used in the notebooks.

		``A`` controls the amplitude and ``M`` the maximum spatial wave number.
		"""
		A = float(A)
		if not np.isfinite(A) or A < 0:
			raise ValueError("`A` must be a finite, non-negative number.")
		if isinstance(M, (bool, np.bool_)) or not isinstance(M, (int, np.integer)) or M < 1:
			raise ValueError("`M` must be a positive integer.")
		if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
			raise TypeError("`seed` must be an integer.")

		grid = Grid.periodic(nx, ny)
		wave_x, wave_y = np.meshgrid(
			np.arange(M + 1),
			np.arange(M + 1),
			indexing="ij",
		)
		spectrum = np.zeros((M + 1, M + 1), dtype=np.complex128)
		phases = 2 * np.pi * np.random.default_rng(int(seed)).random((M, M))
		spectrum[1:, 1:] = (
			A
			/ (wave_x[1:, 1:] ** 2 + wave_y[1:, 1:] ** 2) ** 1.5
			* np.exp(1j * phases)
		)
		spectrum[np.hypot(wave_x, wave_y) > M] = 0

		mode_x, mode_y = np.indices(spectrum.shape)
		x_mesh, y_mesh = np.meshgrid(grid.x, grid.y, indexing="ij")
		phase = np.exp(
			1j
			* (
				mode_x[:, :, None, None] * x_mesh[None, None, :, :]
				+ mode_y[:, :, None, None] * y_mesh[None, None, :, :]
			)
		)
		coefficient = np.asarray(np.einsum("nm,nm...->...", spectrum, phase))
		return cls(
			grid,
			coefficient,
			interpolation_order=interpolation_order,
		)

	def evaluate(
		self,
		t: float | np.ndarray,
		x: np.ndarray | None = None,
		y: np.ndarray | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> np.ndarray:
		"""Evaluate the potential or one of its derivatives.

		Coordinates are paired: ``x[i]`` is evaluated with ``y[i]``.  Omitting
		them evaluates the field on the complete stored grid.
		"""
		if (x is None) != (y is None):
			raise ValueError("`x` and `y` must be provided together.")
		if dt not in (0, 1):
			raise ValueError("`dt` must be 0 or 1.")
		for derivative, name in ((dx, "dx"), (dy, "dy")):
			if (
				isinstance(derivative, (bool, np.bool_))
				or not isinstance(derivative, (int, np.integer))
				or derivative < 0
			):
				raise ValueError(f"`{name}` must be a non-negative integer.")

		time = np.asarray(t)
		if x is None:
			if dx or dy:
				raise ValueError("Spatial derivatives require `x` and `y`.")
			coefficient = self._coefficient
			if time.ndim:
				coefficient = coefficient.reshape(
					coefficient.shape + (1,) * time.ndim
				)
		else:
			assert y is not None
			x_values, y_values = self.grid.normalize(np.asarray(x), np.asarray(y))
			coefficient = self._spline.evaluate(
				x_values,
				y_values,
				dx=int(dx),
				dy=int(dy),
			)

		phase = np.exp(-1j * time)
		if dt == 1:
			phase = phase * -1j
		return np.asarray(2.0 * np.real(coefficient * phase), dtype=float)

	def electric_field(
		self,
		t: float | np.ndarray,
		x: np.ndarray | None = None,
		y: np.ndarray | None = None,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Return the electric field ``-gradient(potential)``."""
		if x is None and y is None:
			x, y = np.meshgrid(self.grid.x, self.grid.y, indexing="ij")
		elif x is None or y is None:
			raise ValueError("`x` and `y` must be provided together.")
		return (
			-self.evaluate(t, x, y, dx=1),
			-self.evaluate(t, x, y, dy=1),
		)

	def gyroaverage(self, rho: float) -> Potential:
		"""Return the Larmor-circle average at radius ``rho``."""
		rho = float(rho)
		if not np.isfinite(rho) or rho < 0:
			raise ValueError("`rho` must be finite and non-negative.")
		kx = fftfreq(self.grid.nx, d=self.grid.dx)
		ky = fftfreq(self.grid.ny, d=self.grid.dy)
		kx_mesh, ky_mesh = np.meshgrid(kx, ky, indexing="ij")
		factor = jv(0, 2 * np.pi * rho * np.hypot(kx_mesh, ky_mesh))
		coefficient = ifft2(fft2(self._coefficient) * factor)
		return Potential(
			self.grid,
			coefficient,
			interpolation_order=self.interpolation_order,
		)

	def plot(
		self,
		*,
		t: float = 0.0,
		contours: int | Sequence[float] | None = 12,
		cmap: str = "RdBu_r",
		show: bool = True,
		**pcolormesh_kwargs: Any,
	) -> tuple[Figure, Axes]:
		"""Plot the potential at one time."""
		field = self.evaluate(t)
		fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
		mesh = ax.pcolormesh(
			self.grid.x,
			self.grid.y,
			field.T,
			shading="auto",
			cmap=cmap,
			norm=_colour_scale(field),
			**pcolormesh_kwargs,
		)
		if contours is not None:
			ax.contour(
				self.grid.x,
				self.grid.y,
				field.T,
				levels=contours,
				colors="black",
				linewidths=0.45,
				alpha=0.55,
			)
		fig.colorbar(mesh, ax=ax, label=r"$\phi$")
		ax.set(xlabel="x", ylabel="y", title=rf"Potential, $t={t:.3f}$", aspect="equal")
		if show:
			plt.show()
		return fig, ax

	def animate(
		self,
		*,
		t_max: float = 2 * np.pi,
		frames: int = 120,
		interval: int = 50,
		cmap: str = "RdBu_r",
		repeat: bool = True,
		**pcolormesh_kwargs: Any,
	) -> FuncAnimation:
		"""Animate the potential from ``t=0`` to ``t=t_max``."""
		if not isinstance(frames, int) or isinstance(frames, bool) or frames < 2:
			raise ValueError("`frames` must be an integer of at least 2.")
		t_max = float(t_max)
		if not np.isfinite(t_max) or t_max <= 0:
			raise ValueError("`t_max` must be positive and finite.")

		times = np.linspace(0.0, t_max, frames, endpoint=False)
		fields = [self.evaluate(time) for time in times]
		vmin = min(float(np.min(field)) for field in fields)
		vmax = max(float(np.max(field)) for field in fields)
		if vmin < 0 < vmax:
			norm: mcolors.Normalize = mcolors.TwoSlopeNorm(
				vmin=vmin,
				vcenter=0.0,
				vmax=vmax,
			)
		else:
			norm = _colour_scale(np.asarray([vmin, vmax]))

		fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
		mesh = ax.pcolormesh(
			self.grid.x,
			self.grid.y,
			fields[0].T,
			shading="auto",
			cmap=cmap,
			norm=norm,
			**pcolormesh_kwargs,
		)
		fig.colorbar(mesh, ax=ax, label=r"$\phi$")
		ax.set(xlabel="x", ylabel="y", aspect="equal")

		def update(index: int) -> tuple[Any, ...]:
			mesh.set_array(fields[index].T)
			ax.set_title(rf"Potential, $t={times[index]:.3f}$")
			return mesh, ax.title

		update(0)
		animation = FuncAnimation(
			fig,
			update,
			frames=frames,
			interval=interval,
			blit=False,
			repeat=repeat,
		)
		plt.close(fig)
		return animation


__all__ = ["Potential"]
