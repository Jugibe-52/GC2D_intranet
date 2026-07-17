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

import logging
import os
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from typing import Any, Sequence, TypeAlias
os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import RectBivariateSpline
from scipy.special import jv

from contracts import Array

logger = logging.getLogger(__name__)

FieldList: TypeAlias = tuple[Array | None, list[Array] | None]
FieldInput: TypeAlias = Sequence[Array | Sequence[Array] | None]
ComplexInterpolator: TypeAlias = tuple[RectBivariateSpline, RectBivariateSpline]
InterpolatorList: TypeAlias = tuple[RectBivariateSpline | None, list[ComplexInterpolator] | None]


def real_imag(z: Array) -> tuple[Array, Array]:
	return z.real, z.imag


def normalize_fields(fields: FieldInput) -> FieldList:
	"""Validate and normalize the heterogeneous public field input."""
	if len(fields) != 2:
		raise ValueError("`fields` must contain [mean_value, fluctuations].")
	mean_value, raw_fluctuations = fields
	if mean_value is not None and not isinstance(mean_value, np.ndarray):
		raise TypeError("The mean field must be a NumPy array or None.")
	if raw_fluctuations is None:
		fluctuations = None
	elif isinstance(raw_fluctuations, np.ndarray):
		fluctuations = [np.asarray(field) for field in raw_fluctuations]
	else:
		fluctuations = [np.asarray(field) for field in raw_fluctuations]
	return mean_value, fluctuations


def _interpolation_grid(
	x: Array,
	y: Array,
	*,
	kinterp: int,
	xy_period: float | None,
) -> tuple[Array, Array, tuple[int, int]]:
	"""Extend a grid with the padding required by ``RectBivariateSpline``."""
	kl = kinterp + 1
	kr = kinterp + 2 if xy_period is not None else kinterp + 1
	dx, dy = x[1] - x[0], y[1] - y[0]
	xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
	x_ = np.pad(x, (kl, kr), mode='linear_ramp', end_values=(xmin - kl * dx, xmax + kr * dx))
	y_ = np.pad(y, (kl, kr), mode='linear_ramp', end_values=(ymin - kl * dy, ymax + kr * dy))
	return x_, y_, (kl, kr)


def _pad_field(field: Array, padding: tuple[int, int], *, xy_period: float | None) -> Array:
	"""Pad a field periodically or with zeros according to the boundary policy."""
	kwargs = {'mode': 'wrap'} if xy_period is not None else {'mode': 'constant', 'constant_values': 0}
	return np.pad(field, (padding, padding), **kwargs)


def _build_interpolators(
	x: Array,
	y: Array,
	fields: FieldList,
	*,
	kinterp: int,
	xy_period: float | None,
) -> InterpolatorList:
	"""Build spline interpolators for a field set and its boundary policy."""
	x_, y_, padding = _interpolation_grid(x, y, kinterp=kinterp, xy_period=xy_period)
	mean_value, fluctuations = None, None
	if fields[0] is not None:
		padded_mean = _pad_field(fields[0], padding, xy_period=xy_period)
		mean_value = RectBivariateSpline(x_, y_, padded_mean, kx=kinterp, ky=kinterp)
	if fields[1] is not None:
		padded_fluctuations = [_pad_field(field, padding, xy_period=xy_period) for field in fields[1]]
		fluctuations = []
		for field in padded_fluctuations:
			interp_real = RectBivariateSpline(x_, y_, field.real, kx=kinterp, ky=kinterp)
			interp_imag = RectBivariateSpline(x_, y_, field.imag, kx=kinterp, ky=kinterp)
			fluctuations.append((interp_real, interp_imag))
	return mean_value, fluctuations


def _resample_fields(xi: Array, yi: Array, interpolators: InterpolatorList) -> FieldList:
	"""Evaluate a field set's interpolators on a new grid."""
	mean_value, fluctuations = None, None
	if interpolators[0] is not None:
		mean_value = np.asarray(interpolators[0](xi, yi))
	if interpolators[1] is not None:
		fluctuations = [
			np.asarray(interp_real(xi, yi) + 1j * interp_imag(xi, yi))
			for interp_real, interp_imag in interpolators[1]
		]
	return mean_value, fluctuations


class Potential:
	def __init__(
		self,
		x: Array,
		y: Array,
		fields: FieldInput,
		freqs: Sequence[float] | Array,
		nx: int | None = None,
		ny: int | None = None,
		xy_period: float | None = None,
		k: int = 3,
	) -> None:
		self.freqs: Array = np.asarray(np.atleast_1d(freqs), dtype=float)
		if x.ndim != 1:
			raise ValueError("`x` must be 1-dimensional.")
		if y.ndim != 1:
			raise ValueError("`y` must be 1-dimensional.")
		diff_x: Array = np.diff(x)
		diff_y: Array = np.diff(y)
		if np.any(diff_x <= 0) or np.any(diff_y <= 0):
			raise ValueError("Values in `x` or `y` are not properly sorted.")
		if not np.allclose(diff_x, diff_x[0]) or not np.allclose(diff_y, diff_y[0]):
			raise ValueError("Values in `x` or `y` are not uniformly spaced.")
		fields_: FieldList = normalize_fields(fields)
		expected_shape: tuple[int, int] = (x.size, y.size)
		if fields_[0] is not None and fields_[0].shape != expected_shape:
			raise ValueError(f"Mean field has shape {fields_[0].shape}, expected {expected_shape}.")
		if fields_[1] is not None:
			for index, field in enumerate(fields_[1]):
				if field.shape != expected_shape:
					raise ValueError(f"Fluctuation {index} has shape {field.shape}, expected {expected_shape}.")
			if len(fields_[1]) != self.freqs.size:
				raise ValueError(
					f"Received {len(fields_[1])} fluctuations for {self.freqs.size} frequencies."
				)
		self.xy_period: float | None = xy_period
		self.kinterp: int = k
		if not 1 <= k <= 5:
			raise ValueError("`k` must be between 1 and 5 for RectBivariateSpline.")
		xi: Array
		yi: Array
		processed_fields: FieldList
		if nx is not None or ny is not None:
			target_nx: int = x.size if nx is None else nx
			target_ny: int = y.size if ny is None else ny
			if target_nx < 2 or target_ny < 2:
				raise ValueError("`nx` and `ny` must be at least 2.")
			xi = np.linspace(x.min(), x.max(), target_nx)
			yi = np.linspace(y.min(), y.max(), target_ny)
			interpolators: InterpolatorList = _build_interpolators(
				x,
				y,
				fields_,
				kinterp=self.kinterp,
				xy_period=self.xy_period,
			)
			processed_fields = _resample_fields(xi, yi, interpolators)
		else:
			xi, yi, processed_fields = x, y, fields_
		self.x: Array = xi
		self.y: Array = yi
		self.fields: FieldList = processed_fields
		self.dx: float = float(self.x[1] - self.x[0])
		self.dy: float = float(self.y[1] - self.y[0])
		self.xmin: float = float(self.x.min())
		self.xmax: float = float(self.x.max())
		self.ymin: float = float(self.y.min())
		self.ymax: float = float(self.y.max())
		self.nx: int = self.x.size
		self.ny: int = self.y.size
		self.interpolators: InterpolatorList = _build_interpolators(
			self.x,
			self.y,
			self.fields,
			kinterp=self.kinterp,
			xy_period=self.xy_period,
		)

	def __str__(self) -> str:
		"""Return a compact summary suitable for notebooks and logs."""
		mean_value, fluctuations = self.fields
		frequencies = np.array2string(self.freqs, precision=5, threshold=8, edgeitems=3)
		boundary = f'periodic (period={self.xy_period:g})' if self.xy_period is not None else 'zero-padded'
		return (
			'Potential(\n'
			f'  grid={self.nx} x {self.ny}, dx={self.dx:g}, dy={self.dy:g},\n'
			f'  x=[{self.xmin:g}, {self.xmax:g}], y=[{self.ymin:g}, {self.ymax:g}],\n'
			f'  mean_field={mean_value is not None}, modes={0 if fluctuations is None else len(fluctuations)},\n'
			f'  freqs={frequencies}, interpolation_order={self.kinterp}, boundary={boundary}\n'
			')'
		)

	def gyroaverage(self, rho: float, fields: FieldList) -> FieldList:
		kx, ky = fftfreq(self.nx, d=self.dx), fftfreq(self.ny, d=self.dy)
		kx_, ky_ = np.meshgrid(kx, ky, indexing='ij')
		mean_value, fluctuations = None, None
		if fields[0] is not None:
			mean_value = ifft2(fft2(fields[0]) * jv(0, 2 * np.pi * rho * np.sqrt(kx_**2 + ky_**2))).real
		if fields[1] is not None:
			fluctuations = []
			for field in fields[1]:
				gyro_field = ifft2(fft2(field) * jv(0, 2 * np.pi * rho * np.sqrt(kx_**2 + ky_**2))) 
				fluctuations.append(gyro_field)
		return mean_value, fluctuations

	def wrap_or_clip(self, xi: Array, yi: Array) -> tuple[Array, Array]:
		"""Apply this potential's boundary policy to evaluation coordinates."""
		if self.xy_period is None:
			xi = np.clip(xi, self.xmin, self.xmax)
			yi = np.clip(yi, self.ymin, self.ymax)
		else:
			xi = ((np.asarray(xi) - self.xmin) % self.xy_period) + self.xmin
			yi = ((np.asarray(yi) - self.ymin) % self.xy_period) + self.ymin
		return np.asarray(xi), np.asarray(yi)

	def phic_interp(self, xi: Array, yi: Array, dx: int = 0, dy: int = 0) -> FieldList:
		"""Evaluate the spatially interpolated potential coefficients.

		The mean coefficient and each complex fluctuation are evaluated at the
		paired coordinates ``(xi, yi)`` using the spline interpolators. Before
		interpolation, coordinates are wrapped on periodic domains or clipped to
		the grid limits otherwise. ``dx`` and ``dy`` select the derivative order
		with respect to each spatial coordinate; both default to zero.

		Returns
		-------
		FieldList
			A pair ``(mean_value, fluctuations)`` with the same coordinate shape as
			``xi`` and ``yi``. Missing mean or fluctuation components are returned
			as ``None``.
		"""
		# Keep every evaluation point inside the domain expected by the splines.
		xi, yi = self.wrap_or_clip(xi, yi)
		mean_value: Array | None = None
		fluctuations: list[Array] | None = None
		if self.fields[0] is not None:
			# The mean field is real, so one spline evaluation is sufficient.
			mean_interpolator = self.interpolators[0]
			if mean_interpolator is None:
				raise RuntimeError("Mean field exists without its interpolator.")
			mean_value = np.asarray(mean_interpolator.ev(xi, yi, dx=dx, dy=dy))
		if self.fields[1] is not None:
			# Each Fourier coefficient uses separate real and imaginary splines.
			fluctuation_interpolators = self.interpolators[1]
			if fluctuation_interpolators is None:
				raise RuntimeError("Fluctuation fields exist without interpolators.")
			fluctuations = [
				np.asarray(
					interp_real.ev(xi, yi, dx=dx, dy=dy)
					+ 1j * interp_imag.ev(xi, yi, dx=dx, dy=dy)
				)
				for interp_real, interp_imag in fluctuation_interpolators
			]
		return mean_value, fluctuations

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
		"""Reconstruct the real potential or one of its derivatives at time ``t``.

		Without ``x`` and ``y``, the field is evaluated on the stored grid. Spatial
		derivatives require coordinates and use the configured spline interpolators.
		``dt=1`` returns the first temporal derivative.
		"""
		if (x is None) != (y is None):
			raise ValueError("`x` and `y` must be provided together.")
		if dt not in (0, 1):
			raise ValueError("`dt` must be 0 or 1.")
		if x is None:
			if dx != 0 or dy != 0:
				raise ValueError("Spatial derivatives require `x` and `y` coordinates.")
			coefficients = self.fields
		else:
			assert y is not None
			coefficients = self.phic_interp(x, y, dx=dx, dy=dy)

		mean_value, fluctuations = coefficients
		if mean_value is not None:
			field = np.zeros_like(mean_value, dtype=float) if dt else np.asarray(mean_value, dtype=float).copy()
		elif fluctuations:
			field = np.zeros_like(fluctuations[0].real, dtype=float)
		else:
			raise ValueError("The potential has no fields to evaluate.")
		if fluctuations is not None:
			for fluctuation, freq in zip(fluctuations, self.freqs):
				phase = np.exp(1j * freq * t)
				if dt == 1:
					phase *= 1j * freq
				field += 2.0 * (fluctuation * phase).real
		return field

	def plot(
		self,
		*,
		contours: int | Sequence[float] | None = 12,
		cmap: str = 'RdBu_r',
		show_quadrature: bool = False,
		show: bool = True,
		**pcolormesh_kwargs: Any,
	) -> list[tuple[Figure, np.ndarray]]:
		"""Plot the mean potential and, optionally, each mode's quadrature."""
		return _plot_potential(
			self,
			contours=contours,
			cmap=cmap,
			show_quadrature=show_quadrature,
			show=show,
			**pcolormesh_kwargs,
		)

	def animate(
		self,
		*,
		t_max: float = 2 * np.pi,
		frames: int = 120,
		interval: int = 50,
		cmap: str = 'RdBu_r',
		repeat: bool = True,
		title: str | None = None,
		**pcolormesh_kwargs: Any,
	) -> FuncAnimation:
		"""Animate the total physical potential reconstructed in time."""
		return _animate_potential(
			self,
			t_max=t_max,
			frames=frames,
			interval=interval,
			cmap=cmap,
			repeat=repeat,
			title=title,
			**pcolormesh_kwargs,
		)


def _field_norm(field: Array) -> mcolors.Normalize:
	"""Return a colour normalization centred at zero whenever possible."""
	vmin, vmax = float(np.nanmin(field)), float(np.nanmax(field))
	if vmin < 0 < vmax:
		return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
	if np.isclose(vmin, vmax):
		delta = abs(vmin) * 0.01 or 1.0
		return mcolors.Normalize(vmin=vmin - delta, vmax=vmax + delta)
	return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _animate_potential(
	potential: Potential,
	*,
	t_max: float = 2 * np.pi,
	frames: int = 120,
	interval: int = 50,
	cmap: str = 'RdBu_r',
	repeat: bool = True,
	title: str | None = None,
	**pcolormesh_kwargs: Any,
) -> FuncAnimation:
	"""Implement :meth:`Potential.animate` outside the data model."""
	if frames < 2:
		raise ValueError('`frames` must be at least 2.')
	if t_max <= 0:
		raise ValueError('`t_max` must be positive.')
	if potential.fields[0] is None and potential.fields[1] is None:
		raise ValueError('The potential has no fields to animate.')

	times = np.linspace(0.0, t_max, frames, endpoint=False)
	first_field = potential.field_at_time(times[0])
	vmin, vmax = float(np.nanmin(first_field)), float(np.nanmax(first_field))
	for t in times[1:]:
		field = potential.field_at_time(t)
		vmin = min(vmin, float(np.nanmin(field)))
		vmax = max(vmax, float(np.nanmax(field)))
	if vmin < 0 < vmax:
		norm: mcolors.Normalize = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
	elif np.isclose(vmin, vmax):
		delta = abs(vmin) * 0.01 or 1.0
		norm = mcolors.Normalize(vmin=vmin - delta, vmax=vmax + delta)
	else:
		norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

	fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
	mesh = ax.pcolormesh(
		potential.x,
		potential.y,
		first_field.T,
		shading='auto',
		cmap=cmap,
		norm=norm,
		**pcolormesh_kwargs,
	)
	fig.colorbar(mesh, ax=ax, label=r'$\phi$')
	ax.set(xlabel='x', ylabel='y', aspect='equal')
	if title is None:
		is_effective = bool(getattr(potential, 'rho', 0))
		name = 'Effective guiding-center potential' if is_effective else 'Potential'
	else:
		name = title

	def update(index: int) -> tuple[Any, ...]:
		mesh.set_array(potential.field_at_time(times[index]).T)
		ax.set_title(f'{name}, t={times[index]:.3f}')
		return mesh, ax.title

	animation = FuncAnimation(
		fig,
		update,
		frames=frames,
		interval=interval,
		blit=False,
		repeat=repeat,
	)
	update(0)
	plt.close(fig)
	return animation


def _draw_field(
	ax: Axes,
	x: Array,
	y: Array,
	field: Array,
	*,
	title: str,
	contours: int | Sequence[float] | None,
	cmap: str,
	pcolormesh_kwargs: dict[str, Any],
) -> None:
	"""Draw one scalar field using the common potential plot style."""
	mesh = ax.pcolormesh(
		x,
		y,
		field.T,
		shading='auto',
		cmap=cmap,
		norm=_field_norm(field),
		**pcolormesh_kwargs,
	)
	if contours is not None:
		ax.contour(x, y, field.T, levels=contours, colors='k', linewidths=0.45, alpha=0.55)
	ax.figure.colorbar(mesh, ax=ax)
	ax.set(title=title, xlabel='x', ylabel='y', aspect='equal')


def _plot_potential(
	potential: Potential,
	*,
	contours: int | Sequence[float] | None = 12,
	cmap: str = 'RdBu_r',
	show_quadrature: bool = False,
	show: bool = True,
	**pcolormesh_kwargs: Any,
) -> list[tuple[Figure, np.ndarray]]:
	"""Implement :meth:`Potential.plot` without coupling plotting to the model."""
	plots: list[tuple[Figure, np.ndarray]] = []
	mean_value, fluctuations = potential.fields
	rho = float(getattr(potential, 'rho', 0.0))
	logger.info(
		"Plotting potential: mean_field=%s, modes=%d, quadrature=%s, gyroaveraged=%s, rho=%g",
		mean_value is not None,
		0 if fluctuations is None else len(fluctuations),
		show_quadrature,
		rho != 0.0,
		rho,
	)
	if mean_value is not None:
		logger.info("Plotting attribute fields[0] as the time-independent mean potential phi_0.")
		fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
		_draw_field(
			ax,
			potential.x,
			potential.y,
			mean_value,
			title=r'Mean potential $\phi_0$',
			contours=contours,
			cmap=cmap,
			pcolormesh_kwargs=pcolormesh_kwargs,
		)
		plots.append((fig, np.asarray([ax], dtype=object)))
	if fluctuations is not None:
		for index, (field, freq) in enumerate(zip(fluctuations, potential.freqs)):
			logger.info(
				"Plotting attribute fields[1][%d] at omega=%g as phi(t=0)=2*Re(phi_c)%s.",
				index,
				freq,
				" with quadrature 2*Im(phi_c)" if show_quadrature else "",
			)
			fig, axes = plt.subplots(
				1, 2 if show_quadrature else 1,
				figsize=(12 if show_quadrature else 6, 5),
				constrained_layout=True,
			)
			axes_array = np.atleast_1d(axes)
			_draw_field(
				axes_array[0],
				potential.x,
				potential.y,
				2.0 * field.real,
				title=rf'$\phi(t=0)=2\operatorname{{Re}}(\phi_c)$, $\omega={freq:g}$',
				contours=contours,
				cmap=cmap,
				pcolormesh_kwargs=pcolormesh_kwargs,
			)
			if show_quadrature:
				_draw_field(
					axes_array[1],
					potential.x,
					potential.y,
					2.0 * field.imag,
					title=rf'$2\operatorname{{Im}}(\phi_c)$ (quadrature), $\omega={freq:g}$',
					contours=contours,
					cmap=cmap,
					pcolormesh_kwargs=pcolormesh_kwargs,
				)
			plots.append((fig, axes_array))
	if not plots:
		raise ValueError('The potential has no fields to plot.')
	if show:
		plt.show()
	return plots
