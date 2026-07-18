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

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from numpy.fft import fft2, ifft2, fftfreq
from typing import Any, Sequence, TypeAlias, cast
os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import RectBivariateSpline
from scipy.special import jv

from contracts import Array
from .grid import Grid

logger = logging.getLogger(__name__)

RealArray: TypeAlias = npt.NDArray[np.float64]
ComplexArray: TypeAlias = npt.NDArray[np.complex128]
FieldArray: TypeAlias = RealArray | ComplexArray

@dataclass(frozen=True, slots=True)
class PotentialMode:
	"""One complex spatial coefficient and its temporal frequency."""

	coefficient: ComplexArray
	frequency: float

	def __post_init__(self) -> None:
		coefficient = cast(ComplexArray, np.asarray(self.coefficient, dtype=np.complex128))
		frequency = float(self.frequency)
		if not np.isfinite(frequency):
			raise ValueError("A potential mode frequency must be finite.")
		object.__setattr__(self, "coefficient", coefficient)
		object.__setattr__(self, "frequency", frequency)

	def copy(self) -> PotentialMode:
		return PotentialMode(self.coefficient.copy(), self.frequency)


@dataclass(frozen=True, slots=True)
class PotentialFields:
	"""Time-independent mean field and frequency-bearing complex modes."""

	mean: RealArray | None = None
	modes: tuple[PotentialMode, ...] = ()

	def __post_init__(self) -> None:
		mean: RealArray | None = None
		if self.mean is not None:
			raw_mean = np.asarray(self.mean)
			if np.iscomplexobj(raw_mean) and not np.allclose(raw_mean.imag, 0.0):
				raise ValueError("The mean potential field must be real-valued.")
			mean = cast(RealArray, np.asarray(raw_mean.real, dtype=np.float64))
		modes = tuple(self.modes)
		if not all(isinstance(mode, PotentialMode) for mode in modes):
			raise TypeError("`modes` must contain only PotentialMode instances.")
		object.__setattr__(self, "mean", mean)
		object.__setattr__(self, "modes", modes)

	@classmethod
	def from_arrays(
		cls,
		mean: RealArray | None,
		coefficients: FieldArray | Sequence[FieldArray] | None,
		frequencies: Sequence[float] | RealArray,
	) -> PotentialFields:
		"""Adapt parallel arrays from storage formats into frequency-bearing modes."""
		frequency_array = np.asarray(frequencies, dtype=float).reshape(-1)
		if coefficients is None:
			coefficient_arrays: tuple[ComplexArray, ...] = ()
		elif isinstance(coefficients, np.ndarray) and coefficients.ndim == 2:
			coefficient_arrays = (
				cast(ComplexArray, np.asarray(coefficients, dtype=np.complex128)),
			)
		else:
			coefficient_arrays = tuple(
				cast(ComplexArray, np.asarray(coefficient, dtype=np.complex128))
				for coefficient in coefficients
			)
		if len(coefficient_arrays) != frequency_array.size:
			raise ValueError(
				f"Received {len(coefficient_arrays)} mode coefficients for "
				f"{frequency_array.size} frequencies."
			)
		return cls(
			mean=mean,
			modes=tuple(
				PotentialMode(coefficient, frequency)
				for coefficient, frequency in zip(coefficient_arrays, frequency_array, strict=True)
			),
		)

	@property
	def frequencies(self) -> RealArray:
		frequencies = cast(RealArray, np.asarray([mode.frequency for mode in self.modes], dtype=np.float64))
		frequencies.setflags(write=False)
		return frequencies

	def copy(self) -> PotentialFields:
		return PotentialFields(
			mean=None if self.mean is None else self.mean.copy(),
			modes=tuple(mode.copy() for mode in self.modes),
		)


@dataclass(frozen=True, slots=True)
class _SplineDomain:
	"""Extended coordinate domain and field-padding policy for splines."""

	x: RealArray
	y: RealArray
	padding: tuple[int, int]
	periodic: bool


@dataclass(frozen=True, slots=True)
class Spline2D:
	"""Real or complex two-dimensional spline with a uniform evaluation API."""

	real: RectBivariateSpline
	imag: RectBivariateSpline | None = None

	def evaluate_points(
		self,
		x: Array,
		y: Array,
		*,
		dx: int = 0,
		dy: int = 0,
	) -> FieldArray:
		"""Evaluate at paired coordinates and optionally take derivatives."""
		value = np.asarray(self.real.ev(x, y, dx=dx, dy=dy))
		if self.imag is not None:
			value = value + 1j * self.imag.ev(x, y, dx=dx, dy=dy)
		return cast(FieldArray, np.asarray(value))

	def evaluate_grid(
		self,
		x: Array,
		y: Array,
		*,
		dx: int = 0,
		dy: int = 0,
	) -> FieldArray:
		"""Evaluate on the Cartesian product of two coordinate axes."""
		value = np.asarray(self.real(x, y, dx=dx, dy=dy))
		if self.imag is not None:
			value = value + 1j * self.imag(x, y, dx=dx, dy=dy)
		return cast(FieldArray, np.asarray(value))


@dataclass(frozen=True, slots=True)
class PotentialInterpolators:
	"""Spline interpolators aligned with a :class:`PotentialFields` object."""

	mean: Spline2D | None = None
	modes: tuple[Spline2D, ...] = ()

	@classmethod
	def build(
		cls,
		grid: Grid,
		fields: PotentialFields,
		*,
		order: int,
	) -> PotentialInterpolators:
		"""Prepare one shared domain, then build one spline per coefficient."""
		domain = _prepare_spline_domain(grid, order=order)
		mean = (
			None
			if fields.mean is None
			else _build_spline(domain, fields.mean, order=order)
		)
		modes = tuple(
			_build_spline(domain, mode.coefficient, order=order)
			for mode in fields.modes
		)
		return cls(mean, modes)


def real_imag(z: ComplexArray) -> tuple[RealArray, RealArray]:
	return z.real, z.imag


def _prepare_spline_domain(grid: Grid, *, order: int) -> _SplineDomain:
	"""Extend both axes far enough to evaluate a spline at the boundaries.

	The boundary policy reserves ``order + 1`` samples beyond each edge. Periodic
	grids omit the duplicated endpoint, so their upper edge receives one additional
	wrapped sample before the full spline margin.
	"""
	margin = order + 1
	periodic = grid.period is not None
	padding = (margin, margin + int(periodic))
	left, right = padding
	x = cast(RealArray, np.pad(
		grid.x,
		padding,
		mode='linear_ramp',
		end_values=(grid.xmin - left * grid.dx, grid.xmax + right * grid.dx),
	))
	y = cast(RealArray, np.pad(
		grid.y,
		padding,
		mode='linear_ramp',
		end_values=(grid.ymin - left * grid.dy, grid.ymax + right * grid.dy),
	))
	return _SplineDomain(x, y, padding, periodic)


def _pad_coefficient(domain: _SplineDomain, coefficient: FieldArray) -> FieldArray:
	"""Extend a coefficient according to the domain's boundary policy."""
	if domain.periodic:
		return cast(FieldArray, np.pad(coefficient, (domain.padding, domain.padding), mode='wrap'))
	return cast(FieldArray, np.pad(
		coefficient,
		(domain.padding, domain.padding),
		mode='constant',
		constant_values=0,
	))


def _build_spline(
	domain: _SplineDomain,
	coefficient: FieldArray,
	*,
	order: int,
) -> Spline2D:
	"""Build one real or complex spline on the prepared domain."""
	padded = _pad_coefficient(domain, coefficient)
	real = RectBivariateSpline(domain.x, domain.y, padded.real, kx=order, ky=order)
	imag = (
		RectBivariateSpline(domain.x, domain.y, padded.imag, kx=order, ky=order)
		if np.iscomplexobj(padded)
		else None
	)
	return Spline2D(real, imag)


def _resample_fields(
	grid: Grid,
	fields: PotentialFields,
	interpolators: PotentialInterpolators,
) -> PotentialFields:
	"""Evaluate a field set's interpolators on a new grid."""
	x, y = grid.x, grid.y
	mean = (
		None
		if interpolators.mean is None
		else cast(RealArray, interpolators.mean.evaluate_grid(x, y))
	)
	modes = tuple(
		PotentialMode(
			cast(ComplexArray, interpolator.evaluate_grid(x, y)),
			mode.frequency,
		)
		for mode, interpolator in zip(
			fields.modes,
			interpolators.modes,
			strict=True,
		)
	)
	return PotentialFields(mean, modes)


def _field_norm(field: RealArray) -> mcolors.Normalize:
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
	if potential.fields.mean is None and not potential.fields.modes:
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
		potential.grid.x,
		potential.grid.y,
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
	field: RealArray,
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
	mean_value = potential.fields.mean
	modes = potential.fields.modes
	rho = float(getattr(potential, 'rho', 0.0))
	logger.info(
		"Plotting potential: mean_field=%s, modes=%d, quadrature=%s, gyroaveraged=%s, rho=%g",
		mean_value is not None,
		len(modes),
		show_quadrature,
		rho != 0.0,
		rho,
	)
	if mean_value is not None:
		logger.info("Plotting fields.mean as the time-independent mean potential phi_0.")
		fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
		_draw_field(
			ax,
			potential.grid.x,
			potential.grid.y,
			mean_value,
			title=r'Mean potential $\phi_0$',
			contours=contours,
			cmap=cmap,
			pcolormesh_kwargs=pcolormesh_kwargs,
		)
		plots.append((fig, np.asarray([ax], dtype=object)))
	for index, mode in enumerate(modes):
		field, freq = mode.coefficient, mode.frequency
		logger.info(
			"Plotting fields.modes[%d] at omega=%g as phi(t=0)=2*Re(phi_c)%s.",
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
			potential.grid.x,
			potential.grid.y,
			2.0 * field.real,
			title=rf'$\phi(t=0)=2\operatorname{{Re}}(\phi_c)$, $\omega={freq:g}$',
			contours=contours,
			cmap=cmap,
			pcolormesh_kwargs=pcolormesh_kwargs,
		)
		if show_quadrature:
			_draw_field(
				axes_array[1],
				potential.grid.x,
				potential.grid.y,
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


class Potential:
	def __init__(
		self,
		grid: Grid,  # Spatial grid and boundary policy.
		fields: PotentialFields,  # Mean field and frequency-bearing complex modes.
		k: int = 3,  # Order of the splines used for interpolation.
	) -> None:
		"""Validate the fields and prepare a potential for evaluation."""
		if not isinstance(grid, Grid):
			raise TypeError("`grid` must be a Grid instance.")
		if not isinstance(fields, PotentialFields):
			raise TypeError("`fields` must be a PotentialFields instance.")
		self.grid: Grid = grid
		self.fields: PotentialFields = fields

		# Ensure every spatial coefficient matches the grid.
		if fields.mean is not None and fields.mean.shape != grid.shape:
			raise ValueError(f"Mean field has shape {fields.mean.shape}, expected {grid.shape}.")
		for index, mode in enumerate(fields.modes):
			if mode.coefficient.shape != grid.shape:
				raise ValueError(
					f"Mode {index} coefficient has shape {mode.coefficient.shape}, expected {grid.shape}."
				)

		# Validate the spline interpolation order.
		self.kinterp: int = k
		if not 1 <= k <= 5:
			raise ValueError("`k` must be between 1 and 5 for RectBivariateSpline.")

		# Build the mean spline and each mode's real/imaginary spline pair.
		self.interpolators: PotentialInterpolators = PotentialInterpolators.build(
			self.grid,
			self.fields,
			order=self.kinterp,
		)

	def resample(self, grid: Grid) -> Potential:
		"""Return this potential evaluated and rebuilt on ``grid``."""
		if not isinstance(grid, Grid):
			raise TypeError("`grid` must be a Grid instance.")
		fields = _resample_fields(grid, self.fields, self.interpolators)
		return Potential(grid, fields, k=self.kinterp)

	def __str__(self) -> str:
		"""Return a compact summary suitable for notebooks and logs."""
		frequencies = np.array2string(self.fields.frequencies, precision=5, threshold=8, edgeitems=3)
		boundary = f'periodic (period={self.grid.period:g})' if self.grid.period is not None else 'zero-padded'
		return (
			'Potential(\n'
			f'  grid={self.grid.nx} x {self.grid.ny}, dx={self.grid.dx:g}, dy={self.grid.dy:g},\n'
			f'  x=[{self.grid.xmin:g}, {self.grid.xmax:g}], y=[{self.grid.ymin:g}, {self.grid.ymax:g}],\n'
			f'  mean_field={self.fields.mean is not None}, modes={len(self.fields.modes)},\n'
			f'  freqs={frequencies}, interpolation_order={self.kinterp}, boundary={boundary}\n'
			')'
		)

	def gyroaveraged(self, rho: float) -> Potential:
		"""Return the effective potential averaged over Larmor circles of radius ``rho``."""
		if rho < 0:
			raise ValueError("`rho` must be non-negative.")
		kx = fftfreq(self.grid.nx, d=self.grid.dx)
		ky = fftfreq(self.grid.ny, d=self.grid.dy)
		kx_, ky_ = np.meshgrid(kx, ky, indexing='ij')
		gyro_factor = jv(0, 2 * np.pi * rho * np.sqrt(kx_**2 + ky_**2))
		mean = None
		if self.fields.mean is not None:
			mean = ifft2(fft2(self.fields.mean) * gyro_factor).real
		modes = tuple(
			PotentialMode(
				ifft2(fft2(mode.coefficient) * gyro_factor),
				mode.frequency,
			)
			for mode in self.fields.modes
		)
		return Potential(self.grid, PotentialFields(mean, modes), k=self.kinterp)

	def phic_interp(self, xi: Array, yi: Array, dx: int = 0, dy: int = 0) -> PotentialFields:
		"""Evaluate the spatially interpolated potential coefficients.

		The mean coefficient and each complex fluctuation are evaluated at the
		paired coordinates ``(xi, yi)`` using the spline interpolators. Before
		interpolation, coordinates are wrapped on periodic domains or clipped to
		the grid limits otherwise. ``dx`` and ``dy`` select the derivative order
		with respect to each spatial coordinate; both default to zero.

		Returns
		-------
		PotentialFields
			The interpolated mean and modes, preserving every mode's frequency.
		"""
		# Keep every evaluation point inside the domain expected by the splines.
		xi, yi = self.grid.wrap_or_clip(xi, yi)
		mean: RealArray | None = None
		if self.fields.mean is not None:
			mean_interpolator = self.interpolators.mean
			if mean_interpolator is None:
				raise RuntimeError("Mean field exists without its interpolator.")
			mean = cast(
				RealArray,
				mean_interpolator.evaluate_points(xi, yi, dx=dx, dy=dy),
			)
		modes = tuple(
		PotentialMode(
			cast(ComplexArray, interpolator.evaluate_points(xi, yi, dx=dx, dy=dy)),
				mode.frequency,
			)
			for mode, interpolator in zip(
				self.fields.modes,
				self.interpolators.modes,
				strict=True,
			)
		)
		return PotentialFields(mean, modes)

	def field_at_time(
		self,
		t: float,
		x: Array | None = None,
		y: Array | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> RealArray:
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
		if coefficients.mean is not None:
			field = (
				np.zeros_like(coefficients.mean, dtype=float)
				if dt
				else np.asarray(coefficients.mean, dtype=float).copy()
			)
		elif coefficients.modes:
			field = np.zeros_like(coefficients.modes[0].coefficient.real, dtype=float)
		else:
			raise ValueError("The potential has no fields to evaluate.")
		for mode in coefficients.modes:
			phase = np.exp(1j * mode.frequency * t)
			if dt == 1:
				phase *= 1j * mode.frequency
			field += 2.0 * (mode.coefficient * phase).real
		return cast(RealArray, field)

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
