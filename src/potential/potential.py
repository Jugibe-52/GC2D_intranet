# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Periodic, time-dependent electrostatic potentials and their interpolation.

The spatial information is stored as a complex amplitude on a regular grid.
Keeping the harmonic time dependence separate makes spatial interpolation and
gyroaveraging independent of the evaluation time.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any
import warnings

import numpy as np
from numpy.fft import fft2, fftfreq, ifft2
from scipy.interpolate import RectBivariateSpline
from scipy.special import jv

from .grid import Grid


@dataclass(frozen=True, slots=True)
class _SplineDomain:
	"""Coordinates and padding used by the periodic spline extension.

	``x`` and ``y`` are the extended one-dimensional coordinate axes.  The pair
	``padding = (left, right)`` records how many wrapped samples surround each
	original axis; the asymmetric right side also represents the omitted periodic
	endpoint.
	"""

	x: np.ndarray
	y: np.ndarray
	padding: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _Spline:
	"""Real-valued splines that jointly interpolate a complex amplitude.

	``real`` and ``imag`` represent the corresponding components of the same
	complex field ``C(x, y)`` and therefore share coordinates and derivative
	conventions.
	"""

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
		"""Evaluate ``C`` or its ``(dx, dy)`` derivative at paired coordinates."""
		return np.asarray(
			self.real.ev(x, y, dx=dx, dy=dy)
			+ 1j * self.imag.ev(x, y, dx=dx, dy=dy)
		)


def _spline_domain(grid: Grid, interpolation_order: int) -> _SplineDomain:
	"""Extend both one-dimensional axes for a periodic spline of the given order."""
	# The upper side includes the omitted periodic endpoint in addition to the
	# interpolation margin.  The resulting coordinates remain strictly ordered,
	# as required by ``RectBivariateSpline``.
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
	"""Build a complex periodic interpolant from ``(nx, ny)`` samples."""
	domain = _spline_domain(grid, interpolation_order)
	# Wrapping copies samples from the opposite edge, giving the spline local
	# support across the seam instead of treating it as a physical boundary.
	padded = np.pad(
		coefficient,
		(domain.padding, domain.padding),
		mode="wrap",
	)
	# SciPy's bivariate spline is real-valued, so interpolate both components
	# independently and recombine them only when evaluating the field.
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


class Potential:
	"""One harmonic time mode interpolated on a regular periodic grid.

	For a complex spatial amplitude ``C(x, y)``, the real physical field is
	reconstructed as ``phi(t, x, y) = 2 Re[C(x, y) exp(-i t)]``.  Its magnitude
	therefore controls the local oscillation amplitude and its argument controls
	the local phase.  Time is normalized so this stored harmonic has angular
	frequency one.

	The coefficient is a complex array with shape ``grid.shape == (nx, ny)``:
	the first axis is x and the second is y.  All spatial derivatives use that
	same axis convention.
	"""

	def __init__(
		self,
		grid: Grid,
		coefficient: np.ndarray,
		*,
		interpolation_order: int = 3,
	) -> None:
		"""Store sampled ``C(x, y)`` and prepare its periodic interpolant.

		``interpolation_order`` is the polynomial degree used independently on both
		spatial axes.  It affects off-grid evaluations but not the stored samples.
		"""
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

		``A`` controls the spectral amplitude and ``M`` the maximum radial
		spatial wave number.  Each retained mode receives a reproducible random
		phase and an amplitude that decays as ``|k|**-3``.  ``nx`` and ``ny`` are
		the numbers of physical-space samples; they determine the returned
		coefficient shape, not the number of generated modes.
		"""
		A = float(A)
		if not np.isfinite(A) or A < 0:
			raise ValueError("`A` must be a finite, non-negative number.")
		if isinstance(M, (bool, np.bool_)) or not isinstance(M, (int, np.integer)) or M < 1:
			raise ValueError("`M` must be a positive integer.")
		if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
			raise TypeError("`seed` must be an integer.")

		grid = Grid.periodic(nx, ny)
		# ``wave_x`` and ``wave_y`` index the non-negative integer wave numbers;
		# their common shape is the spectral square ``(M + 1, M + 1)``.
		wave_x, wave_y = np.meshgrid(
			np.arange(M + 1),
			np.arange(M + 1),
			indexing="ij",
		)
		spectrum = np.zeros((M + 1, M + 1), dtype=np.complex128)
		# Axis and constant modes remain zero; the fluctuating potential is built
		# from modes with positive wave numbers in both spatial directions.
		phases = 2 * np.pi * np.random.default_rng(int(seed)).random((M, M))
		spectrum[1:, 1:] = (
			A
			/ (wave_x[1:, 1:] ** 2 + wave_y[1:, 1:] ** 2) ** 1.5
			* np.exp(1j * phases)
		)
		# Apply a circular rather than square cut-off so ``M`` limits |k| without
		# privileging diagonal modes.
		spectrum[np.hypot(wave_x, wave_y) > M] = 0

		# Evaluate the truncated Fourier sum on every physical grid point.  The
		# explicit phase tensor keeps the construction independent of FFT layout.
		mode_x, mode_y = np.indices(spectrum.shape)
		x_mesh, y_mesh = np.meshgrid(grid.x, grid.y, indexing="ij")
		# The first two axes enumerate modes and the last two enumerate physical
		# grid points, giving ``phase`` shape ``(M + 1, M + 1, nx, ny)``.
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
		"""Evaluate the real potential or one of its derivatives.

		Coordinates are paired: ``x[i]`` is evaluated with ``y[i]``.  Omitting
		them evaluates the field on the complete stored grid.  ``dx`` and ``dy``
		select spatial derivative orders, while ``dt=1`` and ``dt=2`` apply the
		corresponding time derivative of the harmonic phase. Scalar coordinates
		produce scalar-shaped results; array coordinates and time follow NumPy
		broadcasting. With no coordinates, spatial axes ``(nx, ny)`` precede any
		axes contributed by time.
		"""
		if (x is None) != (y is None):
			raise ValueError("`x` and `y` must be provided together.")
		if (
			isinstance(dt, (bool, np.bool_))
			or not isinstance(dt, (int, np.integer))
			or dt not in (0, 1, 2)
		):
			raise ValueError("`dt` must be 0, 1, or 2.")
		for derivative, name in ((dx, "dx"), (dy, "dy")):
			if (
				isinstance(derivative, (bool, np.bool_))
				or not isinstance(derivative, (int, np.integer))
				or derivative < 0
			):
				raise ValueError(f"`{name}` must be a non-negative integer.")

		time = np.asarray(t)
		if x is None:
			# Raw samples are available directly, but their spatial derivatives are
			# defined by the interpolant and therefore require explicit coordinates.
			if dx or dy:
				raise ValueError("Spatial derivatives require `x` and `y`.")
			coefficient = self._coefficient
			if time.ndim:
				# Append singleton dimensions so a time array is broadcast after the
				# two spatial grid dimensions.
				coefficient = coefficient.reshape(
					coefficient.shape + (1,) * time.ndim
				)
		else:
			assert y is not None
			# Normalization makes evaluations outside the base cell obey the same
			# periodicity as the coefficients used to construct the spline.
			x_values, y_values = self.grid.normalize(np.asarray(x), np.asarray(y))
			coefficient = self._spline.evaluate(
				x_values,
				y_values,
				dx=int(dx),
				dy=int(dy),
			)

		# The n-th time derivative of exp(-it) contributes (-i)**n.
		phase = np.exp(-1j * time) * (-1j) ** int(dt)
		return np.asarray(np.real(coefficient * phase), dtype=float)

	def electric_field(
		self,
		t: float | np.ndarray,
		x: np.ndarray | None = None,
		y: np.ndarray | None = None,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Return ``(E_x, E_y) = -grad(phi)`` at paired coordinates.

		Both components have the broadcast result shape of ``t``, ``x`` and ``y``;
		when coordinates are omitted they instead use the full ``(nx, ny)`` grid.
		"""
		if x is None and y is None:
			x, y = np.meshgrid(self.grid.x, self.grid.y, indexing="ij")
		elif x is None or y is None:
			raise ValueError("`x` and `y` must be provided together.")
		return (
			-self.evaluate(t, x, y, dx=1),
			-self.evaluate(t, x, y, dy=1),
		)

	def gyroaverage(self, rho: float) -> Potential:
		"""Return the Larmor-circle average at radius ``rho``.

		A circular average multiplies each Fourier mode by ``J_0(rho |k|)``.
		This spectral form performs the average exactly for the sampled modes.
		``rho`` uses the same length scale as the grid coordinates, making
		``rho |k|`` dimensionless.
		"""
		rho = float(rho)
		if not np.isfinite(rho) or rho < 0:
			raise ValueError("`rho` must be finite and non-negative.")
		# ``fftfreq`` returns cycles per unit length; multiplying its norm by
		# ``2*pi`` below converts it to the angular wave number used by J_0.
		kx = fftfreq(self.grid.nx, d=self.grid.dx)
		ky = fftfreq(self.grid.ny, d=self.grid.dy)
		kx_mesh, ky_mesh = np.meshgrid(kx, ky, indexing="ij")
		# ``factor`` has shape ``(nx, ny)`` and attenuates each discrete spatial
		# Fourier coefficient without mixing modes.
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
	) -> Any:
		"""Deprecated adapter for :func:`visualization.plot_potential`."""
		warnings.warn(
			"`Potential.plot()` is deprecated; use `visualization.plot_potential`.",
			DeprecationWarning,
			stacklevel=2,
		)
		from visualization.potential import plot_potential

		return plot_potential(
			self,
			t=t,
			contours=contours,
			cmap=cmap,
			show=show,
			**pcolormesh_kwargs,
		)

	def animate(
		self,
		*,
		t_max: float = 2 * np.pi,
		frames: int = 120,
		interval: int = 50,
		cmap: str = "RdBu_r",
		repeat: bool = True,
		**pcolormesh_kwargs: Any,
	) -> Any:
		"""Deprecated adapter for :func:`visualization.animate_potential`."""
		warnings.warn(
			"`Potential.animate()` is deprecated; use "
			"`visualization.animate_potential`.",
			DeprecationWarning,
			stacklevel=2,
		)
		from visualization.potential import animate_potential

		return animate_potential(
			self,
			t_max=t_max,
			frames=frames,
			interval=interval,
			cmap=cmap,
			repeat=repeat,
			**pcolormesh_kwargs,
		)


__all__ = ["Potential"]
