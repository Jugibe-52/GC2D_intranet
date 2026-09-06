# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Periodic electrostatic potentials represented by a mean and harmonic modes.

Every potential uses the same runtime convention, independently of whether its
fields were generated artificially or loaded from measured data:

``Phi(t, x, y) = Phi_0(x, y) + 2 Re sum_j[C_j(x, y) exp(i 2*pi*f_j*t)]``.

Keeping the harmonic time dependence separate makes spatial interpolation and
gyroaveraging independent of the evaluation time.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

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


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
	"""Return an owned, immutable array with the requested dtype."""
	array = np.array(values, dtype=dtype, copy=True)
	array.setflags(write=False)
	return array


@dataclass(frozen=True, slots=True)
class _ValidatedPotentialData:
	"""Validated, immutable fields used to initialize a potential."""

	mean: np.ndarray
	modes: np.ndarray
	frequencies: np.ndarray


def _validated_potential_data(
	grid: Grid,
	mean: np.ndarray | None,
	modes: np.ndarray | None,
	frequencies: np.ndarray | None,
) -> _ValidatedPotentialData:
	"""Validate and own the mutually dependent runtime potential fields."""
	shape = grid.shape
	if mean is None:
		mean_values = np.zeros(shape, dtype=float)
	else:
		mean_values = np.asarray(mean, dtype=float)
		if mean_values.shape != shape:
			raise ValueError(
				f"The mean field shape is {mean_values.shape}; expected {shape}."
			)
		if not np.all(np.isfinite(mean_values)):
			raise ValueError("The mean field must contain finite values.")

	if frequencies is None:
		frequency_values = np.empty(0, dtype=float)
	elif not isinstance(frequencies, np.ndarray):
		raise TypeError("`frequencies` must be a NumPy array or None.")
	else:
		frequency_values = np.asarray(frequencies, dtype=float)
	if frequency_values.ndim != 1:
		raise ValueError("`frequencies` must be one-dimensional.")
	if not np.all(np.isfinite(frequency_values)) or np.any(frequency_values <= 0):
		raise ValueError("`frequencies` must contain finite positive values.")

	if modes is None:
		mode_values = np.empty((0, *shape), dtype=np.complex128)
	else:
		mode_values = np.asarray(modes, dtype=np.complex128)
		if mode_values.ndim != 3 or mode_values.shape[1:] != shape:
			raise ValueError(
				"`modes` must have shape "
				f"(mode_count, {shape[0]}, {shape[1]})."
			)
		if not np.all(np.isfinite(mode_values)):
			raise ValueError("The mode fields must contain finite values.")
	if mode_values.shape[0] != frequency_values.size:
		raise ValueError("The mode count must equal the frequency count.")

	return _ValidatedPotentialData(
		mean=_readonly_array(mean_values, dtype=float),
		modes=_readonly_array(mode_values, dtype=np.complex128),
		frequencies=_readonly_array(frequency_values, dtype=float),
	)


def _random_positive_frequency_mode(
	grid: Grid,
	*,
	amplitude: float,
	maximum_wave_number: int,
	seed: int,
) -> np.ndarray:
	"""Generate the canonical positive-frequency mode for a random potential."""
	wave_x, wave_y = np.meshgrid(
		np.arange(maximum_wave_number + 1),
		np.arange(maximum_wave_number + 1),
		indexing="ij",
	)
	spectrum = np.zeros(
		(maximum_wave_number + 1, maximum_wave_number + 1),
		dtype=np.complex128,
	)
	phases = 2.0 * np.pi * np.random.default_rng(seed).random(
		(maximum_wave_number, maximum_wave_number)
	)
	spectrum[1:, 1:] = (
		amplitude
		/ (wave_x[1:, 1:] ** 2 + wave_y[1:, 1:] ** 2) ** 1.5
		* np.exp(1j * phases)
	)
	# A circular cutoff makes ``maximum_wave_number`` limit |k| without
	# privileging diagonal spatial modes.
	spectrum[np.hypot(wave_x, wave_y) > maximum_wave_number] = 0

	mode_x, mode_y = np.indices(spectrum.shape)
	x_mesh, y_mesh = np.meshgrid(grid.x, grid.y, indexing="ij")
	phase = np.exp(
		1j
		* (
			mode_x[:, :, None, None] * x_mesh[None, None, :, :]
			+ mode_y[:, :, None, None] * y_mesh[None, None, :, :]
		)
	)
	legacy_coefficient = np.asarray(np.einsum("nm,nm...->...", spectrum, phase))
	# Re(C exp(-i*t)) equals 2*Re((conj(C)/2) exp(+i*2*pi*f*t)) for
	# f=1/(2*pi), preserving the established artificial field exactly.
	return np.asarray(0.5 * np.conjugate(legacy_coefficient))


class Potential:
	"""Mean plus positive-frequency modes on a regular periodic grid.

	``mean`` has shape ``grid.shape == (nx, ny)``. ``modes`` has
	shape ``(mode_count, nx, ny)``, and ``frequencies`` stores one positive cycle
	rate per mode. A missing mean is normalized to a zero array and missing modes
	to an empty collection, giving every instance the same internal structure.
	"""

	def __init__(
		self,
		grid: Grid,
		mean: np.ndarray | None = None,
		modes: np.ndarray | None = None,
		frequencies: np.ndarray | None = None,
		*,
		metadata: object | None = None,
		interpolation_order: int = 3,
	) -> None:
		"""Store sampled fields and prepare their periodic interpolants.

		``interpolation_order`` is the polynomial degree used independently on both
		spatial axes.  It affects off-grid evaluations but not the stored samples.
		"""
		if not isinstance(grid, Grid):
			raise TypeError("`grid` must be a Grid instance.")
		if (
			isinstance(interpolation_order, (bool, np.bool_))
			or not isinstance(interpolation_order, (int, np.integer))
			or not 2 <= int(interpolation_order) <= 5
		):
			raise ValueError("`interpolation_order` must be an integer from 2 to 5.")
		data = _validated_potential_data(
			grid,
			mean,
			modes,
			frequencies,
		)
		self.grid = grid
		self.interpolation_order = int(interpolation_order)
		self.mean = data.mean
		self.modes = data.modes
		self.frequencies = data.frequencies
		self.metadata = metadata
		self._mean_spline = _build_spline(
			self.grid, self.mean, self.interpolation_order
		)
		self._mode_splines = tuple(
			_build_spline(self.grid, field, self.interpolation_order)
			for field in self.modes
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
		mode shape, not the number of generated spatial wave numbers.
		"""
		A = float(A)
		if not np.isfinite(A) or A < 0:
			raise ValueError("`A` must be a finite, non-negative number.")
		if (
			isinstance(M, (bool, np.bool_))
			or not isinstance(M, (int, np.integer))
			or M < 1
		):
			raise ValueError("`M` must be a positive integer.")
		if isinstance(seed, (bool, np.bool_)) or not isinstance(
			seed, (int, np.integer)
		):
			raise TypeError("`seed` must be an integer.")

		grid = Grid.periodic(nx, ny)
		mode = _random_positive_frequency_mode(
			grid,
			amplitude=A,
			maximum_wave_number=int(M),
			seed=int(seed),
		)
		return cls(
			grid,
			modes=mode[np.newaxis, ...],
			frequencies=np.asarray([1.0 / (2.0 * np.pi)]),
			interpolation_order=interpolation_order,
		)

	def evaluate(
		self,
		t: float | np.ndarray,
		x: np.ndarray,
		y: np.ndarray,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> np.ndarray:
		"""Evaluate the real potential or a derivative at paired coordinates.

		``x[i]`` is evaluated with ``y[i]``. ``dx`` and ``dy`` select spatial
		derivative orders, while ``dt=1`` and ``dt=2`` differentiate the harmonic
		phase. Scalar coordinates produce scalar-shaped results; arrays and time
		follow NumPy broadcasting.
		"""
		self._validate_derivatives(dx, dy, dt)
		dx, dy, dt = int(dx), int(dy), int(dt)
		time = np.asarray(t)
		x_values, y_values = np.broadcast_arrays(np.asarray(x), np.asarray(y))
		x_values, y_values = self.grid.normalize(x_values, y_values)
		mean_coefficient = (
			self._mean_spline.evaluate(x_values, y_values, dx=dx, dy=dy)
			if dt == 0
			else None
		)
		return self._evaluate_time_dependence(
			time,
			mean_coefficient,
			(
				interpolator.evaluate(x_values, y_values, dx=dx, dy=dy)
				for interpolator in self._mode_splines
			),
			coefficient_shape=x_values.shape,
			dt=dt,
		)

	def evaluate_grid(self, t: float | np.ndarray, *, dt: int = 0) -> np.ndarray:
		"""Evaluate the potential or a time derivative on the complete grid.

		The spatial axes ``(nx, ny)`` precede any axes contributed by time.
		"""
		self._validate_derivatives(0, 0, dt)
		dt = int(dt)
		time = np.asarray(t)
		coefficient_shape = self.grid.shape + (1,) * time.ndim
		mean_coefficient = (
			self.mean.reshape(coefficient_shape) if dt == 0 else None
		)
		return self._evaluate_time_dependence(
			time,
			mean_coefficient,
			self.modes.reshape((len(self.modes), *coefficient_shape)),
			coefficient_shape=coefficient_shape,
			dt=dt,
		)

	def _evaluate_time_dependence(
		self,
		time: np.ndarray,
		mean_coefficient: np.ndarray | None,
		mode_coefficients: Iterable[np.ndarray],
		*,
		coefficient_shape: tuple[int, ...],
		dt: int,
	) -> np.ndarray:
		"""Combine spatial coefficients with their harmonic time dependence."""
		result = np.zeros(np.broadcast_shapes(coefficient_shape, time.shape), dtype=float)
		if mean_coefficient is not None:
			result += np.real(mean_coefficient)
		for coefficient, frequency in zip(
			mode_coefficients,
			self.frequencies,
			strict=True,
		):
			angular_frequency = 2.0 * np.pi * float(frequency)
			phase = np.exp(1j * angular_frequency * time) * (
				1j * angular_frequency
			) ** dt
			result += 2.0 * np.real(coefficient * phase)
		return result

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
		"""Return the Larmor-circle average of every field at radius ``rho``.

		A circular average multiplies each Fourier mode by ``J_0(rho |k|)``.
		This spectral form performs the average exactly for the sampled modes.
		``rho`` uses the same length scale as the grid coordinates, making
		``rho |k|`` dimensionless.
		"""
		radius = float(rho)
		if not np.isfinite(radius) or radius < 0:
			raise ValueError("`rho` must be finite and non-negative.")
		if radius == 0:
			return self
		# ``fftfreq`` returns cycles per unit length; multiplying its norm by
		# ``2*pi`` below converts it to the angular wave number used by J_0.
		kx = fftfreq(self.grid.nx, d=self.grid.dx)
		ky = fftfreq(self.grid.ny, d=self.grid.dy)
		kx_mesh, ky_mesh = np.meshgrid(kx, ky, indexing="ij")
		# ``factor`` has shape ``(nx, ny)`` and attenuates each discrete spatial
		# Fourier coefficient without mixing modes.
		factor = jv(0, 2 * np.pi * radius * np.hypot(kx_mesh, ky_mesh))
		mean = np.asarray(ifft2(fft2(self.mean) * factor).real)
		modes = (
			self.modes.copy()
			if self.modes.shape[0] == 0
			else np.asarray(
				[ifft2(fft2(field) * factor) for field in self.modes],
				dtype=np.complex128,
			)
		)
		return Potential(
			self.grid,
			mean=mean,
			modes=modes,
			frequencies=self.frequencies,
			metadata=self.metadata,
			interpolation_order=self.interpolation_order,
		)

	def _validate_derivatives(self, dx: int, dy: int, dt: int) -> None:
		"""Validate derivative orders supported by the configured splines."""
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
			if derivative >= self.interpolation_order:
				raise ValueError(
					f"`{name}` must be at most {self.interpolation_order - 1} "
					f"for interpolation order {self.interpolation_order}."
				)


__all__ = ["Potential"]
