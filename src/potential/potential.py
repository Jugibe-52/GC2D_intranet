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

from collections.abc import Sequence
from os import PathLike
from functools import cached_property
from typing import Any, Self, cast

import numpy as np
from numpy.fft import fft2, fftfreq, ifft2
from scipy import ndimage
from scipy.special import jv

from .grid import Grid
from .load import (
	DEFAULT_CHARACTERISTIC_LENGTH,
	DEFAULT_FIELD_INDICES,
	_load_data,
)
from contracts.arrays import uses_jax

from ._validation import _readonly_array, _validated_potential_data
from ._periodic_spline import _build_periodic_spline
from ._evaluation import validate_derivatives, normalize_coordinates, reconstruct


def _denoise_fields(
	mean: np.ndarray,
	modes: np.ndarray | None,
	*,
	sigma: float | None,
) -> tuple[np.ndarray, np.ndarray | None]:
	"""Smooth selected fields without modifying the input samples.

	``mean`` has shape (ny, nx) and ``modes`` has shape (mode_count, ny, nx)
	in source-grid order. ``sigma`` is a validated finite, non-negative width
	in grid samples; ``None`` returns the inputs unchanged. Gaussian filtering
	uses SciPy's default reflected boundaries, independently for each mode.
	"""
	if sigma is None:
		return mean, modes
	# Filter complex components separately without mixing harmonic modes.
	if modes is not None:
		modes = np.asarray(
			[
				ndimage.gaussian_filter(field.real, sigma=sigma)
				+ 1j * ndimage.gaussian_filter(field.imag, sigma=sigma)
				for field in modes
			],
			dtype=np.complex128,
		)
	return ndimage.gaussian_filter(mean, sigma=sigma), modes


def _validated_random_parameters(A: float, M: int, seed: int) -> tuple[float, int, int]:
	"""Check random-spectrum controls, retaining the generator's seed policy."""
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

	return A, M, seed


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
		if (isinstance(interpolation_order, (bool, np.bool_))
			or not isinstance(interpolation_order, (int, np.integer))
			or not 2 <= int(interpolation_order) <= 5):
			raise ValueError("`interpolation_order` must be an integer from 2 to 5.")
		self._grid = grid
		self._interpolation_order = int(interpolation_order)
		self._mean, self._modes, self._frequencies = _validated_potential_data(
			grid, mean, modes, frequencies,
		)
		self._splines = tuple(
			_build_periodic_spline(
				grid.x, grid.y, field, spacing=(grid.dx, grid.dy),
				interpolation_order=self.interpolation_order,
			)
			for field in (self.mean, *self.modes)
		)
		self.metadata = metadata
		# Concrete device constants only; traced calls never populate this cache.
		self._jax_buffers: dict[Any, tuple[Any, ...]] = {}

	@property
	def grid(self) -> Grid:
		"""Periodic spatial grid of the sampled potential."""
		return self._grid

	@property
	def interpolation_order(self) -> int:
		"""Polynomial degree of the prepared spatial splines."""
		return self._interpolation_order

	@property
	def mean(self) -> np.ndarray:
		"""Read-only mean samples with shape (nx, ny)."""
		return self._mean

	@property
	def modes(self) -> np.ndarray:
		"""Read-only complex samples with shape (mode_count, nx, ny)."""
		return self._modes

	@property
	def frequencies(self) -> np.ndarray:
		"""Read-only positive harmonic frequencies in cycles per normalized time."""
		return self._frequencies

	def __getstate__(self) -> dict[str, Any]:
		"""Serialize physical data only; device buffers and compiled functions are local."""
		return dict(grid=self.grid, mean=self.mean, modes=self.modes, frequencies=self.frequencies,
		            metadata=self.metadata, interpolation_order=self.interpolation_order)

	def __setstate__(self, state: dict[str, Any]) -> None:
		"""Rebuild CPU splines and empty device caches in the receiving process."""
		Potential.__init__(self, **state)

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
	) -> Self:
		"""Create the reproducible periodic potential used in the notebooks.

		``A`` controls the spectral amplitude and ``M`` the maximum radial
		spatial wave number.  Each retained mode receives a reproducible random
		phase and an amplitude that decays as ``|k|**-3``.  ``nx`` and ``ny`` are
		the numbers of physical-space samples; they determine the returned
		mode shape, not the number of generated spatial wave numbers.
		"""
		A, M, seed = _validated_random_parameters(A, M, seed)

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

	@classmethod
	def load(
		cls,
		filename: str | PathLike[str],
		*,
		B: float = 1.5,
		characteristic_length: float = DEFAULT_CHARACTERISTIC_LENGTH,
		characteristic_frequency: float,
		indx: int | Sequence[int] | np.ndarray | None = DEFAULT_FIELD_INDICES,
		nx: int | None = None,
		ny: int | None = None,
		sigma: float | None = None,
		interpolation_order: int = 3,
	) -> Self:
		"""Construct a potential from measured GC2D HDF5 fields.

		Set ``sigma`` to a finite non-negative Gaussian width in grid samples
		to smooth the fields; ``None`` (the default) disables smoothing.

		The source schema, selection options, and normalization are described
		in ``docs/dynamics/gc2d-h5-import.md``.
		Always include source field 0 as the constant term; its frequency must
		be exactly zero. ``indx`` selects original HDF5 variable-field indices,
		defaulting to ``(15,)``. Their frequencies must be finite and positive.
		Use ``()`` for the constant field alone or ``None`` for all positive
		finite-frequency fields in source order. ``characteristic_frequency`` is
		required, finite and positive, even for constant-only fields. It sets
		the time and amplitude scales; ``None`` is not accepted.
		Defaults use ``B=1.5`` and characteristic length ``0.06``. Coordinates use
		``2*pi*(X-X0)/characteristic_length``. The HDF5 adapter handles
		selection, normalization, optional filtering and resampling; this class
		prepares the common runtime representation and retains source metadata.
		"""
		data = _load_data(
			filename,
			B=B,
			characteristic_length=characteristic_length,
			characteristic_frequency=characteristic_frequency,
			indx=indx,
			nx=nx,
			ny=ny,
			sigma=sigma,
			interpolation_order=interpolation_order,
		)
		return cls(
			data.grid,
			mean=data.mean,
			modes=data.modes,
			frequencies=data.frequencies,
			metadata=data.metadata,
			interpolation_order=interpolation_order,
		)

	@cached_property
	def _spline_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Export shared knots and coefficients once, without refitting splines.

		Coefficients have shape ``(1 + mode_count, ncoeff_x, ncoeff_y)``;
		the first component is the mean and the others are complex harmonics.
		"""
		tx, ty = self._splines[0].real.get_knots()
		shape = (tx.size - self.interpolation_order - 1, ty.size - self.interpolation_order - 1)
		coefficients = []
		for spline in self._splines:
			for component in (spline.real, spline.imag):
				kx, ky = component.get_knots()
				if not (np.array_equal(kx, tx) and np.array_equal(ky, ty)):
					raise ValueError("Potential components must share the same spline knots.")
			coefficients.append((spline.real.get_coeffs() + 1j * spline.imag.get_coeffs()).reshape(shape))
		return (_readonly_array(tx, dtype=float), _readonly_array(ty, dtype=float),
			_readonly_array(np.stack(coefficients), dtype=np.complex128))

	def evaluate(
		self, t: Any, x: Any, y: Any, *, dx: int = 0, dy: int = 0, dt: int = 0,
	) -> Any:
		"""Evaluate matching paired coordinates with NumPy and SciPy splines.

		Time broadcasts against the common coordinate shape. Use JaxPotential
		for JAX arrays, compilation and automatic differentiation.
		Spatial orders must be below the spline degree; time orders are 0, 1, 2.
		"""
		validate_derivatives(self.interpolation_order, dx, dy, dt)
		time, x, y = np.asarray(t), np.asarray(x), np.asarray(y)
		if x.shape != y.shape:
			raise ValueError("`x` and `y` must have the same shape.")
		x, y = normalize_coordinates(x, y, (self.grid.xmin, self.grid.ymin), self.grid.period)
		fields = np.stack([spline.evaluate(x, y, dx=int(dx), dy=int(dy)) for spline in self._splines])
		return reconstruct(time, fields, self.frequencies, int(dt), xp=np)

	def evaluate_grid(self, t: Any, *, dt: int = 0) -> np.ndarray:
		"""Return NumPy grid values with spatial axes before the time axes.

		This host-only operation rejects JAX arrays and tracers. Convert concrete
		JAX time arrays explicitly with ``np.asarray`` before requesting a grid.
		"""
		if uses_jax(t):
			raise TypeError("Grid evaluation requires NumPy time; convert concrete JAX arrays explicitly with np.asarray.")
		validate_derivatives(self.interpolation_order, 0, 0, dt)
		time = np.asarray(t)
		fields = np.concatenate((self.mean[None], self.modes))
		fields = fields.reshape(fields.shape + (1,) * time.ndim)
		return cast(np.ndarray, reconstruct(time, fields, self.frequencies, int(dt), xp=np))

	def electric_field(self, t: Any, x: Any = None, y: Any = None) -> tuple[Any, Any]:
		"""Return (-phi_x, -phi_y) through this instance's evaluate method.

		Omitted coordinates use the full spatial grid; the evaluation backend
		still follows the potential class.
		"""
		if x is None and y is None:
			x, y = np.meshgrid(self.grid.x, self.grid.y, indexing="ij")
		elif x is None or y is None:
			raise ValueError("`x` and `y` must be provided together.")
		return -self.evaluate(t, x, y, dx=1), -self.evaluate(t, x, y, dy=1)

	def gyroaverage(self, rho: float) -> Self:
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
		wave_number_norm = np.sqrt(kx_mesh**2 + ky_mesh**2)
		factor = jv(0, 2 * np.pi * radius * wave_number_norm)
		mean = np.asarray(ifft2(fft2(self.mean) * factor).real)
		modes = (
			self.modes.copy()
			if self.modes.shape[0] == 0
			else np.asarray(
				[ifft2(fft2(field) * factor) for field in self.modes],
				dtype=np.complex128,
			)
		)
		return type(self)(
			self.grid,
			mean=mean,
			modes=modes,
			frequencies=self.frequencies,
			metadata=self.metadata,
			interpolation_order=self.interpolation_order,
		)


__all__ = ["Potential"]
