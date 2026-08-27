# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Legacy GC2D HDF5 potentials adapted to the current simulation API.

The historical ``gc2d_classes.extract_potential`` loader stores one real mean
field and one or more complex positive-frequency fields.  This module preserves
its selection, scaling, interpolation and ``exp(+i f t)`` reconstruction while
providing the :class:`~potential.Potential` interface required by the current
dynamics and implicit methods.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any

import h5py
import numpy as np
from numpy.fft import fft2, fftfreq, ifft2
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from scipy.special import jv

from .grid import Grid
from .potential import Potential


@dataclass(frozen=True, slots=True)
class _ComplexSpline:
	"""Real and imaginary splines for one legacy complex spatial field."""

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
		"""Evaluate the complex field or a paired-coordinate derivative."""
		return np.asarray(
			self.real.ev(x, y, dx=dx, dy=dy)
			+ 1j * self.imag.ev(x, y, dx=dx, dy=dy)
		)


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
	"""Return an owned, immutable array with the requested dtype."""
	array = np.array(values, dtype=dtype, copy=True)
	array.setflags(write=False)
	return array


def _validated_axis(values: Any, *, name: str) -> np.ndarray:
	"""Validate one uniformly spaced legacy coordinate axis."""
	axis = np.asarray(values, dtype=float)
	if axis.ndim != 1:
		raise ValueError(f"`{name}` must be one-dimensional.")
	if axis.size < 2:
		raise ValueError(f"`{name}` must contain at least two coordinates.")
	if not np.all(np.isfinite(axis)):
		raise ValueError(f"`{name}` must contain finite coordinates.")
	spacing = np.diff(axis)
	if np.any(spacing <= 0):
		raise ValueError(f"`{name}` must be strictly increasing.")
	if not np.allclose(spacing, spacing[0]):
		raise ValueError(f"`{name}` must be uniformly spaced.")
	return _readonly_array(axis, dtype=float)


def _legacy_spline(
	x: np.ndarray,
	y: np.ndarray,
	coefficient: np.ndarray,
	*,
	interpolation_order: int,
) -> _ComplexSpline:
	"""Build the zero-padded spline used by the historical GC2D potential."""
	padding = interpolation_order + 1
	x_extended = np.pad(
		x,
		(padding, padding),
		mode="linear_ramp",
		end_values=(x[0] - padding * (x[1] - x[0]), x[-1] + padding * (x[1] - x[0])),
	)
	y_extended = np.pad(
		y,
		(padding, padding),
		mode="linear_ramp",
		end_values=(y[0] - padding * (y[1] - y[0]), y[-1] + padding * (y[1] - y[0])),
	)
	field_extended = np.pad(
		coefficient,
		((padding, padding), (padding, padding)),
		mode="constant",
		constant_values=0,
	)
	return _ComplexSpline(
		RectBivariateSpline(
			x_extended,
			y_extended,
			field_extended.real,
			kx=interpolation_order,
			ky=interpolation_order,
		),
		RectBivariateSpline(
			x_extended,
			y_extended,
			field_extended.imag,
			kx=interpolation_order,
			ky=interpolation_order,
		),
	)


def _resample_fields(
	x: np.ndarray,
	y: np.ndarray,
	mean_value: np.ndarray | None,
	fluctuations: np.ndarray | None,
	*,
	nx: int | None,
	ny: int | None,
	interpolation_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
	"""Apply the historical inclusive-grid resampling operation once."""
	if nx is None and ny is None:
		return x, y, mean_value, fluctuations
	if nx is None or ny is None:
		raise ValueError("`nx` and `ny` must either both be set or both be None.")
	for size, name in ((nx, "nx"), (ny, "ny")):
		if (
			isinstance(size, (bool, np.bool_))
			or not isinstance(size, (int, np.integer))
			or size < 2
		):
			raise ValueError(f"`{name}` must be an integer of at least 2.")

	x_resampled = np.linspace(x[0], x[-1], int(nx))
	y_resampled = np.linspace(y[0], y[-1], int(ny))

	def interpolate(field: np.ndarray) -> np.ndarray:
		interpolator = _legacy_spline(
			x,
			y,
			np.asarray(field, dtype=np.complex128),
			interpolation_order=interpolation_order,
		)
		x_mesh, y_mesh = np.meshgrid(x_resampled, y_resampled, indexing="ij")
		return interpolator.evaluate(x_mesh, y_mesh)

	resampled_mean = None
	if mean_value is not None:
		resampled_mean = np.asarray(interpolate(mean_value).real)
	resampled_fluctuations = None
	if fluctuations is not None:
		resampled_fluctuations = np.asarray(
			[interpolate(field) for field in fluctuations],
			dtype=np.complex128,
		)
	return x_resampled, y_resampled, resampled_mean, resampled_fluctuations


class GC2DH5Potential(Potential):
	"""Mean plus positive-frequency modes loaded with legacy GC2D semantics.

	The physical field is

	``Phi(t, x, y) = Phi_0(x, y) + 2 Re sum_j[C_j(x, y) exp(+i f_j t)]``.

	Spatial samples use the historical first-axis-x convention without
	transposing HDF5 fields.  Runtime coordinates are clipped to the sampled
	domain and evaluated through zero-padded rectangular splines.  A periodic
	:class:`Grid` still records the regular coordinates because that is the grid
	contract shared by the current simulation API; evaluation itself remains
	non-periodic, as in the legacy implementation.
	"""

	def __init__(
		self,
		x: np.ndarray,
		y: np.ndarray,
		mean_value: np.ndarray | None,
		fluctuations: np.ndarray | None,
		frequencies: np.ndarray | Sequence[float],
		*,
		source_field_indices: np.ndarray | Sequence[int] | None = None,
		normalization_factor: float = 1.0,
		attributes: Mapping[str, Any] | None = None,
		interpolation_order: int = 3,
		source_path: str | PathLike[str] | None = None,
	) -> None:
		"""Store selected legacy fields and prepare their runtime splines."""
		x_values = _validated_axis(x, name="x")
		y_values = _validated_axis(y, name="y")
		shape = (x_values.size, y_values.size)

		mean = None
		if mean_value is not None:
			mean = np.asarray(mean_value, dtype=float)
			if mean.shape != shape:
				raise ValueError(f"The mean field shape is {mean.shape}; expected {shape}.")
			if not np.all(np.isfinite(mean)):
				raise ValueError("The mean field must contain finite values.")

		frequency_values = np.atleast_1d(np.asarray(frequencies, dtype=float))
		if frequency_values.ndim != 1:
			raise ValueError("`frequencies` must be one-dimensional.")
		if not np.all(np.isfinite(frequency_values)) or np.any(frequency_values <= 0):
			raise ValueError("`frequencies` must contain finite positive values.")

		modes = None
		if fluctuations is not None:
			modes = np.asarray(fluctuations, dtype=np.complex128)
			if modes.ndim != 3 or modes.shape[1:] != shape:
				raise ValueError(
					"`fluctuations` must have shape "
					f"(mode_count, {shape[0]}, {shape[1]})."
				)
			if not np.all(np.isfinite(modes)):
				raise ValueError("The fluctuation fields must contain finite values.")
			if modes.shape[0] != frequency_values.size:
				raise ValueError(
					"The fluctuation count must equal the frequency count."
				)
		elif frequency_values.size:
			raise ValueError("Frequencies require corresponding fluctuation fields.")

		if mean is None and modes is None:
			raise ValueError("At least one mean or fluctuation field is required.")

		if source_field_indices is None:
			source_indices = np.arange(frequency_values.size, dtype=int)
		else:
			source_indices = np.asarray(source_field_indices)
			if source_indices.ndim != 1 or source_indices.size != frequency_values.size:
				raise ValueError(
					"`source_field_indices` must contain one index per frequency."
				)
			if not np.issubdtype(source_indices.dtype, np.integer):
				raise TypeError("`source_field_indices` must contain integers.")
		if np.any(source_indices < 0):
			raise ValueError("`source_field_indices` must be non-negative.")

		normalization = float(normalization_factor)
		if not np.isfinite(normalization) or normalization == 0:
			raise ValueError("`normalization_factor` must be finite and non-zero.")

		period_x = x_values.size * (x_values[1] - x_values[0])
		period_y = y_values.size * (y_values[1] - y_values[0])
		if not np.isclose(period_x, period_y):
			raise ValueError(
				"The current Grid API requires equal sampled spans along x and y."
			)
		grid = Grid(
			float(x_values[0]),
			float(y_values[0]),
			float(x_values[1] - x_values[0]),
			float(y_values[1] - y_values[0]),
			int(x_values.size),
			int(y_values.size),
			float(0.5 * (period_x + period_y)),
		)
		carrier = (
			modes[0]
			if modes is not None
			else np.asarray(mean, dtype=np.complex128)
		)
		super().__init__(
			grid,
			carrier,
			interpolation_order=interpolation_order,
		)

		self.x = x_values
		self.y = y_values
		self.mean_value = (
			None if mean is None else _readonly_array(mean, dtype=float)
		)
		self.fluctuations = (
			None if modes is None else _readonly_array(modes, dtype=np.complex128)
		)
		self.frequencies = _readonly_array(frequency_values, dtype=float)
		# ``freqs`` is retained because legacy notebooks use this shorter name.
		self.freqs = self.frequencies
		self.source_field_indices = _readonly_array(source_indices, dtype=int)
		self.normalization_factor = normalization
		attribute_values: dict[str, np.ndarray] = {}
		for name, value in (attributes or {}).items():
			attribute = np.array(value, copy=True)
			attribute.setflags(write=False)
			attribute_values[str(name)] = attribute
		self.attributes: Mapping[str, np.ndarray] = MappingProxyType(attribute_values)
		self.source_path = None if source_path is None else Path(source_path)

		self._mean_spline = (
			None
			if self.mean_value is None
			else _legacy_spline(
				self.x,
				self.y,
				self.mean_value.astype(np.complex128),
				interpolation_order=self.interpolation_order,
			)
		)
		self._fluctuation_splines = tuple(
			_legacy_spline(
				self.x,
				self.y,
				field,
				interpolation_order=self.interpolation_order,
			)
			for field in (() if self.fluctuations is None else self.fluctuations)
		)

	@staticmethod
	def _validate_derivatives(dx: int, dy: int, dt: int) -> None:
		"""Validate derivative orders supported by the potential interface."""
		if dt not in (0, 1):
			raise ValueError("`dt` must be 0 or 1.")
		for derivative, name in ((dx, "dx"), (dy, "dy")):
			if (
				isinstance(derivative, (bool, np.bool_))
				or not isinstance(derivative, (int, np.integer))
				or derivative < 0
			):
				raise ValueError(f"`{name}` must be a non-negative integer.")

	def _spatial_coefficient(
		self,
		field: np.ndarray,
		interpolator: _ComplexSpline,
		x: np.ndarray | None,
		y: np.ndarray | None,
		*,
		dx: int,
		dy: int,
		time_dimensions: int,
	) -> np.ndarray:
		"""Evaluate one stored mode while preserving NumPy broadcasting."""
		if x is None:
			if dx or dy:
				raise ValueError("Spatial derivatives require `x` and `y`.")
			if time_dimensions:
				return field.reshape(field.shape + (1,) * time_dimensions)
			return field
		assert y is not None
		x_values, y_values = np.broadcast_arrays(np.asarray(x), np.asarray(y))
		x_values = np.clip(x_values, self.x[0], self.x[-1])
		y_values = np.clip(y_values, self.y[0], self.y[-1])
		return interpolator.evaluate(
			x_values,
			y_values,
			dx=int(dx),
			dy=int(dy),
		)

	def _zero_result(
		self,
		time: np.ndarray,
		x: np.ndarray | None,
		y: np.ndarray | None,
		*,
		dx: int,
		dy: int,
	) -> np.ndarray:
		"""Build a correctly broadcast zero when a component is absent."""
		if self.mean_value is not None and self._mean_spline is not None:
			field = self.mean_value.astype(np.complex128)
			interpolator = self._mean_spline
		else:
			assert self.fluctuations is not None
			field = self.fluctuations[0]
			interpolator = self._fluctuation_splines[0]
		spatial = self._spatial_coefficient(
			field,
			interpolator,
			x,
			y,
			dx=dx,
			dy=dy,
			time_dimensions=time.ndim,
		)
		return np.asarray(np.real(spatial) * 0.0 + np.asarray(time, dtype=float) * 0.0)

	def dynamic_part(
		self,
		t: float | np.ndarray,
		x: np.ndarray | None = None,
		y: np.ndarray | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> np.ndarray:
		"""Evaluate only the selected positive-frequency contribution."""
		if (x is None) != (y is None):
			raise ValueError("`x` and `y` must be provided together.")
		self._validate_derivatives(dx, dy, dt)
		time = np.asarray(t)
		if self.fluctuations is None:
			return self._zero_result(time, x, y, dx=dx, dy=dy)

		result: np.ndarray | None = None
		for field, interpolator, frequency in zip(
			self.fluctuations,
			self._fluctuation_splines,
			self.frequencies,
		):
			coefficient = self._spatial_coefficient(
				field,
				interpolator,
				x,
				y,
				dx=dx,
				dy=dy,
				time_dimensions=time.ndim,
			)
			phase = np.exp(1j * float(frequency) * time)
			if dt == 1:
				phase = phase * (1j * float(frequency))
			term = 2.0 * np.real(coefficient * phase)
			result = term if result is None else result + term
		assert result is not None
		return np.asarray(result, dtype=float)

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
		"""Evaluate the total legacy potential or one supported derivative."""
		if (x is None) != (y is None):
			raise ValueError("`x` and `y` must be provided together.")
		self._validate_derivatives(dx, dy, dt)
		time = np.asarray(t)
		dynamic = self.dynamic_part(t, x, y, dx=dx, dy=dy, dt=dt)
		if dt == 1 or self.mean_value is None or self._mean_spline is None:
			return dynamic
		mean = self._spatial_coefficient(
			self.mean_value.astype(np.complex128),
			self._mean_spline,
			x,
			y,
			dx=dx,
			dy=dy,
			time_dimensions=time.ndim,
		)
		return np.asarray(np.real(mean) + dynamic, dtype=float)

	def gyroaverage(self, rho: float) -> GC2DH5Potential:
		"""Return the legacy FFT/Bessel Larmor-circle average of every field."""
		radius = float(rho)
		if not np.isfinite(radius) or radius < 0:
			raise ValueError("`rho` must be finite and non-negative.")
		if radius == 0:
			return self

		kx = fftfreq(self.grid.nx, d=self.grid.dx)
		ky = fftfreq(self.grid.ny, d=self.grid.dy)
		kx_mesh, ky_mesh = np.meshgrid(kx, ky, indexing="ij")
		factor = jv(0, 2 * np.pi * radius * np.hypot(kx_mesh, ky_mesh))
		mean = None
		if self.mean_value is not None:
			mean = np.asarray(ifft2(fft2(self.mean_value) * factor).real)
		modes = None
		if self.fluctuations is not None:
			modes = np.asarray(
				[ifft2(fft2(field) * factor) for field in self.fluctuations],
				dtype=np.complex128,
			)
		return GC2DH5Potential(
			self.x,
			self.y,
			mean,
			modes,
			self.frequencies,
			source_field_indices=self.source_field_indices,
			normalization_factor=self.normalization_factor,
			attributes=self.attributes,
			interpolation_order=self.interpolation_order,
			source_path=self.source_path,
		)


def load_gc2d_h5_potential(
	filename: str | PathLike[str],
	*,
	B: float = 1.0,
	indx: int | Sequence[int] | np.ndarray | None = None,
	nx: int | None = None,
	ny: int | None = None,
	denoising: bool = False,
	sigma: float = 1.0,
	interpolation_order: int = 3,
) -> GC2DH5Potential:
	"""Load one HDF5 field set like ``gc2d_classes.extract_potential``.

	Index zero selects the mean field.  Positive indices select fluctuation modes
	after positive-frequency filtering and descending peak-to-peak sorting.  All
	retained fields are normalized by ``2*pi*f0*B``, where ``f0`` is the first
	mode after that sorting, before selection, denoising and optional resampling.
	"""
	magnetic_field = float(B)
	if not np.isfinite(magnetic_field) or magnetic_field == 0:
		raise ValueError("`B` must be finite and non-zero.")
	if not isinstance(denoising, (bool, np.bool_)):
		raise TypeError("`denoising` must be boolean.")
	denoising_sigma = float(sigma)
	if not np.isfinite(denoising_sigma) or denoising_sigma < 0:
		raise ValueError("`sigma` must be finite and non-negative.")

	path = Path(filename)
	with h5py.File(path, "r") as h5:
		x = np.asarray(h5["Rcells"][()], dtype=float)
		y = np.asarray(h5["Zcells"][()], dtype=float)
		all_frequencies = np.atleast_1d(np.asarray(h5["freqs"][()], dtype=float))
		fields = h5["fields"]
		attributes = {name: np.asarray(value) for name, value in h5.attrs.items()}

		expected_shape = (len(all_frequencies), len(y), len(x))
		if fields.shape != expected_shape:
			raise ValueError(
				f"Shape of `fields` in {path} is {fields.shape}, "
				f"but expected {expected_shape}."
			)

		zero_mask = np.isclose(all_frequencies, 0, atol=1e-5)
		zero_indices = np.flatnonzero(zero_mask)
		mean_value = (
			None
			if not zero_indices.size
			else np.asarray(fields[int(zero_indices[0])].real, dtype=float)
		)

		# Zero and negative modes do not enter either sorting or reconstruction.
		retained_indices = np.flatnonzero((~zero_mask) & (all_frequencies >= 0))
		retained_frequencies = all_frequencies[retained_indices]
		if retained_indices.size:
			retained_fields = np.asarray(
				[fields[int(index)] for index in retained_indices],
				dtype=np.complex128,
			)
			amplitudes = np.ptp(retained_fields, axis=(1, 2))
			sort_indices = np.argsort(amplitudes)[::-1]
			retained_frequencies = retained_frequencies[sort_indices]
			retained_fields = retained_fields[sort_indices]
			retained_indices = retained_indices[sort_indices]
			normalization_factor = float(
				2.0 * np.pi * retained_frequencies[0] * magnetic_field
			)
			retained_fields = retained_fields / normalization_factor
			if mean_value is not None:
				mean_value = mean_value / normalization_factor
		else:
			retained_fields = np.empty(
				(0, len(y), len(x)),
				dtype=np.complex128,
			)
			normalization_factor = 1.0

	if indx is None:
		selected = np.arange(len(retained_frequencies) + 1, dtype=int)
	else:
		selected = np.atleast_1d(indx).astype(int)
		if (
			selected.size == 0
			or selected.min() < 0
			or selected.max() > len(retained_frequencies)
		):
			raise ValueError(
				f"Indices must be in range [0, {len(retained_frequencies)}]."
			)

	selected_mean = mean_value if 0 in selected else None
	fluctuation_selection = selected[selected != 0] - 1
	selected_frequencies = retained_frequencies[fluctuation_selection]
	selected_fields = retained_fields[fluctuation_selection]
	selected_source_indices = retained_indices[fluctuation_selection]
	selected_fluctuations = (
		None
		if not selected_frequencies.size
		else np.asarray(selected_fields, dtype=np.complex128)
	)

	if denoising and selected_fluctuations is not None:
		selected_fluctuations = np.asarray(
			[
				ndimage.gaussian_filter(field.real, sigma=denoising_sigma)
				+ 1j * ndimage.gaussian_filter(field.imag, sigma=denoising_sigma)
				for field in selected_fluctuations
			],
			dtype=np.complex128,
		)
	if denoising and selected_mean is not None:
		selected_mean = ndimage.gaussian_filter(
			selected_mean,
			sigma=denoising_sigma,
		)

	x, y, selected_mean, selected_fluctuations = _resample_fields(
		x,
		y,
		selected_mean,
		selected_fluctuations,
		nx=nx,
		ny=ny,
		interpolation_order=interpolation_order,
	)
	return GC2DH5Potential(
		x,
		y,
		selected_mean,
		selected_fluctuations,
		selected_frequencies,
		source_field_indices=selected_source_indices,
		normalization_factor=normalization_factor,
		attributes=attributes,
		interpolation_order=interpolation_order,
		source_path=path,
	)


__all__ = ["GC2DH5Potential", "load_gc2d_h5_potential"]
