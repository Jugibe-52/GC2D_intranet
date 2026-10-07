# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Import GC2D HDF5 fields into the common potential representation.

The GC2D HDF5 format stores one real mean field and one or more complex
positive-frequency fields. This module defines their selection,
nondimensionalization, periodic resampling and provenance metadata. Runtime
reconstruction is provided by :class:`~potential.Potential`.
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
from scipy import ndimage

from .grid import Grid, _validate_periodic_sizes
from ._periodic_spline import _build_periodic_spline
from .prepared import _readonly_array

DEFAULT_CHARACTERISTIC_LENGTH = 0.06


def _validated_axis(values: Any, *, name: str) -> np.ndarray:
	"""Validate one uniformly spaced HDF5 coordinate axis."""
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


def _finite_positive(value: float, *, name: str) -> float:
	"""Convert a scalar to float and require a finite positive value."""
	number = float(value)
	if not np.isfinite(number) or number <= 0:
		raise ValueError(f"`{name}` must be finite and positive.")
	return number


def _finite_nonnegative(value: float, *, name: str) -> float:
	"""Convert a scalar to float and require a finite non-negative value."""
	number = float(value)
	if not np.isfinite(number) or number < 0:
		raise ValueError(f"`{name}` must be finite and non-negative.")
	return number


def _optional_positive(value: float | None, *, name: str) -> float | None:
	"""Validate one optional positive dimensional scale."""
	if value is None:
		return None
	number = float(value)
	if not np.isfinite(number) or number <= 0:
		raise ValueError(f"`{name}` must be finite and positive when supplied.")
	return number


def _validated_source_fields(
	indices: np.ndarray, frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Check one positive dimensional frequency per original HDF5 field index."""
	source_indices = np.asarray(indices)
	if source_indices.ndim != 1:
		raise ValueError("`source_field_indices` must be one-dimensional.")
	if not np.issubdtype(source_indices.dtype, np.integer):
		raise TypeError("`source_field_indices` must contain integers.")
	if np.any(source_indices < 0):
		raise ValueError("`source_field_indices` must be non-negative.")
	source_frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
	if (
		source_frequencies.ndim != 1
		or source_frequencies.size != source_indices.size
		or not np.all(np.isfinite(source_frequencies))
		or np.any(source_frequencies <= 0)
	):
		raise ValueError(
			"`source_frequencies` must contain one finite positive value "
			"per source field index."
		)
	return source_indices, source_frequencies


def _finite_nonzero(value: float, *, name: str) -> float:
	"""Normalize a signed physical factor without accepting zero or infinity."""
	number = float(value)
	if not np.isfinite(number) or number == 0:
		raise ValueError(f"`{name}` must be finite and non-zero.")
	return number


def _readonly_attributes(attributes: Mapping[str, Any]) -> dict[str, np.ndarray]:
	"""Own each provenance attribute before wrapping the complete mapping."""
	values: dict[str, np.ndarray] = {}
	for name, value in attributes.items():
		values[str(name)] = _readonly_array(value, dtype=None)
	return values


@dataclass(frozen=True, slots=True)
class GC2DH5Metadata:
	"""Immutable dimensional provenance for a processed GC2D HDF5 potential."""

	source_field_indices: np.ndarray
	source_x: np.ndarray
	source_y: np.ndarray
	source_frequencies: np.ndarray
	characteristic_length: float | None
	characteristic_period: float | None
	normalization_factor: float
	attributes: Mapping[str, Any]
	source_path: Path | None = None

	def __post_init__(self) -> None:
		"""Validate, own, and freeze every provenance value."""
		source_indices, source_frequencies = _validated_source_fields(
			self.source_field_indices, self.source_frequencies,
		)
		normalization = _finite_nonzero(self.normalization_factor, name="normalization_factor")
		attribute_values = _readonly_attributes(self.attributes)

		object.__setattr__(
			self,
			"source_field_indices",
			_readonly_array(source_indices, dtype=int),
		)
		object.__setattr__(
			self,
			"source_x",
			_validated_axis(self.source_x, name="source_x"),
		)
		object.__setattr__(
			self,
			"source_y",
			_validated_axis(self.source_y, name="source_y"),
		)
		object.__setattr__(
			self,
			"source_frequencies",
			_readonly_array(source_frequencies, dtype=float),
		)
		object.__setattr__(
			self,
			"characteristic_length",
			_optional_positive(
				self.characteristic_length,
				name="characteristic_length",
			),
		)
		object.__setattr__(
			self,
			"characteristic_period",
			_optional_positive(
				self.characteristic_period,
				name="characteristic_period",
			),
		)
		object.__setattr__(self, "normalization_factor", normalization)
		object.__setattr__(
			self,
			"attributes",
			MappingProxyType(attribute_values),
		)
		object.__setattr__(
			self,
			"source_path",
			None if self.source_path is None else Path(self.source_path),
		)

	@property
	def characteristic_frequency(self) -> float | None:
		"""Return the dimensional angular-frequency scale derived from the period."""
		if self.characteristic_period is None:
			return None
		return 2.0 * np.pi / self.characteristic_period

	def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
		"""Serialize through the constructor despite the read-only mapping proxy."""
		return (
			type(self),
			(
				self.source_field_indices,
				self.source_x,
				self.source_y,
				self.source_frequencies,
				self.characteristic_length,
				self.characteristic_period,
				self.normalization_factor,
				dict(self.attributes),
				self.source_path,
			),
		)


@dataclass(frozen=True, slots=True)
class _GC2DH5Data:
	"""Normalized samples and provenance, before runtime spline preparation.

	Mean samples use (nx, ny), modes use (mode_count, nx, ny), and frequencies
	are cycles per normalized time, following the common potential convention.
	"""

	grid: Grid
	mean: np.ndarray | None
	modes: np.ndarray | None
	frequencies: np.ndarray
	metadata: GC2DH5Metadata


def _grid_from_validated_axes(x: np.ndarray, y: np.ndarray) -> Grid:
	"""Construct the square-span runtime grid from validated coordinate axes."""
	period_x = x.size * (x[1] - x[0])
	period_y = y.size * (y[1] - y[0])
	if not np.isclose(period_x, period_y):
		raise ValueError(
			"The current Grid API requires equal sampled spans along x and y."
		)
	return Grid(
		float(x[0]),
		float(y[0]),
		float(x[1] - x[0]),
		float(y[1] - y[0]),
		int(x.size),
		int(y.size),
		float(0.5 * (period_x + period_y)),
	)


def _resample_fields(
	x: np.ndarray,
	y: np.ndarray,
	mean: np.ndarray | None,
	modes: np.ndarray | None,
	*,
	nx: int | None,
	ny: int | None,
	interpolation_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
	"""Resample periodic HDF5 fields without duplicating the upper endpoint."""
	if nx is None and ny is None:
		return x, y, mean, modes
	if nx is None or ny is None:
		raise ValueError("`nx` and `ny` must either both be set or both be None.")
	_validate_periodic_sizes(nx, ny)

	period_x = x.size * (x[1] - x[0])
	period_y = y.size * (y[1] - y[0])
	x_resampled = x[0] + period_x * np.arange(int(nx), dtype=float) / int(nx)
	y_resampled = y[0] + period_y * np.arange(int(ny), dtype=float) / int(ny)

	def interpolate(field: np.ndarray) -> np.ndarray:
		interpolator = _build_periodic_spline(
			x,
			y,
			np.asarray(field, dtype=np.complex128),
			spacing=(float(x[1] - x[0]), float(y[1] - y[0])),
			interpolation_order=interpolation_order,
		)
		x_mesh, y_mesh = np.meshgrid(x_resampled, y_resampled, indexing="ij")
		return interpolator.evaluate(x_mesh, y_mesh)

	resampled_mean = None
	if mean is not None:
		resampled_mean = np.asarray(interpolate(mean).real)
	resampled_modes = None
	if modes is not None:
		resampled_modes = np.asarray(
			[interpolate(field) for field in modes],
			dtype=np.complex128,
		)
	return x_resampled, y_resampled, resampled_mean, resampled_modes


def _validated_import_controls(
	# Signed, finite nonzero magnetic field used to normalize field amplitudes.
	B: float,
	# Finite positive length scale in the same units as the source coordinates.
	characteristic_length: float,
	# Positive frequency scale in source units; None defers its choice to loading.
	characteristic_frequency: float | None,
	# Gaussian standard deviation in grid samples; None disables smoothing.
	sigma: float | None,
) -> tuple[float, float, float | None, float | None]:
	"""Normalize dimensional scales and filtering options before opening the file."""
	# Convert public numeric inputs once. The validated local names below carry
	# their physical meaning and avoid repeating implicit scalar conversions.
	magnetic_field = _finite_nonzero(B, name="B")
	length_scale = _finite_positive(
		characteristic_length, name="characteristic_length",
	)
	frequency_scale = _optional_positive(characteristic_frequency, name="characteristic_frequency")
	denoising_sigma = None if sigma is None else _finite_nonnegative(sigma, name="sigma")

	return magnetic_field, length_scale, frequency_scale, denoising_sigma


def _validated_field_selection(
	indx: int | Sequence[int] | np.ndarray | None, mode_count: int,
) -> np.ndarray:
	"""Resolve the public mean selector zero and one-based positive-mode selectors."""
	# Translate public selectors into array indices. Selector zero is reserved for
	# the separately stored mean, hence positive selectors require subtracting one.
	if indx is None:
		selected = np.arange(mode_count + 1, dtype=int)
	else:
		selected = np.atleast_1d(indx).astype(int)
		if (
			selected.size == 0
			or selected.min() < 0
			or selected.max() > mode_count
		):
			raise ValueError(
				f"Indices must be in range [0, {mode_count}]."
			)

	return selected


def _load_data(
	filename: str | PathLike[str],
	*,
	B: float,
	characteristic_length: float,
	characteristic_frequency: float | None,
	indx: int | Sequence[int] | np.ndarray | None,
	nx: int | None,
	ny: int | None,
	sigma: float | None,
	interpolation_order: int,
) -> _GC2DH5Data:
	"""Read and normalize GC2D fields without constructing a runtime potential."""
	magnetic_field, length_scale, frequency_scale, denoising_sigma = _validated_import_controls(
		B, characteristic_length, characteristic_frequency, sigma,
	)

	# Read all source metadata while the HDF5 handle is open. Field samples remain
	# lazy until their selected slices are converted to NumPy arrays below.
	path = Path(filename)
	with h5py.File(path, "r") as h5:
		# Preserve the dimensional coordinate arrays as provenance. The separate
		# ``x`` and ``y`` values are validated immutable working copies.
		source_x = np.asarray(h5["Rcells"][()], dtype=float)
		source_y = np.asarray(h5["Zcells"][()], dtype=float)
		x = _validated_axis(source_x, name="Rcells")
		y = _validated_axis(source_y, name="Zcells")
		all_frequencies = np.atleast_1d(np.asarray(h5["freqs"][()], dtype=float))
		fields = h5["fields"]
		attributes = {name: np.asarray(value) for name, value in h5.attrs.items()}

		# GC2D stores one two-dimensional field for each frequency. Rejecting a
		# mismatched layout here prevents a later mode or coordinate misalignment.
		expected_shape = (len(all_frequencies), len(y), len(x))
		if fields.shape != expected_shape:
			raise ValueError(
				f"Shape of `fields` in {path} is {fields.shape}, "
				f"but expected {expected_shape}."
			)

		# Frequencies close to zero represent stationary data. Only the first such
		# field is the mean-potential channel defined by this importer; its imaginary
		# part is intentionally discarded because a stationary potential is real.
		zero_mask = np.isclose(all_frequencies, 0, atol=1e-5)
		zero_indices = np.flatnonzero(zero_mask)
		mean = (
			None
			if not zero_indices.size
			else np.asarray(fields[int(zero_indices[0])].real, dtype=float)
		)

		# Zero and negative modes do not enter either sorting or reconstruction. A
		# positive coefficient later contributes twice its real part, which already
		# accounts for the conjugate negative-frequency coefficient of a real field.
		retained_indices = np.flatnonzero((~zero_mask) & (all_frequencies >= 0))
		retained_frequencies = all_frequencies[retained_indices]
		if retained_indices.size:
			# Materialize only the positive-frequency slices, then rank them by their
			# spatial variation. ``retained_indices`` follows the same permutation so
			# every runtime mode can still be traced to its original HDF5 slice.
			retained_fields = np.asarray(
				[fields[int(index)] for index in retained_indices],
				dtype=np.complex128,
			)
			amplitudes = np.ptp(retained_fields, axis=(1, 2))
			sort_indices = np.argsort(amplitudes)[::-1]
			retained_frequencies = retained_frequencies[sort_indices]
			retained_fields = retained_fields[sort_indices]
			retained_indices = retained_indices[sort_indices]
			# In the usual case the dominant mode defines omega0. Dividing all fields
			# by the same factor preserves the relative mean/mode amplitudes.
			if frequency_scale is None:
				frequency_scale = float(retained_frequencies[0])
			normalization_factor = float(
				frequency_scale
				* length_scale**2
				* magnetic_field
				/ (2.0 * np.pi) ** 2
			)
			retained_fields = retained_fields / normalization_factor
			if mean is not None:
				mean = mean / normalization_factor
		else:
			# Keep a correctly shaped empty collection so the common selection logic
			# below also works for a file containing only a stationary mean field.
			retained_fields = np.empty(
				(0, len(y), len(x)),
				dtype=np.complex128,
			)
			if frequency_scale is None:
				normalization_factor = 1.0
			else:
				normalization_factor = float(
					frequency_scale
					* length_scale**2
					* magnetic_field
					/ (2.0 * np.pi) ** 2
				)
				if mean is not None:
					mean = mean / normalization_factor

	# Preserve dimensional frequencies before replacing them with omega_j/omega0.
	# The source arrays let callers recover the physical meaning of runtime data.
	retained_source_frequencies = retained_frequencies.copy()
	if frequency_scale is not None:
		retained_frequencies = retained_frequencies / frequency_scale
	# Shift each physical axis to zero and map one characteristic length to 2*pi.
	# The full source period may contain several characteristic lengths.
	coordinate_scale = 2.0 * np.pi / length_scale
	x = (np.asarray(x) - float(x[0])) * coordinate_scale
	y = (np.asarray(y) - float(y[0])) * coordinate_scale

	selected = _validated_field_selection(indx, len(retained_frequencies))

	selected_mean = mean if 0 in selected else None
	mode_selection = selected[selected != 0] - 1
	selected_frequencies = retained_frequencies[mode_selection]
	selected_source_frequencies = retained_source_frequencies[mode_selection]
	selected_fields = retained_fields[mode_selection]
	selected_source_indices = retained_indices[mode_selection]
	selected_modes = (
		None
		if not selected_frequencies.size
		else np.asarray(selected_fields, dtype=np.complex128)
	)
	if selected_mean is None and selected_modes is None:
		raise ValueError("At least one mean or mode field must be selected.")

	# Filter only selected data. Treat real and imaginary components independently
	# rather than relying on complex-valued behavior inside scipy.ndimage.
	if denoising_sigma is not None and selected_modes is not None:
		selected_modes = np.asarray(
			[
				ndimage.gaussian_filter(field.real, sigma=denoising_sigma)
				+ 1j * ndimage.gaussian_filter(field.imag, sigma=denoising_sigma)
				for field in selected_modes
			],
			dtype=np.complex128,
		)
	if denoising_sigma is not None and selected_mean is not None:
		selected_mean = ndimage.gaussian_filter(
			selected_mean,
			sigma=denoising_sigma,
		)

	# Optional resampling uses periodic splines and returns the original arrays
	# unchanged when both requested sizes are None.
	x, y, selected_mean, selected_modes = _resample_fields(
		x,
		y,
		selected_mean,
		selected_modes,
		nx=nx,
		ny=ny,
		interpolation_order=interpolation_order,
	)
	# Keep dimensional provenance alongside the normalized samples; the runtime
	# constructor prepares persistent periodic splines from these data.
	metadata = GC2DH5Metadata(
		source_field_indices=selected_source_indices,
		source_x=source_x,
		source_y=source_y,
		source_frequencies=selected_source_frequencies,
		characteristic_length=length_scale,
		characteristic_period=(
			None if frequency_scale is None else 2.0 * np.pi / frequency_scale
		),
		normalization_factor=normalization_factor,
		attributes=attributes,
		source_path=path,
	)
	return _GC2DH5Data(
		_grid_from_validated_axes(x, y),
		mean=selected_mean,
		modes=selected_modes,
		frequencies=selected_frequencies,
		metadata=metadata,
	)


__all__ = [
	"DEFAULT_CHARACTERISTIC_LENGTH",
	"GC2DH5Metadata",
]
