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
from typing import Any, Literal

import h5py
import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

from .grid import Grid
from .potential import Potential, _readonly_array


DEFAULT_CHARACTERISTIC_LENGTH = 0.06
SpatialNormalization = Literal["characteristic_length", "unit_box"]


@dataclass(frozen=True, slots=True)
class _ComplexSpline:
	"""Real and imaginary splines for one HDF5 complex spatial field."""

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


def _optional_positive(value: float | None, *, name: str) -> float | None:
	"""Validate one optional positive dimensional scale."""
	if value is None:
		return None
	number = float(value)
	if not np.isfinite(number) or number <= 0:
		raise ValueError(f"`{name}` must be finite and positive when supplied.")
	return number


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
	spatial_normalization: SpatialNormalization = "characteristic_length"

	def __post_init__(self) -> None:
		"""Validate, own, and freeze every provenance value."""
		source_indices = np.asarray(self.source_field_indices)
		if source_indices.ndim != 1:
			raise ValueError("`source_field_indices` must be one-dimensional.")
		if not np.issubdtype(source_indices.dtype, np.integer):
			raise TypeError("`source_field_indices` must contain integers.")
		if np.any(source_indices < 0):
			raise ValueError("`source_field_indices` must be non-negative.")

		source_frequencies = np.atleast_1d(
			np.asarray(self.source_frequencies, dtype=float)
		)
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

		normalization = float(self.normalization_factor)
		if not np.isfinite(normalization) or normalization == 0:
			raise ValueError("`normalization_factor` must be finite and non-zero.")

		attribute_values: dict[str, np.ndarray] = {}
		for name, value in self.attributes.items():
			attribute_values[str(name)] = _readonly_array(value, dtype=None)

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
		if not isinstance(self.spatial_normalization, str) or (
			self.spatial_normalization not in ("characteristic_length", "unit_box")
		):
			raise ValueError(
				"`spatial_normalization` must be 'characteristic_length' or 'unit_box'."
			)
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
				self.spatial_normalization,
			),
		)


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


def _h5_spline(
	x: np.ndarray,
	y: np.ndarray,
	coefficient: np.ndarray,
	*,
	interpolation_order: int,
) -> _ComplexSpline:
	"""Build a periodic spline over independent samples of one HDF5 field."""
	margin = interpolation_order + 1
	padding = (margin, margin + 1)
	x_extended = np.pad(
		x,
		padding,
		mode="linear_ramp",
		end_values=(
			x[0] - padding[0] * (x[1] - x[0]),
			x[-1] + padding[1] * (x[1] - x[0]),
		),
	)
	y_extended = np.pad(
		y,
		padding,
		mode="linear_ramp",
		end_values=(
			y[0] - padding[0] * (y[1] - y[0]),
			y[-1] + padding[1] * (y[1] - y[0]),
		),
	)
	field_extended = np.pad(
		coefficient,
		(padding, padding),
		mode="wrap",
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
	for size, name in ((nx, "nx"), (ny, "ny")):
		if (
			isinstance(size, (bool, np.bool_))
			or not isinstance(size, (int, np.integer))
			or size < 2
		):
			raise ValueError(f"`{name}` must be an integer of at least 2.")

	period_x = x.size * (x[1] - x[0])
	period_y = y.size * (y[1] - y[0])
	x_resampled = x[0] + period_x * np.arange(int(nx), dtype=float) / int(nx)
	y_resampled = y[0] + period_y * np.arange(int(ny), dtype=float) / int(ny)

	def interpolate(field: np.ndarray) -> np.ndarray:
		interpolator = _h5_spline(
			x,
			y,
			np.asarray(field, dtype=np.complex128),
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


def load_gc2d_h5_potential(
	filename: str | PathLike[str],
	*,
	B: float = 1.5,
	characteristic_length: float = DEFAULT_CHARACTERISTIC_LENGTH,
	characteristic_frequency: float | None = None,
	indx: int | Sequence[int] | np.ndarray | None = (0, 1),
	nx: int | None = None,
	ny: int | None = None,
	denoising: bool = False,
	sigma: float = 1.0,
	interpolation_order: int = 3,
	spatial_normalization: SpatialNormalization = "characteristic_length",
) -> Potential:
	"""Load a GC2D HDF5 field set and prepare it for runtime evaluation.

	The source file must contain one-dimensional ``Rcells`` and ``Zcells``
	coordinate datasets, a one-dimensional ``freqs`` dataset, and a ``fields``
	dataset with shape ``(len(freqs), len(Zcells), len(Rcells))``. A field whose
	frequency is numerically zero is interpreted as the time-independent mean
	potential. Strictly positive-frequency fields are the complex coefficients of
	the oscillatory potential; negative-frequency fields are omitted because their
	contribution is supplied by taking twice the real part during reconstruction.

	The loader performs the following operations, in order:

	1. Validate the physical scales and the uniformly spaced source axes.
	2. Use the first zero-frequency field as the mean potential, when present.
	3. Discard zero and negative frequencies from the mode collection.
	4. Sort the positive-frequency fields by decreasing spatial peak-to-peak
	   range. The public ``indx`` selectors refer to this sorted collection, not
	   to the original positions in the HDF5 ``fields`` dataset.
	5. Choose the characteristic frequency, normalize the fields and frequencies,
	   and map the physical coordinates to the selected runtime coordinates.
	6. Apply ``indx``, optional Gaussian denoising, and optional periodic
	   resampling before constructing interpolation splines.

	Parameters
	----------
	filename:
		Path to the source HDF5 file.
	B:
		Finite, non-zero magnetic-field scale used in the potential normalization.
		Its sign is retained. The default is ``1.5``.
	characteristic_length:
		Positive physical length ``lambda`` represented by ``2*pi`` in runtime
		coordinates. It must use the same length units as ``Rcells`` and ``Zcells``.
		The coordinate mapping is ``x_hat = 2*pi*(R-Rcells[0])/lambda`` and
		``y_hat = 2*pi*(Z-Zcells[0])/lambda``. The default is ``0.06``.
	characteristic_frequency:
		Optional positive source angular frequency ``omega0``. When omitted, the
		frequency of the first mode after amplitude sorting is used. Runtime
		frequencies are ``omega_j/omega0`` and the corresponding physical period is
		``T0 = 2*pi/omega0``.
	indx:
		Field selectors in the loader's public convention. ``0`` selects the mean;
		``1`` selects the largest retained mode; ``2`` selects the next one,
		and so forth. The default ``(0, 1)`` selects the mean and dominant
		mode. ``None`` selects the mean and every retained mode. The
		order and repetitions in an explicit selector are preserved.
	nx, ny:
		Optional output grid sizes. Supply both to periodically resample every
		selected field, or leave both as ``None`` to retain the source resolution.
		The resampled cell does not duplicate its periodic upper endpoint.
	denoising:
		If true, apply a Gaussian filter before optional resampling. Complex
		modes are filtered component-wise so that their real and imaginary
		parts remain separate.
	sigma:
		Non-negative standard deviation passed to the Gaussian filter. It is
		expressed in grid-sample units and is ignored when ``denoising`` is false.
	interpolation_order:
		Degree of the periodic rectangular splines used for resampling and runtime
		evaluation.
	spatial_normalization:
		Runtime coordinate convention. ``"characteristic_length"`` maps one
		physical characteristic length to ``2*pi``. ``"unit_box"`` maps the
		complete sampled period of each source axis to ``1``. The default is
		``"characteristic_length"``.

	Returns
	-------
	Potential
		A dimensionless, periodically interpolated potential. It evaluates

		``Phi_hat(t_hat, x_hat, y_hat) = Phi_hat_0 +``
		``2*Re(sum_j(C_hat_j*exp(i*2*pi*f_hat_j*t_hat)))``.

		The object retains the original axes, frequencies, HDF5 field indices,
		attributes, source path, and normalization scales for provenance.

	Notes
	-----
	The mean and modes are divided by
	``normalization_factor = omega0*lambda**2*B/(2*pi)**2``. Equivalently,
	``Phi_hat = (2*pi)**2*Phi/(omega0*lambda**2*B)``. If the source has no
	positive-frequency fields and no explicit ``characteristic_frequency``, the
	normalization factor remains one because no temporal scale can be inferred.
	"""
	# Convert public numeric inputs once. The validated local names below carry
	# their physical meaning and avoid repeating implicit scalar conversions.
	magnetic_field = float(B)
	if not np.isfinite(magnetic_field) or magnetic_field == 0:
		raise ValueError("`B` must be finite and non-zero.")
	length_scale = float(characteristic_length)
	if not np.isfinite(length_scale) or length_scale <= 0:
		raise ValueError("`characteristic_length` must be finite and positive.")
	frequency_scale = None
	if characteristic_frequency is not None:
		frequency_scale = float(characteristic_frequency)
		if not np.isfinite(frequency_scale) or frequency_scale <= 0:
			raise ValueError(
				"`characteristic_frequency` must be finite and positive when supplied."
			)
	if not isinstance(denoising, (bool, np.bool_)):
		raise TypeError("`denoising` must be boolean.")
	if not isinstance(spatial_normalization, str) or spatial_normalization not in (
		"characteristic_length",
		"unit_box",
	):
		raise ValueError(
			"`spatial_normalization` must be 'characteristic_length' or 'unit_box'."
		)
	denoising_sigma = float(sigma)
	if not np.isfinite(denoising_sigma) or denoising_sigma < 0:
		raise ValueError("`sigma` must be finite and non-negative.")

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
	# Shift each physical axis to start at zero. The established convention maps
	# one characteristic length to 2*pi; the opt-in unit-box convention maps each
	# complete sampled source period to one.
	if spatial_normalization == "characteristic_length":
		coordinate_scale = 2.0 * np.pi / length_scale
		x = (np.asarray(x) - float(x[0])) * coordinate_scale
		y = (np.asarray(y) - float(y[0])) * coordinate_scale
	else:
		x_period = x.size * (x[1] - x[0])
		y_period = y.size * (y[1] - y[0])
		x = (np.asarray(x) - float(x[0])) / x_period
		y = (np.asarray(y) - float(y[0])) / y_period

	# Translate public selectors into array indices. Selector zero is reserved for
	# the separately stored mean, hence positive selectors require subtracting one.
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
	if denoising and selected_modes is not None:
		selected_modes = np.asarray(
			[
				ndimage.gaussian_filter(field.real, sigma=denoising_sigma)
				+ 1j * ndimage.gaussian_filter(field.imag, sigma=denoising_sigma)
				for field in selected_modes
			],
			dtype=np.complex128,
		)
	if denoising and selected_mean is not None:
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
	# Construct persistent periodic splines and attach both dimensionless runtime
	# data and dimensional provenance to the returned potential object.
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
		spatial_normalization=spatial_normalization,
	)
	return Potential(
		_grid_from_validated_axes(x, y),
		mean=selected_mean,
		modes=selected_modes,
		frequencies=selected_frequencies,
		metadata=metadata,
		interpolation_order=interpolation_order,
	)


__all__ = [
	"DEFAULT_CHARACTERISTIC_LENGTH",
	"GC2DH5Metadata",
	"SpatialNormalization",
	"load_gc2d_h5_potential",
]
