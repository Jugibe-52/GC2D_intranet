# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Import GC2D HDF5 fields into the common potential representation.

The GC2D HDF5 format stores one real mean field and one or more complex
positive-frequency fields. This module defines their selection,
independent coordinate transformations and provenance metadata. Runtime
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

from .grid import Grid
from ._validation import _readonly_array

DEFAULT_CHARACTERISTIC_LENGTH = 0.06
# Original HDF5 variable-field indices; field zero is always the constant term.
DEFAULT_FIELD_INDICES: tuple[int, ...] = (15,)


def _validated_axis(values: Any, *, name: str) -> np.ndarray:
	"""Validate one uniformly spaced HDF5 coordinate axis."""
	axis: np.ndarray = np.asarray(values, dtype=float)
	if axis.ndim != 1:
		raise ValueError(f"`{name}` must be one-dimensional.")
	if axis.size < 2:
		raise ValueError(f"`{name}` must contain at least two coordinates.")
	if not np.all(np.isfinite(axis)):
		raise ValueError(f"`{name}` must contain finite coordinates.")
	spacing: np.ndarray = np.diff(axis)
	if np.any(spacing <= 0):
		raise ValueError(f"`{name}` must be strictly increasing.")
	if not np.allclose(spacing, spacing[0]):
		raise ValueError(f"`{name}` must be uniformly spaced.")
	return _readonly_array(axis, dtype=float)


def _finite_positive(value: float, *, name: str) -> float:
	"""Convert a scalar to float and require a finite positive value."""
	number: float = float(value)
	if not np.isfinite(number) or number <= 0:
		raise ValueError(f"`{name}` must be finite and positive.")
	return number


def _finite_nonnegative(value: float, *, name: str) -> float:
	"""Convert a scalar to float and require a finite non-negative value."""
	number: float = float(value)
	if not np.isfinite(number) or number < 0:
		raise ValueError(f"`{name}` must be finite and non-negative.")
	return number


def _optional_positive(value: float | None, *, name: str) -> float | None:
	"""Validate one optional positive dimensional scale."""
	if value is None:
		return None
	number: float = float(value)
	if not np.isfinite(number) or number <= 0:
		raise ValueError(f"`{name}` must be finite and positive when supplied.")
	return number


def _validated_source_fields(
	indices: np.ndarray, frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Check one positive dimensional frequency per original HDF5 field index."""
	source_indices: np.ndarray = np.asarray(indices)
	if source_indices.ndim != 1:
		raise ValueError("`source_field_indices` must be one-dimensional.")
	if not np.issubdtype(source_indices.dtype, np.integer):
		raise TypeError("`source_field_indices` must contain integers.")
	if np.any(source_indices < 0):
		raise ValueError("`source_field_indices` must be non-negative.")
	source_frequencies: np.ndarray = np.atleast_1d(np.asarray(frequencies, dtype=float))
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
	number: float = float(value)
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
	# Physical coordinates/time per runtime unit; amplitudes divide by normalization_factor.
	spatial_scale: float = 1.0
	time_scale: float = 1.0
	spatial_rescaling: bool = False
	temporal_rescaling: bool = False
	sigma: float | None = None
	# Missing fields in old archives must not assert corrected physical time units.
	source_frequency_unit: str = "legacy_unspecified"
	time_convention: str = "legacy_unspecified"

	def __post_init__(self) -> None:
		"""Validate, own, and freeze every provenance value."""
		if self.source_frequency_unit not in ("Hz", "legacy_unspecified"):
			raise ValueError("Unsupported source frequency unit.")
		if self.time_convention not in ("physical_seconds", "physical_cycles", "legacy_cycles", "legacy_unspecified"):
			raise ValueError("Unsupported time convention.")
		source_indices: np.ndarray
		source_frequencies: np.ndarray
		source_indices, source_frequencies = _validated_source_fields(
			self.source_field_indices, self.source_frequencies,
		)
		normalization: float = _finite_nonzero(self.normalization_factor, name="normalization_factor")
		attribute_values: dict[str, np.ndarray] = _readonly_attributes(self.attributes)
		for name in ("spatial_scale", "time_scale"):
			object.__setattr__(self, name, _finite_positive(getattr(self, name), name=name))
		for name in ("spatial_rescaling", "temporal_rescaling"):
			object.__setattr__(self, name, _validated_boolean(getattr(self, name), name=name))
		if self.sigma is not None:
			object.__setattr__(self, "sigma", _finite_nonnegative(self.sigma, name="sigma"))

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
				self.spatial_scale,
				self.time_scale,
				self.spatial_rescaling,
				self.temporal_rescaling,
				self.sigma,
				self.source_frequency_unit,
				self.time_convention,
			),
		)


@dataclass(frozen=True, slots=True)
class _GC2DH5Data:
	"""Normalized samples and provenance, before runtime spline preparation.

	Mean samples use (nx, ny), modes use (mode_count, nx, ny), and frequencies
	count cycles per runtime time unit.
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


def _validated_boolean(value: bool, *, name: str) -> bool:
	"""Require an actual Boolean rather than coercing arbitrary truth values."""
	if not isinstance(value, (bool, np.bool_)):
		raise TypeError(f"`{name}` must be a Boolean.")
	return bool(value)


def _validated_field_selection(
	indx: int | Sequence[int] | np.ndarray | None, frequencies: np.ndarray,
) -> np.ndarray:
	"""Validate original HDF5 indices of variable fields, preserving their order."""
	if indx is None:
		selected: np.ndarray = np.flatnonzero(np.isfinite(frequencies) & (frequencies > 0))
	else:
		selected = np.atleast_1d(np.asarray(indx))
		if selected.ndim != 1 or (selected.size and not np.issubdtype(selected.dtype, np.integer)):
			raise ValueError("`indx` must contain one-dimensional integer source field indices.")
		if np.any(selected < 0) or np.any(selected >= frequencies.size):
			raise ValueError(f"Source field indices must be in range [0, {frequencies.size - 1}].")
		selected = selected.astype(int)
	selected_frequencies = frequencies[selected]
	if not np.all(np.isfinite(selected_frequencies) & (selected_frequencies > 0)):
		raise ValueError("Selected variable fields must have finite, strictly positive frequencies.")

	return selected


def _load_data(
	filename: str | PathLike[str],
	*,
	indx: int | Sequence[int] | np.ndarray | None,
	spatial_rescaling: bool,
	temporal_rescaling: bool,
	sigma: float | None,
) -> _GC2DH5Data:
	"""Read samples unchanged by default; apply equivalent variable changes on request.

	Source frequencies are cyclic (Hz for seconds). The historical matrix orientation is retained
	without transposition; only square sampled fields are supported by this import.
	For x=x0+s*q and t=T*u, all amplitudes acquire T/s**2, independently of B.
	"""
	from .potential import _denoise_fields

	spatial: bool = _validated_boolean(spatial_rescaling, name="spatial_rescaling")
	temporal: bool = _validated_boolean(temporal_rescaling, name="temporal_rescaling")
	width: None | float = None if sigma is None else _finite_nonnegative(sigma, name="sigma")
	path: Path = Path(filename)
	with h5py.File(path, "r") as h5:
		x: np.ndarray = _validated_axis(h5["Rcells"][()], name="Rcells")
		y: np.ndarray = _validated_axis(h5["Zcells"][()], name="Zcells")
		frequencies: np.ndarray = np.asarray(h5["freqs"][()], dtype=float)
		if frequencies.ndim != 1:
			raise ValueError("`freqs` must be one-dimensional.")
		fields = h5["fields"]
		expected_shape = (frequencies.size, y.size, x.size)
		if fields.shape != expected_shape:
			raise ValueError(f"Shape of `fields` is {fields.shape}, but expected {expected_shape}.")
		if x.size != y.size:
			raise ValueError("HDF5 loading preserves the existing orientation and requires square sampled fields.")
		if not frequencies.size or frequencies[0] != 0:
			raise ValueError("The first source frequency must be exactly zero for the constant field.")
		indices: np.ndarray = _validated_field_selection(indx, frequencies)
		source_f = frequencies[indices]
		mean: np.ndarray = np.asarray(fields[0].real, dtype=float)
		modes: np.ndarray | None = np.asarray([fields[int(i)] for i in indices], dtype=np.complex128) if indices.size else None
		attributes = {name: np.asarray(value) for name, value in h5.attrs.items()}

	grid: Grid = _grid_from_validated_axes(x, y)
	mean, modes = _denoise_fields(mean, modes, sigma=width)
	length = grid.period if spatial else 1.0
	period: float | None = None
	if temporal:
		if not source_f.size:
			raise ValueError("Temporal rescaling requires a selected positive-frequency mode.")
		period = float(1.0 / source_f[0])
	time_scale = period if period is not None else 1.0
	# This factor follows Hamilton's change of variables and applies to the mean too.
	amplitude_scale = time_scale / length**2
	if amplitude_scale != 1.0:
		mean = mean * amplitude_scale
		if modes is not None:
			modes = modes * amplitude_scale
	if spatial:
		grid = Grid.periodic(grid.nx, grid.ny, period=1.0)
	metadata: GC2DH5Metadata = GC2DH5Metadata(
		source_field_indices=indices, source_x=x, source_y=y,
		source_frequencies=source_f, characteristic_length=None,
		characteristic_period=period, normalization_factor=1.0 / amplitude_scale,
		attributes=attributes, source_path=path, spatial_scale=length,
		time_scale=time_scale, spatial_rescaling=spatial, temporal_rescaling=temporal,
		sigma=width, source_frequency_unit="Hz",
		time_convention="physical_cycles" if temporal else "physical_seconds",
	)
	return _GC2DH5Data(grid, mean, modes, source_f * time_scale, metadata)


__all__ = ["DEFAULT_CHARACTERISTIC_LENGTH", "DEFAULT_FIELD_INDICES", "GC2DH5Metadata"]
