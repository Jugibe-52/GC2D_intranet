"""Explicit reconstruction of the old measured-field study convention.

This is study composition, not an alternate HDF5 loader: source selection and
validation always use Potential.load. Historical experiments used a legacy amplitude scale
while their forcing clock counted cycles. Keep that choice explicit
when reproducing those experiments; new physical models use GC dynamics with B.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from os import PathLike
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from potential.grid import _validate_periodic_sizes
from potential._periodic_spline import _build_periodic_spline
from potential.load import DEFAULT_FIELD_INDICES, GC2DH5Metadata, _grid_from_validated_axes
from potential.potential import Potential


if TYPE_CHECKING:
	from potential.grid import Grid
	from typing import Any


_P = TypeVar("_P", bound=Potential)


def historical_h5_hamiltonian(
	source: str | PathLike[str], *,
	magnetic_field: float = 1.5,
	length: float = 0.06,
	angular_reference: float | None = None,
	indx: int | Sequence[int] | np.ndarray | None = DEFAULT_FIELD_INDICES,
	sigma: float | None = None,
	interpolation_order: int = 3,
	potential_class: type[Potential] = Potential,
) -> Potential:
	"""Reproduce legacy amplitudes/coordinates and a cycle-time forcing phase.

	The legacy-named angular_reference preserves recorded custom clocks; its
	default is raw source f, not the physical angular rate 2*pi*f. With no
	recorded reference, historical single-mode studies use the first selected mode.
	The returned field is the archived Hamiltonian, including its recorded B factor.
	New physical callers supply B to GuidingCenterDynamics instead.
	"""
	if sigma is not None and (not np.isfinite(sigma) or sigma < 0):
		raise ValueError("`sigma` must be finite and non-negative.")
	length = float(length)
	if not np.isfinite(length) or length <= 0:
		raise ValueError("Historical length must be finite and positive.")
	raw: Potential = potential_class.load(source, indx=indx, sigma=None,
		interpolation_order=interpolation_order)
	metadata = raw.metadata
	assert isinstance(metadata, GC2DH5Metadata)
	if angular_reference is None:
		if metadata.source_frequencies.size == 0:
			raise ValueError("A mean-only historical field requires its recorded angular reference.")
		angular_reference = float(metadata.source_frequencies[0])
	omega0: float = float(angular_reference)
	if not np.isfinite(omega0) or omega0 <= 0:
		raise ValueError("Historical angular reference must be finite and positive.")
	B: float = float(magnetic_field)
	if not np.isfinite(B) or B == 0:
		raise ValueError("`B` must be finite and non-zero.")
	# Match the archived floating-point operation order for fingerprint stability.
	divisor = omega0 * length**2 * B / (2 * np.pi)**2
	mean: np.ndarray
	modes: np.ndarray
	mean, modes = raw.mean / divisor, raw.modes / divisor
	# Filtering historically followed normalization; keep this order at this boundary.
	from potential.potential import _denoise_fields
	filtered: np.ndarray | None
	mean, filtered = _denoise_fields(mean, modes, sigma=sigma)
	assert filtered is not None
	scale = 2 * np.pi / length
	x = (metadata.source_x - metadata.source_x[0]) * scale
	y = (metadata.source_y - metadata.source_y[0]) * scale
	grid: Grid = _grid_from_validated_axes(x, y)
	# The returned object is already a Hamiltonian. B=1 in its downstream dynamics.
	# Normalization above retains exact legacy arithmetic, rather than dividing
	# twice or round-tripping a newly normalized import.
	return potential_class(grid, mean=mean, modes=filtered,
		frequencies=metadata.source_frequencies / omega0,
		metadata=replace(metadata, characteristic_length=length,
			characteristic_period=2 * np.pi / omega0, normalization_factor=divisor,
			spatial_scale=length / (2 * np.pi), time_scale=2 * np.pi / omega0,
			spatial_rescaling=True, temporal_rescaling=True, sigma=sigma, time_convention="legacy_cycles"),
		interpolation_order=interpolation_order)


def sample_study_grid(potential: _P, nx: int | None, ny: int | None) -> _P:
	"""Explicitly sample a historical study field on its recorded evaluation grid.

	This is deliberately separate from loading; ordinary new imports never resample.
	"""
	if nx is None and ny is None:
		return potential
	if nx is None or ny is None:
		raise ValueError("Both study sample counts must be supplied.")
	_validate_periodic_sizes(nx, ny)
	x: np.ndarray
	y: np.ndarray
	x, y = potential.grid.x, potential.grid.y
	metadata = potential.metadata
	if isinstance(metadata, GC2DH5Metadata) and metadata.characteristic_length is not None:
		# Historical resampling fitted the original axes, whose roundoff differs
		# from Grid's regular reconstruction. Retain that fit for archive hashes.
		if (metadata.source_x.size, metadata.source_y.size) != potential.grid.shape:
			raise ValueError("Historical sampling requires the original source resolution.")
		scale = 2 * np.pi / metadata.characteristic_length
		x = (metadata.source_x - metadata.source_x[0]) * scale
		y = (metadata.source_y - metadata.source_y[0]) * scale
	x_out = x[0] + x.size * (x[1] - x[0]) * np.arange(nx, dtype=float) / nx
	y_out = y[0] + y.size * (y[1] - y[0]) * np.arange(ny, dtype=float) / ny
	x_mesh: np.ndarray
	y_mesh: np.ndarray
	x_mesh, y_mesh = np.meshgrid(x_out, y_out, indexing="ij")
	fields = [_build_periodic_spline(x, y, np.asarray(field, dtype=complex),
		spacing=(float(x[1]-x[0]), float(y[1]-y[0])),
		interpolation_order=potential.interpolation_order).evaluate(x_mesh, y_mesh)
		for field in (potential.mean, *potential.modes)]
	grid: Grid = _grid_from_validated_axes(x_out, y_out)
	return type(potential)(grid, mean=fields[0].real,
		modes=np.asarray(fields[1:], dtype=complex).reshape((-1, nx, ny)),
		frequencies=potential.frequencies, metadata=potential.metadata,
		interpolation_order=potential.interpolation_order)
