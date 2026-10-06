# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Immutable host-side fields and splines shared by potential evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from .grid import Grid
from ._periodic_spline import _ComplexSpline, _build_periodic_spline


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


@dataclass(frozen=True)
class SplineCoefficients:
    """Read-only tensor-product coefficients, with a leading mean/mode axis."""

    knots_x: np.ndarray
    knots_y: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True)
class PreparedPotential:
    """Canonical prepared field; devices derive their buffers from this record.

    Samples use ``(nx, ny)`` spatial axes and retain their physical normalization.
    SciPy interpolants are private preparation resources. Their exported knots
    and coefficients are cached read-only arrays, never refitted for a device.
    """

    grid: Grid
    mean: np.ndarray
    modes: np.ndarray
    frequencies: np.ndarray
    interpolation_order: int
    _splines: tuple[_ComplexSpline, ...]

    @classmethod
    def build(cls, grid: Grid, mean: Any, modes: Any, frequencies: Any, degree: int) -> PreparedPotential:
        """Validate samples, own their storage, and fit each component once."""
        if not isinstance(grid, Grid):
            raise TypeError("`grid` must be a Grid instance.")
        if (isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer))
                or not 2 <= int(degree) <= 5):
            raise ValueError("`interpolation_order` must be an integer from 2 to 5.")
        data = _validated_potential_data(grid, mean, modes, frequencies)
        splines = tuple(
            _build_periodic_spline(
                grid.x, grid.y, field,
                spacing=(grid.dx, grid.dy), interpolation_order=int(degree),
            )
            for field in (data.mean, *data.modes)
        )
        return cls(grid, data.mean, data.modes, data.frequencies, int(degree), splines)

    @cached_property
    def spline_data(self) -> SplineCoefficients:
        """Export a common, read-only coefficient representation on first use."""
        tx, ty = self._splines[0].real.get_knots()
        shape = (tx.size - self.interpolation_order - 1, ty.size - self.interpolation_order - 1)
        coefficients = []
        for spline in self._splines:
            for component in (spline.real, spline.imag):
                kx, ky = component.get_knots()
                if not (np.array_equal(kx, tx) and np.array_equal(ky, ty)):
                    raise ValueError("Potential components must share the same spline knots.")
            coefficients.append((spline.real.get_coeffs() + 1j * spline.imag.get_coeffs()).reshape(shape))
        return SplineCoefficients(_readonly_array(tx, dtype=float), _readonly_array(ty, dtype=float),
                                  _readonly_array(np.stack(coefficients), dtype=np.complex128))

    def spatial_fields(self, x: np.ndarray, y: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Evaluate the canonical SciPy splines, with the mean followed by modes."""
        return np.stack([spline.evaluate(x, y, dx=dx, dy=dy) for spline in self._splines])

    @cached_property
    def samples(self) -> np.ndarray:
        """Stack immutable sampled fields for shared time reconstruction."""
        return _readonly_array(np.concatenate((self.mean[None], self.modes)), dtype=np.complex128)


def prepared_source(source: Any) -> PreparedPotential:
    """Resolve the public potential or an already prepared immutable record."""
    if isinstance(source, PreparedPotential):
        return source
    from .potential import Potential

    if not isinstance(source, Potential):
        raise TypeError("`potential` must be a Potential or PreparedPotential instance.")
    return source.prepared


__all__ = ["PreparedPotential", "SplineCoefficients"]
