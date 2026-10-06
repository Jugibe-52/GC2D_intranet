# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Shared complex periodic spline construction for preparation and resampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RectBivariateSpline


@dataclass(frozen=True, slots=True)
class _ComplexSpline:
	"""Real and imaginary interpolants of one complex spatial field."""

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
		"""Evaluate the field or its ``(dx, dy)`` derivative at paired points."""
		return np.asarray(
			self.real.ev(x, y, dx=dx, dy=dy)
			+ 1j * self.imag.ev(x, y, dx=dx, dy=dy)
		)


def _build_periodic_spline(
	x: np.ndarray,
	y: np.ndarray,
	coefficient: np.ndarray,
	*,
	spacing: tuple[float, float],
	interpolation_order: int,
) -> _ComplexSpline:
	"""Interpolate validated ``(nx, ny)`` samples on half-open uniform axes.

	Axes and their spacings use the same physical or normalized coordinates.
	Callers supply the validated spacing explicitly so runtime grids retain
	their original precision instead of recovering it by coordinate subtraction.
	"""
	margin = interpolation_order + 1
	padding = (margin, margin + 1)
	# The extra upper sample represents the omitted periodic endpoint. Wrapped
	# data give the interpolant local support across both periodic seams.
	extended_axes = tuple(
		np.pad(
			axis,
			padding,
			mode="linear_ramp",
			end_values=(
				axis[0] - padding[0] * step,
				axis[-1] + padding[1] * step,
			),
		)
		for axis, step in zip((x, y), spacing, strict=True)
	)
	padded = np.pad(coefficient, (padding, padding), mode="wrap")
	# SciPy fits real fields; both components share axes and derivative rules.
	return _ComplexSpline(
		RectBivariateSpline(
			*extended_axes, padded.real,
			kx=interpolation_order, ky=interpolation_order,
		),
		RectBivariateSpline(
			*extended_axes, padded.imag,
			kx=interpolation_order, ky=interpolation_order,
		),
	)
