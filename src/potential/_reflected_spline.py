"""Cubic tensor splines with zero normal derivatives on a closed base cell."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, NdBSpline, make_interp_spline

from ._periodic_spline import _ComplexSpline


@dataclass(frozen=True, slots=True)
class _TensorSplineComponent:
	"""Expose a public SciPy tensor spline through the shared component interface."""

	_spline: NdBSpline

	def ev(
		self, x: np.ndarray, y: np.ndarray, /, *, dx: int = 0, dy: int = 0,
	) -> np.ndarray:
		"""Evaluate paired coordinates or their spatial partial derivatives."""
		points: np.ndarray = np.stack(np.broadcast_arrays(x, y), axis=-1)
		return np.asarray(self._spline(points, nu=(dx, dy), extrapolate=False))

	def get_knots(self) -> tuple[np.ndarray, np.ndarray]:
		"""Return the two clamped cubic knot vectors."""
		return np.asarray(self._spline.t[0]), np.asarray(self._spline.t[1])

	def get_coeffs(self) -> np.ndarray:
		"""Flatten coefficients in the same x-major order as RectBivariateSpline."""
		return np.asarray(self._spline.c).reshape(-1)


def _build_reflected_spline(
	x: np.ndarray, y: np.ndarray, coefficient: np.ndarray,
) -> _ComplexSpline:
	"""Interpolate closed-cell samples with cubic zero-normal boundary conditions.

	The validated axes include both endpoints, and ``coefficient`` has shape
	``(x.size, y.size)``. Each one-dimensional clamped fit sets the first
	derivative to zero at both ends. Applying it to the other axis's coefficients
	preserves these conditions along complete edges, including the corners.
	"""
	x_spline: BSpline = make_interp_spline(
		x, coefficient, k=3, bc_type="clamped", axis=0,
	)
	y_spline: BSpline = make_interp_spline(
		y, x_spline.c.T, k=3, bc_type="clamped", axis=0,
	)
	# SciPy places the interpolation axis first; restore (coefficient_x, coefficient_y).
	coefficients: np.ndarray = np.asarray(y_spline.c.T)
	knots: tuple[np.ndarray, np.ndarray] = (x_spline.t, y_spline.t)
	return _ComplexSpline(
		_TensorSplineComponent(NdBSpline(knots, coefficients.real, (3, 3))),
		_TensorSplineComponent(NdBSpline(knots, coefficients.imag, (3, 3))),
	)
