"""Shared spatial extension and display transforms for potential fields."""

from typing import Any, Literal, cast

import numpy as np


SpatialExtension = Literal["periodic", "reflected"]


def validate_spatial_extension(value: str) -> SpatialExtension:
	"""Require one explicitly supported scalar extension convention."""
	if not isinstance(value, str) or value not in ("periodic", "reflected"):
		raise ValueError("`spatial_extension` must be 'periodic' or 'reflected'.")
	return cast(SpatialExtension, value)


def reflected_coordinates(
	x: Any, y: Any, origin: Any, width: Any, *, xp: Any,
) -> tuple[Any, Any, Any, Any]:
	"""Return folded coordinates and one-sided orientation signs.

	The base square has width L, whereas the scalar repeats after 2L. Signs
	remain +/-1 at seams: clamped spline coefficients supply zero normal
	derivatives without masking the Hessian during automatic differentiation.
	"""
	u: Any = (x - origin[0]) % (2 * width)
	v: Any = (y - origin[1]) % (2 * width)
	sx: Any = xp.where(u <= width, 1.0, -1.0)
	sy: Any = xp.where(v <= width, 1.0, -1.0)
	return (
		origin[0] + xp.where(u <= width, u, 2 * width - u),
		origin[1] + xp.where(v <= width, v, 2 * width - v),
		sx, sy,
	)


def closed_reflected_samples(fields: np.ndarray, samples: np.ndarray | None) -> np.ndarray:
	"""Own closed fields, completing missing upper endpoints only on input.

	Fields have shape (1+m,nx,ny). Explicit closed data preserve independent
	upper edges after averaging; the initial fallback copies the existing
	periodic trace rather than inventing values from an extrapolation.
	"""
	expected: tuple[int, ...] = (fields.shape[0], fields.shape[1] + 1, fields.shape[2] + 1)
	closed: np.ndarray = (
		np.pad(fields, ((0, 0), (0, 1), (0, 1)), mode="wrap")
		if samples is None else np.array(samples, dtype=np.complex128, copy=True)
	)
	if closed.shape != expected:
		raise ValueError(f"`reflected_samples` must have shape {expected}.")
	if not np.all(np.isfinite(closed)):
		raise ValueError("`reflected_samples` must contain finite values.")
	if np.any(closed[0].imag != 0):
		raise ValueError("The reflected mean samples must be real.")
	# Real and complex division may round the same real mean differently.
	# Accept only roundoff-sized assembly differences, then retain the canonical
	# half-open samples exactly; upper edges remain independent.
	tolerance: float = float(8 * np.finfo(float).eps)
	scale: float = float(np.max(np.abs(fields)))
	if not np.allclose(closed[:, :-1, :-1], fields, rtol=tolerance, atol=tolerance * scale):
		raise ValueError("The interior of `reflected_samples` must match the mean and modes.")
	closed[:, :-1, :-1] = fields
	closed.setflags(write=False)
	return closed
