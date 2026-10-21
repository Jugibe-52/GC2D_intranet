"""Scalar and fixed-grid rules shared by study configurations and stored results."""

from __future__ import annotations
import numpy as np


def require_uncoupled_bm4(value: float | None) -> None:
	"""Reject coupled legacy calculation requests without relabeling archives.

	Study formats may retain a zero rate (or None for methods with no rate).
	Stored-result readers keep historical metadata without invoking this guard.
	"""
	if value is not None and (
		isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value)) or float(value) != 0.0
	):
		raise ValueError("BM4 no longer supports harmonic coupling; use coupling_frequency=0 for new calculations.")


def positive_finite(value: float, name: str) -> float:
	"""Normalize a positive finite study parameter."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be positive and finite.")
	result = float(value)
	if not np.isfinite(result) or result <= 0:
		raise ValueError(f"`{name}` must be positive and finite.")
	return result



def nonnegative_finite(value: float, name: str) -> float:
	"""Normalize a finite study parameter that may be zero."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be finite and non-negative.")
	result = float(value)
	if not np.isfinite(result) or result < 0:
		raise ValueError(f"`{name}` must be finite and non-negative.")
	return result



def positive_integer(value: int, name: str) -> int:
	"""Normalize a positive integer study parameter."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 1
	):
		raise ValueError(f"`{name}` must be a positive integer.")
	return int(value)



def integer_ratio(numerator: float, denominator: float, name: str) -> int:
	"""Return an exact positive sampling ratio within floating-point tolerance."""
	ratio = numerator / denominator
	rounded = int(round(ratio))
	if rounded < 1 or not np.isclose(ratio, rounded, rtol=1e-12, atol=1e-12):
		raise ValueError(f"`{name}` must be a positive integer ratio.")
	return rounded



def finite_time_span(
	values: tuple[float, float],
	*,
	message: str = "`t_span` must contain two finite increasing times.",
) -> tuple[float, float]:
	"""Normalize an array-shaped pair without changing conversion exceptions."""
	span = np.asarray(values, dtype=float)
	if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
		raise ValueError(message)
	return float(span[0]), float(span[1])
