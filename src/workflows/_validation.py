"""Shared validation helpers for reproducible numerical-study workflows."""

from __future__ import annotations

import numpy as np


def positive_finite(value: float, name: str) -> float:
	"""Normalize a positive finite workflow parameter."""
	result = float(value)
	if not np.isfinite(result) or result <= 0:
		raise ValueError(f"`{name}` must be positive and finite.")
	return result


def positive_integer(value: int, name: str) -> int:
	"""Normalize a positive integer workflow parameter."""
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


__all__ = ["integer_ratio", "positive_finite", "positive_integer"]
