"""Shared validation for diagnostic sampling and persistence controls."""

from __future__ import annotations

import numpy as np


def positive_integer(value: int, name: str) -> int:
	"""Normalize a positive integer control, rejecting Boolean values."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 1
	):
		raise ValueError(f"`{name}` must be a positive integer.")
	return int(value)


def optional_relative_step(value: float | None) -> float | None:
	"""Check the optional finite-difference scale without changing its value."""
	if value is not None and (
		not np.isfinite(float(value)) or float(value) <= 0.0
	):
		raise ValueError("`relative_step` must be positive and finite.")
	return value
