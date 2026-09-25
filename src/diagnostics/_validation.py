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
