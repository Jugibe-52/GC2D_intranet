"""Shared validation helpers for reproducible numerical studies."""

from __future__ import annotations

import numpy as np


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


def resolve_rho(explicit: float | None, legacy_source: object) -> float:
	"""Resolve canonical study rho with a legacy-configuration fallback."""
	if explicit is not None:
		return nonnegative_finite(explicit, "rho")
	legacy = getattr(legacy_source, "rho", None)
	if legacy is None:
		raise ValueError(
			"`rho` must be explicit because the initial configuration has no "
			"legacy physical metadata."
		)
	return nonnegative_finite(legacy, "rho")


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


__all__ = [
	"integer_ratio",
	"nonnegative_finite",
	"positive_finite",
	"positive_integer",
	"resolve_rho",
]
