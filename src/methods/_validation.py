"""Scalar configuration contracts shared by numerical methods.

Some established constructors accept booleans through float conversion, while
extended implicit methods reject them. Callers declare that distinction instead
of changing accepted inputs when sharing validation.
"""

from __future__ import annotations

import numpy as np


def _positive_finite(
	value: float,
	name: str,
	*,
	allow_boolean: bool = False,
	message: str | None = None,
) -> float:
	"""Return a positive finite float under the caller's boolean-input contract."""
	error = message if message is not None else f"`{name}` must be positive and finite."
	if not allow_boolean and isinstance(value, (bool, np.bool_)):
		raise ValueError(error)
	result = float(value)
	if not np.isfinite(result) or result <= 0:
		raise ValueError(error)
	return result


def _nonnegative_finite(
	value: float,
	name: str,
	*,
	allow_boolean: bool = False,
	message: str | None = None,
) -> float:
	"""Return a nonnegative finite float with the constructor's existing policy."""
	error = message if message is not None else f"`{name}` must be non-negative and finite."
	if not allow_boolean and isinstance(value, (bool, np.bool_)):
		raise ValueError(error)
	result = float(value)
	if not np.isfinite(result) or result < 0:
		raise ValueError(error)
	return result


def _positive_integer(value: int, name: str) -> int:
	"""Return a positive built-in integer without accepting boolean values."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 1
	):
		raise ValueError(f"`{name}` must be a positive integer.")
	return int(value)


__all__: list[str] = []
