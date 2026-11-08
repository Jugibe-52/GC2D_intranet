"""Immutable Butcher coefficients for consistent Runge--Kutta methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def _real_coefficients(value: ArrayLike) -> np.ndarray:
	"""Own finite real coefficients without silently discarding imaginary parts."""
	array: np.ndarray = np.asarray(value)
	if array.dtype.kind not in 'iuf' or not np.all(np.isfinite(array)):
		raise ValueError('Butcher coefficients must be finite real numbers.')
	return np.asarray(array, dtype=float)


def _immutable_array(value: np.ndarray) -> np.ndarray:
	"""Use an immutable buffer so callers cannot re-enable write access."""
	return np.frombuffer(value.tobytes(), dtype=float).reshape(value.shape)


@dataclass(frozen=True, slots=True, init=False)
class ButcherTableau:
	"""Dimensionless A (s,s), B (s,) and C (s,) with s positive stages.

	Row-sum and weight checks establish consistency only, never higher order,
	stability, symmetry or symplecticity. Coefficients determine equality/hash.
	"""

	A: np.ndarray = field(compare=False, hash=False, repr=False)
	B: np.ndarray = field(compare=False, hash=False, repr=False)
	C: np.ndarray = field(compare=False, hash=False, repr=False)
	_coefficients: tuple[tuple[float, ...], ...] = field(repr=False)

	def __init__(self, A: ArrayLike, B: ArrayLike, C: ArrayLike) -> None:
		matrix: np.ndarray = _real_coefficients(A)
		weights: np.ndarray = _real_coefficients(B)
		nodes: np.ndarray = _real_coefficients(C)
		if (weights.ndim != 1 or weights.size == 0
			or nodes.shape != weights.shape or matrix.shape != (weights.size, weights.size)):
			raise ValueError('A must have shape (s,s) and B/C shape (s,), with s > 0.')
		# Scale the roundoff allowance by the terms being summed, including cancellation.
		epsilon: float = float(64 * np.finfo(float).eps)
		with np.errstate(over='ignore', invalid='ignore'):
			row_sums: np.ndarray = matrix.sum(axis=1)
			row_scale: np.ndarray = np.abs(matrix).sum(axis=1)
			weight_sum: float = float(weights.sum())
			weight_scale: float = float(np.abs(weights).sum())
		if not (np.all(np.isfinite(row_sums)) and np.all(np.isfinite(row_scale))
			and np.isfinite(weight_sum) and np.isfinite(weight_scale)):
			raise ValueError('Butcher coefficient sums must remain finite.')
		if not np.all(np.abs(row_sums - nodes) <= epsilon * np.maximum(1., row_scale)):
			raise ValueError('C must equal the row sums of A within roundoff.')
		if abs(weight_sum - 1.) > epsilon * max(1., weight_scale):
			raise ValueError('B must sum to one within roundoff.')
		object.__setattr__(self, 'A', _immutable_array(matrix))
		object.__setattr__(self, 'B', _immutable_array(weights))
		object.__setattr__(self, 'C', _immutable_array(nodes))
		object.__setattr__(self, '_coefficients',
			tuple(tuple(float(x) for x in row) for row in matrix)
			+ (tuple(float(x) for x in weights), tuple(float(x) for x in nodes)))

	@property
	def stage_count(self) -> int:
		return int(self.B.size)

	def __reduce__(self) -> tuple[Any, tuple[np.ndarray, np.ndarray, np.ndarray]]:
		"""Restore through validation and immutable buffers after serialization."""
		return type(self), (self.A, self.B, self.C)


__all__ = ['ButcherTableau']
