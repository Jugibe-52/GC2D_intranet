"""Shared differentiation and packing without assumptions about physical fields."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np


JacobianMethod: TypeAlias = Literal["auto", "analytic", "finite_difference"]
ResolvedJacobianMethod: TypeAlias = Literal["analytic", "finite_difference"]


def centered_difference(
	function: Callable[[np.ndarray], np.ndarray], state: np.ndarray, *, relative_step: float,
) -> np.ndarray:
	"""Differentiate a square map with scale-aware centered differences.

	The callable may wrap field/residual counters or additional validation.
	Every coordinate is probed separately, including coupled particle fields.
	"""
	value: np.ndarray = np.asarray(state, dtype=float)
	result: np.ndarray = np.empty((value.size, value.size), dtype=float)
	for column in range(value.size):
		delta = relative_step * max(1.0, abs(float(value[column])))
		perturbation: np.ndarray = np.zeros_like(value)
		perturbation[column] = delta
		forward: np.ndarray = np.asarray(function(value + perturbation), dtype=float)
		backward: np.ndarray = np.asarray(function(value - perturbation), dtype=float)
		if forward.shape != value.shape or backward.shape != value.shape:
			raise ValueError("The differentiated function must preserve the state shape.")
		result[:, column] = (forward - backward) / (2.0 * delta)
	if not np.all(np.isfinite(result)):
		raise ValueError("The finite-difference Jacobian must be finite.")
	return result


def packed_particle_jacobian(blocks: np.ndarray) -> np.ndarray:
	"""Embed independent ``(N,d,d)`` blocks in component-major coordinates."""
	count: int
	components: int
	count, components, _ = blocks.shape
	result: np.ndarray = np.zeros((components * count, components * count))
	for particle in range(count):
		indices = particle + count * np.arange(components)
		result[np.ix_(indices, indices)] = blocks[particle]
	return result


__all__: list[str] = []
