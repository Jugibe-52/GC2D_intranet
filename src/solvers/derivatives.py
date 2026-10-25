"""Shared differentiation and packing without assumptions about physical fields."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Literal, TypeAlias

import numpy as np
from typing import cast
from contracts.arrays import BackendArray
from contracts.arrays import array_namespace, checked_finite, uses_jax


JacobianMethod: TypeAlias = Literal["auto", "analytic", "finite_difference"]
ResolvedJacobianMethod: TypeAlias = Literal["analytic", "finite_difference"]


def centered_difference(
	function: Callable[[BackendArray], BackendArray], state: BackendArray, *, relative_step: float,
) -> BackendArray:
	"""Differentiate a square map with scale-aware, independent column probes.

	Every coordinate is probed separately, retaining cross-particle coupling.
	The caller can count evaluations; each column uses exactly two calls.
	"""
	xp: ModuleType = array_namespace(state)
	value: BackendArray = xp.asarray(state, dtype=float)
	basis: BackendArray = xp.eye(value.size)
	def column_derivative(direction: BackendArray) -> BackendArray:
		# One basis direction perturbs one unknown; no particle independence is
		# assumed. Vectorization changes scheduling, not these residual probes.
		delta = relative_step * xp.maximum(1.0, xp.abs(xp.sum(value * direction)))
		perturbation: BackendArray = direction * delta
		forward: BackendArray = xp.asarray(function(value + perturbation), dtype=float)
		backward: BackendArray = xp.asarray(function(value - perturbation), dtype=float)
		if forward.shape != value.shape or backward.shape != value.shape:
			raise ValueError("The differentiated function must preserve the state shape.")
		return cast(BackendArray, (forward - backward) / (2.0 * delta))
	if uses_jax(value):
		from jax import vmap
		result: BackendArray = vmap(column_derivative)(basis).T
	else:
		result = xp.stack([column_derivative(direction) for direction in basis], axis=1)
	return checked_finite(result, "The finite-difference Jacobian must be finite.")


def packed_particle_jacobian(blocks: BackendArray) -> BackendArray:
	"""Embed independent ``(N,d,d)`` blocks in component-major coordinates."""
	xp: ModuleType = array_namespace(blocks)
	count: int
	components: int
	count, components, _ = blocks.shape
	return cast(BackendArray, xp.einsum('nij,nm->injm', blocks, xp.eye(count)).reshape(components * count, components * count))


__all__: list[str] = []
