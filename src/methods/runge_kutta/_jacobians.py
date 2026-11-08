"""Validated field functions and derivative layouts shared by numerical methods."""

from __future__ import annotations

import numpy as np
from contracts.arrays import BackendArray
from types import ModuleType
from contracts.arrays import array_namespace, checked_finite

from contracts.methods.functions import VectorField
from solvers.derivatives import centered_difference


def _checked_vector_field(f: VectorField, time: float, state: BackendArray) -> BackendArray:
	"""Evaluate a finite field without changing physical state shape."""
	xp: ModuleType = array_namespace(state)
	result: BackendArray = xp.asarray(f(time, state), dtype=float)
	result = checked_finite(result, "The vector field must be finite and preserve state shape.", ValueError)
	if result.shape != state.shape:
		raise ValueError("The vector field must be finite and preserve state shape.")
	return result


def _dense_finite_difference_jacobian(
	f: VectorField, time: float, state: BackendArray, *, relative_step: float,
) -> BackendArray:
	"""Differentiate the supplied field, retaining coupled coordinates."""
	return centered_difference(lambda value: _checked_vector_field(f, time, value),
		state, relative_step=relative_step)


def _checked_analytic_jacobians(
	jacobian: VectorField, time: float, state: BackendArray,
) -> BackendArray:
	"""Validate independent particle derivatives in component-major packing."""
	xp: ModuleType = array_namespace(state)
	result: BackendArray = xp.asarray(jacobian(time, state), dtype=float)
	valid = (
		result.ndim == 3 and result.shape[0] > 0 and result.shape[1] > 0
		and result.shape[1] == result.shape[2]
		and result.shape[0] * result.shape[1] == state.size
	)
	result = checked_finite(result, 'The analytic Jacobian must be finite and match its declared structure.', ValueError)
	if not valid:
		raise ValueError('The analytic Jacobian must be finite and match its declared structure.')
	return result
