"""Momentum updates shared by direct/adjoint numerical formulations."""

from __future__ import annotations


import numpy as np

from contracts.dynamics.protocols import DynamicalSystem, HamiltonianSystem

def _updated_momentum(
	dynamics: DynamicalSystem,
	momentum: np.ndarray | None,
	duration: float,
	t: float,
	state: np.ndarray,
) -> np.ndarray | None:
	"""Update per-particle momentum over one signed split-map duration.

	The physical state uses its formulation's packed coordinates. The momentum
	derivative must preserve the momentum array's shape; None disables tracking.
	"""
	if momentum is None:
		return None
	assert isinstance(dynamics, HamiltonianSystem)
	derivative: np.ndarray = np.asarray(dynamics.extended_momentum_derivative(t, state))
	if derivative.shape != momentum.shape:
		raise ValueError("The extended-momentum derivative changed its shape.")
	return np.asarray(momentum + duration * derivative)
