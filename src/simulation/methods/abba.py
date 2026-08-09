"""Compatibility API for the original symmetric-projected ABBA method."""

from __future__ import annotations

from dataclasses import dataclass

from ._projected_abba import (
	_dense_component_major_jacobian,
	_differentiate_stages,
	_evaluate_residual,
	_evaluate_stages,
	_ideal_projected_state_jacobian,
	_integrate_projected_abba,
	_positive_finite,
	_positive_integer,
	_simultaneous_newton_jacobian,
	_simultaneous_residual_blocks,
	_solve_projected_step,
	_solve_simultaneous_projected_step,
)
from .abba_implicit_1 import ImplicitABBA1


@dataclass(frozen=True, slots=True)
class SymmetricProjectedABBA(ImplicitABBA1):
	"""Backward-compatible name for :class:`ImplicitABBA1`."""


__all__ = ["SymmetricProjectedABBA"]
