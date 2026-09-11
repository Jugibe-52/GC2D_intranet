"""Compatibility imports for the shared reduced projection formulation."""
from .projection_reduced import (
    _evaluate_stages as _evaluate_stages,
    _evaluate_residual as _evaluate_residual,
    _solve_reduced_multiplier_step as _solve_reduced_multiplier_step,
)

__all__: list[str] = []
