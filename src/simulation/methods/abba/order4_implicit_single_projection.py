"""Compatibility imports for the physical outer-projection numerical helpers.

The public configuration is ABBA4Implicit(projection_placement=
"around_complete_composition"); its integration uses the shared family runtime.
"""
from .projection_outer import (
    _SingleProjectionBaseEvaluation as _SingleProjectionBaseEvaluation,
    _SingleProjectionResidualEvaluation as _SingleProjectionResidualEvaluation,
    _ABBA4SingleProjectionStep as _ABBA4SingleProjectionStep,
    _validated_state as _validated_state,
    _evaluate_single_projection_base as _evaluate_single_projection_base,
    _differentiate_single_projection_base as _differentiate_single_projection_base,
    _evaluate_single_projection_residual as _evaluate_single_projection_residual,
    _projected_state as _projected_state,
    _solve_reduced_abba4_single_projection_step as _solve_reduced_abba4_single_projection_step,
    _simultaneous_residual_blocks as _simultaneous_residual_blocks,
    _simultaneous_newton_jacobian as _simultaneous_newton_jacobian,
    _solve_simultaneous_abba4_single_projection_step as _solve_simultaneous_abba4_single_projection_step,
    _solve_abba4_single_projection_step as _solve_abba4_single_projection_step,
)

__all__: list[str] = []
