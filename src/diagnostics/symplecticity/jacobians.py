"""Compatibility exports for Jacobian helpers now shared by diagnostics."""

from diagnostics.jacobians import (
	STEP_JACOBIAN_METHODS,
	StepJacobianMethod,
	_dense_component_major_jacobian,
	calculate_step_jacobian,
	central_difference_jacobian,
	gauss_legendre4_step_jacobian,
	implicit_function_step_jacobian,
	stage_increment_step_jacobian,
)

# Kept outside ``__all__`` for the existing development review notebook that
# intentionally inspects this private block-layout helper.

__all__ = [
	"STEP_JACOBIAN_METHODS",
	"StepJacobianMethod",
	"calculate_step_jacobian",
	"central_difference_jacobian",
	"gauss_legendre4_step_jacobian",
	"implicit_function_step_jacobian",
	"stage_increment_step_jacobian",
]
