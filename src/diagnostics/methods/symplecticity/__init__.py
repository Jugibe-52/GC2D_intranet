"""Numerical Jacobian observations for GC symplecticity studies."""

from diagnostics.methods.symplecticity.observer import (
	OutputBlock,
	SymplecticityObserver,
	SymplecticityRecord,
	gc_extended_symplectic_form,
	gc_physical_symplectic_form,
	gc_reduced_time_extended_symplectic_form,
	gc_time_extended_symplectic_form,
)
from diagnostics.methods.symplecticity.time_extended import (
	GCReducedTimeExtendedSymplecticityObserver,
	GCReducedTimeExtendedSymplecticityRecord,
	GCTimeExtendedSymplecticityObserver,
	GCTimeExtendedSymplecticityRecord,
)
from diagnostics.methods.jacobians import (
	STEP_JACOBIAN_METHODS,
	StepJacobianMethod,
	calculate_step_jacobian,
	central_difference_jacobian,
	gauss_legendre4_step_jacobian,
	implicit_function_step_jacobian,
	stage_increment_step_jacobian,
)
from diagnostics.methods.symplecticity.area import (
	GCAreaSymplecticityObserver,
	GCAreaSymplecticityOutputBlock,
	GCAreaSymplecticityRecord,
)
from diagnostics.io.paths import notebook_output_directory

__all__ = [
	"OutputBlock",
	"STEP_JACOBIAN_METHODS",
	"StepJacobianMethod",
	"GCAreaSymplecticityObserver",
	"GCAreaSymplecticityOutputBlock",
	"GCAreaSymplecticityRecord",
	"GCTimeExtendedSymplecticityObserver",
	"GCTimeExtendedSymplecticityRecord",
	"GCReducedTimeExtendedSymplecticityObserver",
	"GCReducedTimeExtendedSymplecticityRecord",
	"SymplecticityObserver",
	"SymplecticityRecord",
	"calculate_step_jacobian",
	"central_difference_jacobian",
	"gauss_legendre4_step_jacobian",
	"gc_extended_symplectic_form",
	"gc_physical_symplectic_form",
	"gc_reduced_time_extended_symplectic_form",
	"gc_time_extended_symplectic_form",
	"implicit_function_step_jacobian",
	"notebook_output_directory",
	"stage_increment_step_jacobian",
]
