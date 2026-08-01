"""Reusable experiment composition for concise, reproducible notebooks."""

from .area_comparison import (
	AreaComparisonConfig,
	AreaComparisonResult,
	AreaStep,
	pi_area_steps,
	run_area_comparison,
)
from .energy import (
	GeneralizedEnergyConfig,
	GeneralizedEnergyResult,
	run_generalized_energy_comparison,
)
from .fc_visualization import animate_fc_particle_solution, animate_gc_particle_solution
from .initial_conditions import (
	centered_circle,
	centered_gc_trajectory,
	centered_square,
	domain_center,
)
from .gc_visualization import (
	animate_gc_area,
	animate_gc_area_comparison,
	animate_gc_area_solution,
)
from .notebooks import display_animation
from .potentials import RandomPotentialConfig
from .rk4_symplecticity import (
	RK4ConvergenceOrder,
	RK4SymplecticityConfig,
	RK4SymplecticityResult,
	RK4SymplecticitySummary,
	run_rk4_symplecticity_study,
)

__all__ = [
	"AreaComparisonConfig",
	"AreaComparisonResult",
	"AreaStep",
	"GeneralizedEnergyConfig",
	"GeneralizedEnergyResult",
	"RandomPotentialConfig",
	"RK4ConvergenceOrder",
	"RK4SymplecticityConfig",
	"RK4SymplecticityResult",
	"RK4SymplecticitySummary",
	"animate_gc_area",
	"animate_gc_area_comparison",
	"animate_gc_area_solution",
	"animate_fc_particle_solution",
	"animate_gc_particle_solution",
	"centered_circle",
	"centered_gc_trajectory",
	"centered_square",
	"display_animation",
	"domain_center",
	"pi_area_steps",
	"run_area_comparison",
	"run_generalized_energy_comparison",
	"run_rk4_symplecticity_study",
]
