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

__all__ = [
	"AreaComparisonConfig",
	"AreaComparisonResult",
	"AreaStep",
	"GeneralizedEnergyConfig",
	"GeneralizedEnergyResult",
	"RandomPotentialConfig",
	"animate_gc_area",
	"animate_gc_area_comparison",
	"animate_gc_area_solution",
	"centered_circle",
	"centered_gc_trajectory",
	"centered_square",
	"display_animation",
	"domain_center",
	"pi_area_steps",
	"run_area_comparison",
	"run_generalized_energy_comparison",
]
