"""Reusable experiment composition for concise, reproducible notebooks."""

from .abba_comparison import (
	ABBA_METHOD_NAMES,
	ABBAComparisonConfig,
	ABBAComparisonResult,
	ABBARuntimeSummary,
	ABBATrajectoryDifferenceSeries,
	ABBATrajectoryDifferenceSummary,
	run_abba_comparison,
)
from .abba_symplecticity import (
	ABBAProjectionMultiplierOrder,
	ABBASymplecticityConfig,
	ABBASymplecticityResult,
	ABBASymplecticitySummary,
	run_abba_symplecticity_study,
)
from .abba_explicit_symplecticity import (
	ExplicitABBADefectOrder,
	ExplicitABBASymplecticityConfig,
	ExplicitABBASymplecticityResult,
	ExplicitABBASymplecticitySummary,
	run_explicit_abba_symplecticity_study,
)
from .abba_implicit_symplecticity import (
	IMPLICIT_ABBA_FORMULATIONS,
	ImplicitABBA1SymplecticityResult,
	ImplicitABBA2SymplecticityResult,
	ImplicitABBASymplecticityComparison,
	ImplicitABBASymplecticityConfig,
	run_implicit_abba_1_symplecticity_study,
	run_implicit_abba_2_symplecticity_study,
	run_implicit_abba_symplecticity_study,
)
from .abba_semiimplicit_symplecticity import (
	SemiImplicitABBASymplecticityConfig,
	SemiImplicitABBASymplecticityResult,
	SemiImplicitABBASymplecticitySummary,
	run_semiimplicit_abba_symplecticity_study,
)
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
	centered_gc_configuration,
	centered_gc_trajectory,
	centered_square,
	domain_center,
)
from .potentials import RandomPotentialConfig
from .rk4_symplecticity import (
	RK4ConvergenceOrder,
	RK4SymplecticityConfig,
	RK4SymplecticityResult,
	RK4SymplecticitySummary,
	run_rk4_symplecticity_study,
)

__all__ = [
	"ABBA_METHOD_NAMES",
	"ABBAComparisonConfig",
	"ABBAComparisonResult",
	"ABBAProjectionMultiplierOrder",
	"ABBARuntimeSummary",
	"ABBASymplecticityConfig",
	"ABBASymplecticityResult",
	"ABBASymplecticitySummary",
	"ABBATrajectoryDifferenceSeries",
	"ABBATrajectoryDifferenceSummary",
	"AreaComparisonConfig",
	"AreaComparisonResult",
	"AreaStep",
	"ExplicitABBADefectOrder",
	"ExplicitABBASymplecticityConfig",
	"ExplicitABBASymplecticityResult",
	"ExplicitABBASymplecticitySummary",
	"GeneralizedEnergyConfig",
	"GeneralizedEnergyResult",
	"IMPLICIT_ABBA_FORMULATIONS",
	"ImplicitABBA1SymplecticityResult",
	"ImplicitABBA2SymplecticityResult",
	"ImplicitABBASymplecticityComparison",
	"ImplicitABBASymplecticityConfig",
	"RandomPotentialConfig",
	"RK4ConvergenceOrder",
	"RK4SymplecticityConfig",
	"RK4SymplecticityResult",
	"RK4SymplecticitySummary",
	"SemiImplicitABBASymplecticityConfig",
	"SemiImplicitABBASymplecticityResult",
	"SemiImplicitABBASymplecticitySummary",
	"centered_circle",
	"centered_gc_configuration",
	"centered_gc_trajectory",
	"centered_square",
	"domain_center",
	"pi_area_steps",
	"run_area_comparison",
	"run_abba_comparison",
	"run_abba_symplecticity_study",
	"run_generalized_energy_comparison",
	"run_implicit_abba_1_symplecticity_study",
	"run_implicit_abba_2_symplecticity_study",
	"run_implicit_abba_symplecticity_study",
	"run_explicit_abba_symplecticity_study",
	"run_rk4_symplecticity_study",
	"run_semiimplicit_abba_symplecticity_study",
]
