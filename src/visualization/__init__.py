"""Optional Matplotlib presentation for potentials and simulation results."""

from .abba_jacobian import (
	plot_implicit_abba_particle_step_series,
	plot_implicit_abba_jacobian_directions,
	plot_implicit_abba_jacobian_matrices,
	plot_implicit_abba_jacobian_polar_snapshots,
	plot_implicit_abba_jacobian_spectrum,
)
from .gc_area import (
	animate_gc_area,
	animate_gc_area_comparison,
	animate_gc_area_solution,
)
from .implicit_iterations import (
	plot_implicit_abba_iteration_comparison,
	plot_implicit_abba_iteration_diagnostics,
	plot_implicit_bm4_iteration_comparison,
	plot_implicit_bm4_iteration_diagnostics,
	plot_implicit_iteration_comparison,
	plot_implicit_iteration_diagnostics,
)
from .implicit_comparison import (
	IMPLICIT_METHOD_COLORS,
	animate_implicit_method_trajectories,
	plot_implicit_method_iterations,
	plot_implicit_trajectory_differences,
)
from .notebooks import display_animation, display_records_table, records_table_html
from .particles import animate_fc_particle_solution, animate_gc_particle_solution
from .potential import animate_potential, plot_potential

__all__ = [
	"animate_fc_particle_solution",
	"animate_gc_area",
	"animate_gc_area_comparison",
	"animate_gc_area_solution",
	"animate_gc_particle_solution",
	"animate_implicit_method_trajectories",
	"animate_potential",
	"display_animation",
	"display_records_table",
	"IMPLICIT_METHOD_COLORS",
	"plot_implicit_abba_jacobian_directions",
	"plot_implicit_abba_iteration_diagnostics",
	"plot_implicit_abba_iteration_comparison",
	"plot_implicit_bm4_iteration_comparison",
	"plot_implicit_bm4_iteration_diagnostics",
	"plot_implicit_iteration_comparison",
	"plot_implicit_iteration_diagnostics",
	"plot_implicit_method_iterations",
	"plot_implicit_trajectory_differences",
	"plot_implicit_abba_jacobian_matrices",
	"plot_implicit_abba_jacobian_polar_snapshots",
	"plot_implicit_abba_jacobian_spectrum",
	"plot_implicit_abba_particle_step_series",
	"plot_potential",
	"records_table_html",
]
