"""Optional Matplotlib presentation for potentials and simulation results."""

from .gc_area import (
	animate_gc_area,
	animate_gc_area_comparison,
	animate_gc_area_solution,
)
from .notebooks import display_animation
from .particles import animate_fc_particle_solution, animate_gc_particle_solution
from .potential import animate_potential, plot_potential

__all__ = [
	"animate_fc_particle_solution",
	"animate_gc_area",
	"animate_gc_area_comparison",
	"animate_gc_area_solution",
	"animate_gc_particle_solution",
	"animate_potential",
	"display_animation",
	"plot_potential",
]
