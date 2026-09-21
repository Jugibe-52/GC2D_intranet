"""Compatibility imports for the shared simultaneous projection formulation."""
from .projection_simultaneous import (
    _particle_blocks as _particle_blocks,
    _packed_particle_blocks as _packed_particle_blocks,
    _simultaneous_residual_blocks as _simultaneous_residual_blocks,
    _simultaneous_newton_jacobian as _simultaneous_newton_jacobian,
    _solve_simultaneous_state_multiplier_step as _solve_simultaneous_state_multiplier_step,
)

__all__: list[str] = []
