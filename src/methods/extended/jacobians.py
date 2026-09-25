"""Exact particle-block tangents of shared direct/adjoint compositions."""
from __future__ import annotations

from collections.abc import Callable
import numpy as np
from dynamics import GuidingCenterJacobianSystem
from formulations.gc import GCDoubledMaps, gc_coupling_matrix
from methods.extended.records import CompositionTrace


def particle_jacobians(maps: GCDoubledMaps, trace: CompositionTrace) -> np.ndarray:
    """Return ``(N, 4, 4)`` tangents in particle order ``(ux, uy, vx, vy)``.

    Shear sources were retained by the residual evaluation. Differentiation
    therefore evaluates only field Jacobians, without replaying spatial maps.
    """
    dynamics = maps.dynamics
    if not isinstance(dynamics, GuidingCenterJacobianSystem):
        raise TypeError("Analytic composition Jacobians require planar GC Jacobians.")
    count = maps.particle_count
    identity = np.broadcast_to(np.eye(4), (count, 4, 4))
    total = identity.copy()
    for stage in trace.stages:
        factors = []
        for _, duration, source in stage.energy_points:
            field = np.asarray(dynamics.particle_vector_field_jacobians(stage.time, source), dtype=float)
            if field.shape != (count, 2, 2) or not np.all(np.isfinite(field)):
                raise ValueError("The GC vector-field Jacobian changed shape or became non-finite.")
            factors.append(duration * field)
        if len(factors) != 2:
            raise ValueError("A GC stage must retain two shear sources for differentiation.")
        first, second = identity.copy(), identity.copy()
        coupling = np.eye(4) if maps.coupling_frequency is None else gc_coupling_matrix(stage.duration, maps.coupling_frequency)
        if stage.direct:
            first[:, 2:, :2], second[:, :2, 2:] = factors
            factor = coupling @ second @ first
        else:
            first[:, :2, 2:], second[:, 2:, :2] = factors
            factor = second @ first @ coupling
        total = factor @ total
    if not np.all(np.isfinite(total)):
        raise ValueError("The composition Jacobian became non-finite.")
    return np.asarray(total)


def packed_jacobian(blocks: np.ndarray) -> np.ndarray:
    """Embed independent particle blocks in the global component-major layout."""
    count, dimension, _ = blocks.shape
    result = np.zeros((dimension * count, dimension * count))
    for particle in range(count):
        indices = particle + count * np.arange(dimension)
        result[np.ix_(indices, indices)] = blocks[particle]
    return result


def central_difference_jacobian(
    map_state: Callable[[np.ndarray], np.ndarray], state: np.ndarray, *, relative_step: float,
) -> np.ndarray:
    """Differentiate the complete map with scale-aware centered differences."""
    value = np.asarray(state, dtype=float)
    result = np.empty((value.size, value.size))
    for column in range(value.size):
        delta = relative_step * max(1, abs(float(value[column])))
        perturbation = np.zeros_like(value)
        perturbation[column] = delta
        forward, backward = map_state(value + perturbation), map_state(value - perturbation)
        if forward.shape != value.shape or backward.shape != value.shape:
            raise ValueError("The differentiated map changed the state shape.")
        result[:, column] = (forward - backward) / (2 * delta)
    if not np.all(np.isfinite(result)):
        raise ValueError("The map Jacobian became non-finite.")
    return result
