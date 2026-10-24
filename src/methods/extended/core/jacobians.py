"""Exact tangents of shared direct/adjoint compositions from supplied derivatives."""
from __future__ import annotations

import numpy as np
from methods.extended.maps import GCDoubledMaps, gc_coupling_matrix
from methods.extended.core.records import CompositionTrace
from solvers.derivatives import packed_particle_jacobian


def _checked_vector_field_jacobian(maps: GCDoubledMaps, t: float, state: np.ndarray) -> np.ndarray:
    """Validate an explicit dense derivative or independent particle blocks."""
    if maps.jacobian is None:
        raise TypeError('Analytic composition derivatives require a Jacobian function.')
    result: np.ndarray = np.asarray(maps.jacobian(t, state), dtype=float)
    expected = ((maps.particle_count, 2, 2) if maps.jacobian_structure == 'particle'
                else (maps.physical_size, maps.physical_size))
    if result.shape != expected or not np.all(np.isfinite(result)):
        raise ValueError(f'The vector-field Jacobian must be finite and have shape {expected}.')
    return result


def composition_jacobian(maps: GCDoubledMaps, trace: CompositionTrace) -> np.ndarray:
    """Differentiate retained shear inputs without replaying spatial maps.

    Independent particles retain batched 4x4 tangents in (ux,uy,vx,vy) order.
    Dense derivatives retain cross-particle terms in component-major packing.
    """
    particle = maps.jacobian_structure == 'particle'
    size = 2 if particle else maps.physical_size
    identity: np.ndarray = np.eye(2 * size)
    if particle:
        identity = np.broadcast_to(identity, (maps.particle_count, 4, 4))
    total: np.ndarray = identity.copy()
    for stage in trace.operations:
        if stage.operation == 'T':
            continue
        if stage.operation == 'C':
            if maps.coupling_frequency is None:
                raise ValueError('A C tangent requires a configured coupling rate.')
            factor: np.ndarray = gc_coupling_matrix(stage.duration, maps.coupling_frequency)
            if not particle:
                factor = packed_particle_jacobian(np.broadcast_to(factor, (maps.particle_count, 4, 4)))
        else:
            if stage.source is None:
                raise ValueError('A spatial shear must retain its source for differentiation.')
            derivative: np.ndarray = stage.duration * _checked_vector_field_jacobian(maps, stage.time, stage.source)
            factor = identity.copy()
            if stage.operation == 'A':
                factor[..., :size, size:] = derivative
            else:
                factor[..., size:, :size] = derivative
        total = factor @ total
    if not np.all(np.isfinite(total)):
        raise ValueError('The composition Jacobian became non-finite.')
    return np.asarray(total)
