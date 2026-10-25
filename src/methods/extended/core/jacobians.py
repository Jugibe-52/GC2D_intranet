"""Exact tangents of shared direct/adjoint compositions from supplied derivatives."""
from __future__ import annotations

from typing import Any

import numpy as np
from typing import cast
from contracts.arrays import BackendArray
from types import ModuleType
from contracts.arrays import array_namespace, checked_finite, scan_arrays, select_array_branch
from methods.extended.maps import GCDoubledMaps, gc_coupling_matrix
from methods.extended.core.records import CompositionTrace
from solvers.derivatives import packed_particle_jacobian


def _checked_vector_field_jacobian(maps: GCDoubledMaps, t: float, state: BackendArray) -> BackendArray:
    """Validate an explicit dense derivative or independent particle blocks."""
    xp: ModuleType = array_namespace(state)
    if maps.jacobian is None:
        raise TypeError('Analytic composition derivatives require a Jacobian function.')
    result: BackendArray = xp.asarray(maps.jacobian(t, state), dtype=float)
    expected = ((maps.particle_count, 2, 2) if maps.jacobian_structure == 'particle'
                else (maps.physical_size, maps.physical_size))
    result = checked_finite(result, f'The vector-field Jacobian must be finite and have shape {expected}.', ValueError)
    if result.shape != expected:
        raise ValueError(f'The vector-field Jacobian must be finite and have shape {expected}.')
    return result


def composition_jacobian(maps: GCDoubledMaps, trace: CompositionTrace) -> BackendArray:
    """Differentiate retained shear inputs without replaying spatial maps.

    Independent particles retain batched 4x4 tangents in (ux,uy,vx,vy) order.
    Dense derivatives retain cross-particle terms in component-major packing.
    """
    xp: ModuleType = array_namespace(trace.state)
    particle = maps.jacobian_structure == 'particle'
    size = 2 if particle else maps.physical_size
    identity: BackendArray = xp.eye(2 * size)
    if particle:
        identity = xp.broadcast_to(identity, (maps.particle_count, 4, 4))
    for stage in trace.operations:
        if stage.operation == 'C' and maps.coupling_frequency is None:
            raise ValueError('A C tangent requires a configured coupling rate.')
        if stage.operation in ('A', 'B') and stage.source is None:
            raise ValueError('A spatial shear must retain its source for differentiation.')
    unit: BackendArray = identity[..., :size, :size]
    codes: np.ndarray = np.asarray([('A', 'B', 'T', 'C').index(op.operation) for op in trace.operations])
    times: BackendArray = xp.asarray([op.time for op in trace.operations])
    durations: BackendArray = xp.asarray([op.duration for op in trace.operations])
    sources: BackendArray = xp.stack([op.source if op.source is not None else xp.zeros(maps.physical_size)
                                     for op in trace.operations])

    def advance(total: Any, operation: Any) -> tuple[Any, Any]:
        code: Any
        time: Any
        duration: Any
        source: Any
        code, time, duration, source = operation
        def shear(first: bool) -> BackendArray:
            derivative: BackendArray = duration * _checked_vector_field_jacobian(maps, time, source)
            zero: BackendArray = xp.zeros_like(derivative)
            upper: BackendArray = derivative if first else zero
            lower: BackendArray = zero if first else derivative
            return cast(BackendArray, xp.concatenate((xp.concatenate((unit, upper), axis=-1),
                                      xp.concatenate((lower, unit), axis=-1)), axis=-2))
        def coupling() -> BackendArray:
            if maps.coupling_frequency is None:
                return identity
            factor: BackendArray = gc_coupling_matrix(duration, maps.coupling_frequency)
            if particle:
                return cast(BackendArray, xp.broadcast_to(factor, identity.shape))
            return packed_particle_jacobian(xp.broadcast_to(factor, (maps.particle_count, 4, 4)))
        factor: BackendArray = select_array_branch(code, (lambda: shear(True), lambda: shear(False),
                                                         lambda: identity, coupling))
        return factor @ total, ()

    total: BackendArray
    total, _ = scan_arrays(advance, identity.copy(), (codes, times, durations, sources))
    total = checked_finite(total, 'The composition Jacobian became non-finite.', ValueError)
    return cast(BackendArray, xp.asarray(total))
