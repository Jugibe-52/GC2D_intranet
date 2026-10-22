"""Exact particle-block tangents of shared direct/adjoint compositions."""
from __future__ import annotations

from collections.abc import Callable
import numpy as np
from dynamics import GuidingCenterJacobianSystem
from formulations.gc import GCDoubledMaps, gc_coupling_matrix
from methods.extended.core.records import CompositionTrace


def particle_jacobians(maps: GCDoubledMaps, trace: CompositionTrace) -> np.ndarray:
    """Return ``(N, 4, 4)`` tangents in particle order ``(ux, uy, vx, vy)``.

    Shear sources were retained by the residual evaluation. Differentiation
    therefore evaluates only field Jacobians, without replaying spatial maps.
    """
    dynamics = maps.dynamics
    if not isinstance(dynamics, GuidingCenterJacobianSystem):
        raise TypeError("Analytic composition Jacobians require planar GC Jacobians.")
    count = maps.particle_count
    identity: np.ndarray = np.broadcast_to(np.eye(4), (count, 4, 4))
    total: np.ndarray = identity.copy()
    for stage in trace.operations:
        if stage.operation == 'T':
            continue
        if stage.operation == 'C':
            if maps.coupling_frequency is None:
                raise ValueError('A C tangent requires a configured coupling rate.')
            factor: np.ndarray = gc_coupling_matrix(stage.duration, maps.coupling_frequency)
        else:
            if stage.source is None:
                raise ValueError('A spatial shear must retain its source for differentiation.')
            field = stage.duration * _checked_vector_field_jacobian(dynamics, stage.time, stage.source)
            factor = identity.copy()
            if stage.operation == 'A':
                factor[:, :2, 2:] = field
            else:
                factor[:, 2:, :2] = field
        total = factor @ total
    if not np.all(np.isfinite(total)):
        raise ValueError("The composition Jacobian became non-finite.")
    return np.asarray(total)


def packed_jacobian(blocks: np.ndarray) -> np.ndarray:
    """Embed independent particle blocks in the global component-major layout."""
    count: int
    dimension: int
    count, dimension, _ = blocks.shape
    result: np.ndarray = np.zeros((dimension * count, dimension * count))
    for particle in range(count):
        indices = particle + count * np.arange(dimension)
        result[np.ix_(indices, indices)] = blocks[particle]
    return result


def central_difference_jacobian(
    map_state: Callable[[np.ndarray], np.ndarray], state: np.ndarray, *, relative_step: float,
) -> np.ndarray:
    """Differentiate the complete map with scale-aware centered differences."""
    value: np.ndarray = np.asarray(state, dtype=float)
    result: np.ndarray = np.empty((value.size, value.size))
    for column in range(value.size):
        delta = relative_step * max(1, abs(float(value[column])))
        perturbation: np.ndarray = np.zeros_like(value)
        perturbation[column] = delta
        forward: np.ndarray
        backward: np.ndarray
        forward, backward = map_state(value + perturbation), map_state(value - perturbation)
        if forward.shape != value.shape or backward.shape != value.shape:
            raise ValueError("The differentiated map changed the state shape.")
        result[:, column] = (forward - backward) / (2 * delta)
    if not np.all(np.isfinite(result)):
        raise ValueError("The map Jacobian became non-finite.")
    return result


def _checked_vector_field_jacobian(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate batched exact GC Jacobians with one matrix per particle."""
	result: np.ndarray = np.asarray(
		dynamics.particle_vector_field_jacobians(t, state),
		dtype=float,
	)
	particle_count = state.size // dynamics.state_dimension
	expected_shape = (particle_count, dynamics.state_dimension, dynamics.state_dimension)
	if result.shape != expected_shape or not np.all(np.isfinite(result)):
		raise ValueError(
			"The GC vector-field Jacobian changed shape or became non-finite."
		)
	return result
