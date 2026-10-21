"""Compiled ABBA/BM4 compositions and the canonical Hairer equations."""

from typing import Any

import jax
import jax.numpy as jnp

from dynamics.protocols import HamiltonianSystem, GuidingCenterJacobianSystem
from formulations.gc import _COUPLING_BASE, _COUPLING_COS, _COUPLING_SIN
from formulations.projection import resolve_projection
from methods._jax_options import CompositionOptions
from solvers._jax import (
    newton, broyden, particle_finite_difference,
    solve_particle_blocks, infinity_norm,
)


def coupling_matrix(duration: Any, frequency: float | None) -> Any:
    """Exact mixing with the same canonical coefficient matrices as CPU GC."""
    if frequency is None:
        return jnp.eye(4)
    angle = 2 * frequency * duration
    return (jnp.asarray(_COUPLING_BASE) + jnp.cos(angle) * jnp.asarray(_COUPLING_COS)
            + jnp.sin(angle) * jnp.asarray(_COUPLING_SIN)) / 2


def compose(time: Any, state: Any, step: Any, dynamics: HamiltonianSystem,
            options: CompositionOptions) -> tuple[Any, Any]:
    """Execute the same explicit A/B/T/C recipe as NumPy inside one scan.

    The trace holds one time, signed duration and source per operation. Only
    spatial shear entries are used for energy quadrature; no field is sampled
    by T or C. The clock changes exclusively in the T branch.
    """
    size = state.size // 2
    codes = tuple(('A', 'B', 'T', 'C').index(op.name) for op in options.recipe.operations)
    weights = tuple(op.coefficient for op in options.recipe.operations)
    def operation(carry: Any, inputs: Any) -> Any:
        current, clock = carry
        code, coefficient = inputs
        duration = coefficient * step
        empty = jnp.zeros(size)
        def shear_a() -> Any:
            source = current[size:]
            first = current[:size] + duration * dynamics.vector_field(clock, source)
            return jnp.concatenate((first, source)), clock, source
        def shear_b() -> Any:
            source = current[:size]
            second = current[size:] + duration * dynamics.vector_field(clock, source)
            return jnp.concatenate((source, second)), clock, source
        def translate() -> Any:
            return current, clock + duration, empty
        def couple() -> Any:
            matrix = coupling_matrix(duration, options.coupling)
            return (matrix @ current.reshape(4, -1)).reshape(-1), clock, empty
        after, new_clock, source = jax.lax.switch(code, (shear_a, shear_b, translate, couple))
        return (after, new_clock), (clock, duration, source)
    (after, _), trace = jax.lax.scan(operation, (state, time), (jnp.asarray(codes), jnp.asarray(weights)))
    return after, trace


def composition_jacobian(trace: Any, dynamics: HamiltonianSystem, options: CompositionOptions) -> Any:
    """Differentiate each elementary flow in its actual execution order."""
    if not isinstance(dynamics, GuidingCenterJacobianSystem):
        raise TypeError('Analytic particle Jacobians require GC dynamics.')
    times, durations, sources = trace
    n = sources.shape[-1] // 2
    identity = jnp.broadcast_to(jnp.eye(4), (n, 4, 4))
    codes = jnp.asarray(tuple(('A', 'B', 'T', 'C').index(op.name) for op in options.recipe.operations))
    def operation(total: Any, inputs: Any) -> Any:
        code, time, duration, source = inputs
        def shear_a() -> Any:
            field = duration * dynamics.particle_vector_field_jacobians(time, source)
            return identity.at[:, :2, 2:].set(field)
        def shear_b() -> Any:
            field = duration * dynamics.particle_vector_field_jacobians(time, source)
            return identity.at[:, 2:, :2].set(field)
        def translate() -> Any:
            return identity
        def couple() -> Any:
            return jnp.broadcast_to(coupling_matrix(duration, options.coupling), identity.shape)
        factor = jax.lax.switch(code, (shear_a, shear_b, translate, couple))
        return factor @ total, None
    total, _ = jax.lax.scan(operation, identity, (codes, times, durations, sources))
    return total


def energy_increment(trace: Any, dynamics: HamiltonianSystem, options: CompositionOptions) -> Any:
    """Accumulate accepted A/B sources with the physical momentum factor 1/2."""
    times, durations, sources = trace
    indices = jnp.asarray(tuple(i for i, op in enumerate(options.recipe.operations) if op.name in ('A', 'B')))
    times, durations, sources = times[indices], durations[indices], sources[indices]
    rates = jax.vmap(dynamics.extended_momentum_derivative)(times, sources)
    contributions = durations[:, None] * rates
    def add(total: Any, value: Any) -> Any:
        return total + value, None
    total, _ = jax.lax.scan(add, jnp.zeros(contributions.shape[-1]), contributions)
    return total / 2


def extended_step(time: Any, state: Any, step: Any, dynamics: HamiltonianSystem,
                  options: CompositionOptions) -> tuple[Any, Any, dict[str, Any], Any]:
    """Project a complete recipe without changing its residual or solver axis."""
    size, n = state.size, state.size // 2
    equations = resolve_projection(options.projection)
    def evaluate(unknown: Any) -> Any:
        mu = equations.multiplier(unknown, size)
        mapped, trace = compose(time, jnp.concatenate((state + mu, state - mu)), step, dynamics, options)
        residual = equations.residual(unknown, mapped, size)
        return residual, (mapped, trace)
    if options.solve is None:
        mapped, trace = compose(time, jnp.concatenate((state, state)), step, dynamics, options)
        after = (mapped[:size] + mapped[size:]) / 2
        statistics = {'copy_separation_norms': infinity_norm(mapped[:size] - mapped[size:])}
        valid = jnp.array(True)
    else:
        solve = options.solve
        tolerance = solve.tolerance(state)
        def correction(unknown: Any, residual: Any, payload: Any) -> Any:
            _, trace = payload
            if solve.jacobian == 'analytic':
                base = composition_jacobian(trace, dynamics, options)
            else:
                mu = equations.multiplier(unknown, size)
                initial = jnp.concatenate((state + mu, state - mu))
                base = particle_finite_difference(lambda z: compose(time, z, step, dynamics, options)[0],
                                                 initial, 4, solve.relative_step)
            matrix = equations.particle_jacobian(base)
            return solve_particle_blocks(matrix, -residual)
        def initial_map(candidate: Any) -> Any:
            mapped, trace = compose(time, candidate, step, dynamics, options)
            return mapped, (mapped, trace)
        initial, cached = equations.prepare(state, initial_map)
        if solve.solver == 'broyden':
            matrix = equations.initial_jacobian(state)
            root = broyden(evaluate, initial, matrix, tolerance, solve.max_iterations, initial_evaluation=cached)
        else:
            root = newton(evaluate, initial, correction, tolerance, solve.max_iterations, initial_evaluation=cached)
        mapped, trace = root.payload
        mu = equations.multiplier(root.unknown, size)
        after = equations.reconstruct(root.unknown, mapped, size)
        statistics = {'nonlinear_iterations': root.iterations, 'residual_evaluations': root.evaluations,
                      'nonlinear_residual_norms': root.norm, 'nonlinear_tolerances': tolerance,
                      'projection_multiplier_norms': infinity_norm(mu)}
        if options.publish_substeps:
            statistics.update({f'substep_{key}': jnp.reshape(value, (1,)) for key, value in statistics.items()})
        valid = root.converged
    increment = energy_increment(trace, dynamics, options) if options.track_energy else jnp.zeros(n)
    return after, increment, statistics, valid


__all__: list[str] = []
