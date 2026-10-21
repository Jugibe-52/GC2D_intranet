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
    """Traverse signed adjoint/direct stages and retain their two shear sources.

    The trace consists of (stage_times, durations, sources). Sources have shape
    (stages, 2, 2*N), in execution order, matching CPU energy/Jacobian records.
    """
    size = state.size // 2
    def stage(carry: Any, inputs: Any) -> Any:
        current, clock = carry
        index, coefficient = inputs
        duration = coefficient * step
        direct = index % 2 == 1
        evaluation_time = jnp.where(direct, clock + duration, clock)
        matrix = coupling_matrix(duration, options.coupling)
        def couple(z: Any) -> Any:
            if options.coupling is None:
                return z
            return (matrix @ z.reshape(4, -1)).reshape(-1)
        def forward(z: Any) -> Any:
            first, second = z[:size], z[size:]
            source1 = first
            second = second + duration * dynamics.vector_field(evaluation_time, first)
            source2 = second
            first = first + duration * dynamics.vector_field(evaluation_time, second)
            return couple(jnp.concatenate((first, second))), jnp.stack((source1, source2))
        def adjoint(z: Any) -> Any:
            z = couple(z)
            first, second = z[:size], z[size:]
            source1 = second
            first = first + duration * dynamics.vector_field(evaluation_time, second)
            source2 = first
            second = second + duration * dynamics.vector_field(evaluation_time, first)
            return jnp.concatenate((first, second)), jnp.stack((source1, source2))
        current, sources = jax.lax.cond(direct, forward, adjoint, current)
        return (current, clock + duration), (evaluation_time, duration, sources)
    (after, _), trace = jax.lax.scan(stage, (state, time),
        (jnp.arange(len(options.coefficients)), jnp.asarray(options.coefficients)))
    return after, trace


def composition_jacobian(trace: Any, dynamics: HamiltonianSystem, options: CompositionOptions) -> Any:
    """Multiply analytic shear tangents in the original signed stage order."""
    if not isinstance(dynamics, GuidingCenterJacobianSystem):
        raise TypeError("Analytic particle Jacobians require GC dynamics.")
    times, durations, sources = trace
    n = sources.shape[-1] // 2
    identity = jnp.broadcast_to(jnp.eye(4), (n, 4, 4))
    def stage(total: Any, inputs: Any) -> Any:
        index, time, duration, pair = inputs
        first = duration * dynamics.particle_vector_field_jacobians(time, pair[0])
        second = duration * dynamics.particle_vector_field_jacobians(time, pair[1])
        coupling = coupling_matrix(duration, options.coupling)
        def forward() -> Any:
            a = identity.at[:, 2:, :2].set(first)
            b = identity.at[:, :2, 2:].set(second)
            return coupling @ b @ a
        def adjoint() -> Any:
            a = identity.at[:, :2, 2:].set(first)
            b = identity.at[:, 2:, :2].set(second)
            return b @ a @ coupling
        factor = jax.lax.cond(index % 2 == 1, forward, adjoint)
        return factor @ total, None
    total, _ = jax.lax.scan(stage, identity, (jnp.arange(times.size), times, durations, sources))
    return total


def energy_increment(trace: Any, dynamics: HamiltonianSystem) -> Any:
    """Integrate physical kappa, including the doubled-Hamiltonian factor 1/2."""
    times, durations, sources = trace
    rates = jax.vmap(lambda t, pair: jax.vmap(lambda z: dynamics.extended_momentum_derivative(t, z))(pair))(times, sources)
    # A scan retains the CPU's sequential addition of signed shear contributions.
    contributions = (durations[:, None, None] * rates).reshape(-1, sources.shape[-1] // 2)
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
    increment = energy_increment(trace, dynamics) if options.track_energy else jnp.zeros(n)
    return after, increment, statistics, valid


__all__: list[str] = []
