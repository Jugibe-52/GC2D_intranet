"""Compiled ABBA/BM4 compositions and the canonical Hairer equations."""

from typing import Any

import jax
import jax.numpy as jnp

from dynamics._jax import JaxDynamics
from formulations.gc import _COUPLING_BASE, _COUPLING_COS, _COUPLING_SIN
from methods._jax_common import (
    MethodOptions, newton, broyden, particle_finite_difference,
    solve_particle_blocks, infinity_norm,
)


def coupling_matrix(duration: Any, frequency: float | None) -> Any:
    """Exact mixing with the same canonical coefficient matrices as CPU GC."""
    if frequency is None:
        return jnp.eye(4)
    angle = 2 * frequency * duration
    return (jnp.asarray(_COUPLING_BASE) + jnp.cos(angle) * jnp.asarray(_COUPLING_COS)
            + jnp.sin(angle) * jnp.asarray(_COUPLING_SIN)) / 2


def compose(time: Any, state: Any, step: Any, dynamics: JaxDynamics,
            options: MethodOptions) -> tuple[Any, Any]:
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


def composition_jacobian(trace: Any, dynamics: JaxDynamics, options: MethodOptions) -> Any:
    """Multiply analytic shear tangents in the original signed stage order."""
    times, durations, sources = trace
    n = sources.shape[-1] // 2
    identity = jnp.broadcast_to(jnp.eye(4), (n, 4, 4))
    def stage(total: Any, inputs: Any) -> Any:
        index, time, duration, pair = inputs
        first = duration * dynamics.particle_jacobians(time, pair[0])
        second = duration * dynamics.particle_jacobians(time, pair[1])
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


def energy_increment(trace: Any, dynamics: JaxDynamics) -> Any:
    """Integrate physical kappa, including the doubled-Hamiltonian factor 1/2."""
    times, durations, sources = trace
    rates = jax.vmap(lambda t, pair: jax.vmap(lambda z: dynamics.momentum_rate(t, z))(pair))(times, sources)
    # A scan retains the CPU's sequential addition of signed shear contributions.
    contributions = (durations[:, None, None] * rates).reshape(-1, sources.shape[-1] // 2)
    def add(total: Any, value: Any) -> Any:
        return total + value, None
    total, _ = jax.lax.scan(add, jnp.zeros(contributions.shape[-1]), contributions)
    return total / 2


def extended_step(time: Any, state: Any, step: Any, dynamics: JaxDynamics,
                  options: MethodOptions) -> tuple[Any, Any, dict[str, Any], Any]:
    """Project a complete recipe without changing its residual or solver axis."""
    size, n = state.size, state.size // 2
    tolerance = options.tolerance(state)
    simultaneous = options.projection == 'simultaneous_state_multiplier'
    midpoint = options.name in ('ABBA2Midpoint', 'BM4Midpoint')
    def evaluate(unknown: Any) -> Any:
        mu = unknown[2*size:] if simultaneous else unknown
        mapped, trace = compose(time, jnp.concatenate((state + mu, state - mu)), step, dynamics, options)
        first, second = mapped[:size], mapped[size:]
        if simultaneous:
            residual = jnp.concatenate((unknown[:size] - mu - first,
                                        unknown[size:2*size] + mu - second,
                                        unknown[:size] - unknown[size:2*size]))
        else:
            residual = first - second + 2 * mu
        return residual, (mapped, trace)
    if midpoint:
        mapped, trace = compose(time, jnp.concatenate((state, state)), step, dynamics, options)
        after = (mapped[:size] + mapped[size:]) / 2
        statistics = {'copy_separation_norms': infinity_norm(mapped[:size] - mapped[size:])}
        valid = jnp.array(True)
    else:
        identity = jnp.broadcast_to(jnp.eye(2), (n, 2, 2))
        identity4 = jnp.broadcast_to(jnp.eye(4), (n, 4, 4))
        normal = jnp.concatenate((identity, -identity), axis=1)
        constraint = jnp.swapaxes(normal, -1, -2)
        def correction(unknown: Any, residual: Any, payload: Any) -> Any:
            _, trace = payload
            if options.jacobian == 'analytic':
                base = composition_jacobian(trace, dynamics, options)
            else:
                mu = unknown[2*size:] if simultaneous else unknown
                initial = jnp.concatenate((state + mu, state - mu))
                base = particle_finite_difference(lambda z: compose(time, z, step, dynamics, options)[0],
                                                 initial, 4, options.relative_step)
            if simultaneous:
                matrix = jnp.concatenate((
                    jnp.concatenate((identity4, -(identity4 + base) @ normal), axis=2),
                    jnp.concatenate((constraint, jnp.zeros((n, 2, 2))), axis=2)), axis=1)
            else:
                matrix = base[:, :2, :2] - base[:, :2, 2:] - base[:, 2:, :2] + base[:, 2:, 2:] + 2 * identity
            return solve_particle_blocks(matrix, -residual)
        initial = jnp.zeros(size)
        cached = None
        if simultaneous:
            mapped, trace = compose(time, jnp.concatenate((state, state)), step, dynamics, options)
            initial = jnp.concatenate((mapped, initial))
            cached = (jnp.concatenate((jnp.zeros(2*size), mapped[:size] - mapped[size:])), (mapped, trace))
        if options.solver == 'broyden':
            if simultaneous:
                ident = jnp.eye(size)
                norm = jnp.concatenate((ident, -ident), axis=0)
                matrix = jnp.block([[jnp.eye(2*size), -2*norm], [norm.T, jnp.zeros((size, size))]])
            else:
                matrix = 4 * jnp.eye(size)
            root = broyden(evaluate, initial, matrix, tolerance, options.max_iterations, initial_evaluation=cached)
        else:
            root = newton(evaluate, initial, correction, tolerance, options.max_iterations, initial_evaluation=cached)
        mapped, trace = root.payload
        mu = root.unknown[2*size:] if simultaneous else root.unknown
        if simultaneous:
            after = (root.unknown[:size] + root.unknown[size:2*size]) / 2
        else:
            after = ((mapped[:size] + mu) + (mapped[size:] - mu)) / 2
        statistics = {'nonlinear_iterations': root.iterations, 'residual_evaluations': root.evaluations,
                      'nonlinear_residual_norms': root.norm, 'nonlinear_tolerances': tolerance,
                      'projection_multiplier_norms': infinity_norm(mu)}
        if options.name in ('ABBA4Implicit', 'ABBA6Implicit'):
            statistics.update({f'substep_{key}': jnp.reshape(value, (1,)) for key, value in statistics.items()})
        valid = root.converged
    increment = energy_increment(trace, dynamics) if options.track_energy else jnp.zeros(n)
    return after, increment, statistics, valid


__all__: list[str] = []
