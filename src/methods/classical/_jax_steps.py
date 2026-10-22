"""Device kernels for classical explicit and implicit Runge--Kutta methods.

Tableaux, predictors, residual scaling and passive energy weights follow the
canonical CPU methods. Independent particle linear systems are solved in batch.
"""

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp

from dynamics.protocols import HamiltonianSystem, GuidingCenterJacobianSystem
from solvers._jax import newton, particle_finite_difference, solve_particle_blocks
from methods._jax_options import ExplicitOptions, ImplicitOptions, NonlinearOptions
from methods.classical._rk4_core import physical_step, momentum_increment
from methods.classical.gauss_legendre import _GAUSS_MATRIX, _GAUSS_NODES
from methods.classical.sdirk import SDIRK4_TABLEAU_A, SDIRK4_TABLEAU_B, SDIRK4_TABLEAU_C
from methods.hbvm.order4 import _HBVM42_NODES, _HBVM42_WEIGHTS, _HBVM42_LEGENDRE, _HBVM42_INTEGRALS


if TYPE_CHECKING:
    from solvers._jax import Root


def field_jacobian(dynamics: HamiltonianSystem, time: Any, state: Any, options: NonlinearOptions) -> Any:
    """Honor analytic or finite-difference selection without a dense N² matrix."""
    if options.jacobian == 'analytic':
        if not isinstance(dynamics, GuidingCenterJacobianSystem):
            raise TypeError("Analytic particle Jacobians require GC dynamics.")
        return dynamics.particle_vector_field_jacobians(time, state)
    return particle_finite_difference(lambda z: dynamics.vector_field(time, z), state,
                                      dynamics.state_dimension, options.relative_step)


def coupled_matrix(jacobians: Any, coefficients: Any, step: Any) -> Any:
    """Assemble one coupled-stage Newton matrix per independent particle."""
    count: int
    dimension: int
    count, dimension = jacobians.shape[1:3]
    identity: jax.Array = jnp.broadcast_to(jnp.eye(dimension), (count, dimension, dimension))
    return jnp.concatenate([
        jnp.concatenate([(identity if row == col else 0.) - step * coefficients[row, col] * jacobians[col]
                         for col in range(coefficients.shape[1])], axis=-1)
        for row in range(coefficients.shape[0])], axis=-2)


def gauss_step(time: Any, state: Any, step: Any, dynamics: HamiltonianSystem,
               options: ImplicitOptions) -> tuple[Any, Any, dict[str, Any], Any]:
    """Two coupled collocation stages and their accepted physical quadrature."""
    tolerance: jax.Array = options.solve.tolerance(state)
    nodes: jax.Array = jnp.asarray(_GAUSS_NODES)
    matrix: jax.Array = jnp.asarray(_GAUSS_MATRIX)
    guess = state + step * nodes[:, None] * dynamics.vector_field(time, state)
    def evaluate(stages: Any) -> Any:
        fields: jax.Array = jax.vmap(dynamics.vector_field)(time + step * nodes, stages)
        residual = stages - state - step * (matrix @ fields)
        return residual, fields
    def correction(stages: Any, residual: Any, fields: Any) -> Any:
        jacobians: jax.Array = jax.vmap(lambda t, z: field_jacobian(dynamics, t, z, options.solve))(time + step * nodes, stages)
        return solve_particle_blocks(coupled_matrix(jacobians, matrix, step), -residual)
    root: Root = newton(evaluate, guess, correction, tolerance, options.solve.max_iterations)
    after = state + step * .5 * (root.payload[0] + root.payload[1])
    increment: jax.Array = jnp.zeros(state.size // dynamics.state_dimension)
    if options.track_energy:
        rates: jax.Array = jax.vmap(dynamics.extended_momentum_derivative)(time + step * nodes, root.unknown)
        increment = step * .5 * (rates[0] + rates[1])
    return after, increment, {
        'nonlinear_iterations': root.iterations, 'residual_evaluations': root.evaluations,
        'nonlinear_residual_norms': root.norm, 'nonlinear_tolerances': tolerance,
    }, root.converged


def sdirk_step(time: Any, state: Any, step: Any, dynamics: HamiltonianSystem,
               options: ImplicitOptions) -> tuple[Any, Any, dict[str, Any], Any]:
    """Five sequential S54b solves with batched particle Newton corrections."""
    tolerance: jax.Array = options.solve.tolerance(state)
    a: jax.Array
    b: jax.Array
    c: jax.Array
    a, b, c = map(jnp.asarray, (SDIRK4_TABLEAU_A, SDIRK4_TABLEAU_B, SDIRK4_TABLEAU_C))
    initial_field: jax.Array = dynamics.vector_field(time, state)
    count = state.size // dynamics.state_dimension
    identity: jax.Array = jnp.broadcast_to(jnp.eye(dynamics.state_dimension), (count, dynamics.state_dimension, dynamics.state_dimension))
    def stage(carry: Any, index: Any) -> Any:
        fields: jax.Array
        valid: jax.Array
        fields, valid = carry
        rhs = state + step * jnp.sum(a[index, :, None] * fields, axis=0)
        stage_time = time + step * c[index]
        guess = state + step * c[index] * initial_field
        def evaluate(z: Any) -> Any:
            field: jax.Array = dynamics.vector_field(stage_time, z)
            return z - rhs - step * .25 * field, field
        def correction(z: Any, residual: Any, field: Any) -> Any:
            matrix = identity - step * .25 * field_jacobian(dynamics, stage_time, z, options.solve)
            return solve_particle_blocks(matrix, -residual)
        root: Root = newton(evaluate, guess, correction, tolerance, options.solve.max_iterations)
        fields = fields.at[index].set(root.payload)
        return (fields, valid & root.converged), (root.unknown, root.iterations, root.evaluations, root.norm)
    fields: jax.Array
    valid: jax.Array
    stages: jax.Array
    iterations: jax.Array
    evaluations: jax.Array
    norms: jax.Array
    (fields, valid), (stages, iterations, evaluations, norms) = jax.lax.scan(
        stage, (jnp.zeros((5, state.size)), jnp.array(True)), jnp.arange(5))
    after = state + step * sum(b[i] * fields[i] for i in range(5))
    increment: jax.Array = jnp.zeros(count)
    if options.track_energy:
        rates: jax.Array = jax.vmap(dynamics.extended_momentum_derivative)(time + step * c, stages)
        increment = step * sum(b[i] * rates[i] for i in range(5))
    return after, increment, {
        'nonlinear_iterations': jnp.sum(iterations), 'residual_evaluations': jnp.sum(evaluations),
        'nonlinear_residual_norms': jnp.max(norms), 'nonlinear_tolerances': tolerance,
        'stage_nonlinear_iterations': iterations, 'stage_residual_evaluations': evaluations,
        'stage_nonlinear_residual_norms': norms,
    }, valid


def hbvm_step(time: Any, state: Any, step: Any, dynamics: HamiltonianSystem,
              options: ImplicitOptions) -> tuple[Any, Any, dict[str, Any], Any]:
    """Rank-two Legendre solve, preserving HBVM's matrix norm and damping."""
    nodes: jax.Array
    weights: jax.Array
    legendre: jax.Array
    integrals: jax.Array
    nodes, weights, legendre, integrals = map(jnp.asarray, (
        _HBVM42_NODES, _HBVM42_WEIGHTS, _HBVM42_LEGENDRE, _HBVM42_INTEGRALS))
    stage_times = time + step * nodes
    predicted = state + step * nodes[:, None] * dynamics.vector_field(time, state)
    fields: jax.Array = jax.vmap(dynamics.vector_field)(stage_times, predicted)
    guess = legendre.T @ (weights[:, None] * fields)
    tolerance: jax.Array = options.solve.tolerance(state)
    def evaluate(coefficients: Any) -> Any:
        stages = state + step * (integrals @ coefficients)
        fields: jax.Array = jax.vmap(dynamics.vector_field)(stage_times, stages)
        projected = legendre.T @ (weights[:, None] * fields)
        return coefficients - projected, (stages, projected)
    def correction(coefficients: Any, residual: Any, payload: Any) -> Any:
        stages: jax.Array
        stages, _ = payload
        jacobians: jax.Array = jax.vmap(lambda t, z: field_jacobian(dynamics, t, z, options.solve))(stage_times, stages)
        identity: jax.Array = jnp.broadcast_to(jnp.eye(dynamics.state_dimension), jacobians.shape[1:])
        rows = []
        for i in range(2):
            row = []
            for j in range(2):
                weighted: jax.Array | int = sum(weights[k] * legendre[k, i] * integrals[k, j] * jacobians[k] for k in range(4))
                row.append((identity if i == j else 0.) - step * weighted)
            rows.append(jnp.concatenate(row, axis=-1))
        return solve_particle_blocks(jnp.concatenate(rows, axis=-2), -residual)
    # CPU HBVM uses ||R||_infinity on its (2, state_size) coefficient matrix.
    root: Root = newton(evaluate, guess, correction, tolerance, options.solve.max_iterations,
                  norm=lambda r: jnp.max(jnp.sum(jnp.abs(r), axis=1)), backtracking=True)
    stages: jax.Array
    projected: jax.Array
    stages, projected = root.payload
    after = state + step * projected[0]
    increment: jax.Array = jnp.zeros(state.size // dynamics.state_dimension)
    if options.track_energy:
        rates: jax.Array = jax.vmap(dynamics.extended_momentum_derivative)(stage_times, stages)
        increment = step * sum(weights[k] * rates[k] for k in range(4))
    fd_calls = 8 * dynamics.state_dimension * root.iterations if options.solve.jacobian == 'finite_difference' else 0
    return after, increment, {
        'nonlinear_iterations': root.iterations, 'nonlinear_residual_norms': root.norm,
        'nonlinear_tolerances': tolerance, 'residual_evaluations': root.evaluations,
        'residual_evaluations_per_step': root.evaluations,
        'jacobian_evaluations_per_step': root.iterations,
        'vector_field_evaluations_per_step': 5 + 4 * root.evaluations + fd_calls,
    }, root.converged


def rk4_step(time: Any, state: Any, step: Any, dynamics: HamiltonianSystem,
             options: ExplicitOptions) -> tuple[Any, Any, dict[str, Any], Any]:
    """Apply the shared RK4 map and its optional passive quadrature."""
    after: jax.Array
    stages: tuple[Any, ...]
    after, stages = physical_step(dynamics.vector_field, time, state, step)
    increment: Any | jax.Array = (momentum_increment(dynamics.extended_momentum_derivative, time, step, stages)
                 if options.track_energy else jnp.zeros(state.size // dynamics.state_dimension))
    return after, increment, {}, jnp.array(True)


def euler_step(time: Any, state: Any, step: Any, dynamics: HamiltonianSystem,
               options: ExplicitOptions) -> tuple[Any, Any, dict[str, Any], Any]:
    """Apply the physical Euler map with the same initial-stage energy rate."""
    after = state + step * dynamics.vector_field(time, state)
    increment: Any | jax.Array = (step * dynamics.extended_momentum_derivative(time, state)
                 if options.track_energy else jnp.zeros(state.size // dynamics.state_dimension))
    return after, increment, {}, jnp.array(True)


__all__: list[str] = []
