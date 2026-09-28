"""Optional device-side nonlinear control and independent particle Jacobians."""

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class MethodOptions:
    """Hashable numerical controls; physical data stays in the dynamics binding."""

    name: str
    track_energy: bool
    copies: int
    atol: float = 0.
    rtol: float = 0.
    max_iterations: int = 1
    jacobian: str = 'analytic'
    relative_step: float = 0.
    solver: str = 'newton'
    projection: str = 'reduced_multiplier'
    coupling: float | None = None
    coefficients: tuple[float, ...] = ()

    def tolerance(self, state: Any) -> Any:
        """Preserve the CPU's global physical-state infinity-norm scaling."""
        return self.atol + self.rtol * jnp.maximum(1., jnp.max(jnp.abs(state)))


class Root(NamedTuple):
    """A solved residual and counters, represented as a JAX pytree."""

    unknown: Any
    residual: Any
    payload: Any
    iterations: Any
    evaluations: Any
    norm: Any
    converged: Any


def infinity_norm(value: Any) -> Any:
    """Maximum absolute packed component (not a matrix row-sum norm)."""
    return jnp.max(jnp.abs(value))


def newton(residual: Callable[..., Any], initial: Any, correction: Callable[..., Any],
           tolerance: Any, limit: int, *, norm: Callable[..., Any] = infinity_norm,
           backtracking: bool = False, initial_evaluation: Any = None) -> Root:
    """Full Newton with the CPU stopping rule and optional HBVM backtracking.

    Controls are global across the particle batch. Only linear algebra is
    decomposed into independent blocks, so particles share the same iteration
    count and accepted residual criterion as the CPU calculation.
    """
    r, payload = residual(initial) if initial_evaluation is None else initial_evaluation
    def active(carry: Any) -> Any:
        u, r, p, iteration, evaluations = carry
        return (iteration < limit) & (norm(r) > tolerance) & jnp.all(jnp.isfinite(r)) & jnp.all(jnp.isfinite(u))
    def update(carry: Any) -> Any:
        u, r, payload, iteration, evaluations = carry
        delta = correction(u, r, payload)
        candidate = u + delta
        new_r, new_payload = residual(candidate)
        evaluations += 1
        if backtracking:
            def needs_damping(trial: Any) -> Any:
                damping, candidate, new_r, new_payload, evaluations = trial
                return ~(norm(new_r) < norm(r)) & (damping > 1. / 128.)
            def damp(trial: Any) -> Any:
                damping, _, _, _, evaluations = trial
                damping = damping / 2
                candidate = u + damping * delta
                new_r, new_payload = residual(candidate)
                return damping, candidate, new_r, new_payload, evaluations + 1
            _, candidate, new_r, new_payload, evaluations = jax.lax.while_loop(
                needs_damping, damp, (jnp.array(1.), candidate, new_r, new_payload, evaluations))
        return candidate, new_r, new_payload, iteration + 1, evaluations
    u, r, p, iterations, evaluations = jax.lax.while_loop(
        active, update, (initial, r, payload, jnp.array(0), jnp.array(1)))
    size = norm(r)
    return Root(u, r, p, iterations, evaluations, size,
                (size <= tolerance) & jnp.all(jnp.isfinite(u)))


def broyden(residual: Callable[..., Any], initial: Any, matrix: Any,
            tolerance: Any, limit: int, *, initial_evaluation: Any = None) -> Root:
    """Preserve the global good-Broyden secant update, including cross blocks.

    A per-particle rank-one update would be a different iteration. The full
    matrix is deliberately retained for this explicitly requested solver.
    """
    r, payload = residual(initial) if initial_evaluation is None else initial_evaluation
    def active(carry: Any) -> Any:
        u, r, p, matrix, iteration, valid = carry
        return valid & (iteration < limit) & (infinity_norm(r) > tolerance)
    def update(carry: Any) -> Any:
        u, r, p, matrix, iteration, valid = carry
        delta = jnp.linalg.solve(matrix, -r)
        candidate = u + delta
        new_r, new_p = residual(candidate)
        denominator = delta @ delta
        safe = denominator > jnp.finfo(jnp.float64).tiny
        increment = jnp.outer(new_r - r - matrix @ delta, delta) / jnp.where(safe, denominator, 1.)
        matrix = matrix + jnp.where(safe, increment, 0.)
        valid = jnp.all(jnp.isfinite(new_r)) & jnp.all(jnp.isfinite(candidate)) & (safe | (infinity_norm(new_r) <= tolerance))
        return candidate, new_r, new_p, matrix, iteration + 1, valid
    u, r, p, _, iterations, valid = jax.lax.while_loop(
        active, update, (initial, r, payload, matrix, jnp.array(0), jnp.all(jnp.isfinite(r))))
    size = infinity_norm(r)
    return Root(u, r, p, iterations, iterations + 1, size, valid & (size <= tolerance))


def particle_finite_difference(function: Callable[..., Any], state: Any,
                               components: int, relative_step: float) -> Any:
    """Differentiate independent particle maps with simultaneous column probes.

    Packed state is (components*N,). Each probe perturbs the same coordinate
    of every particle; independence makes the resulting (N,d,d) blocks exact
    counterparts of the CPU's full finite-difference matrix diagonal blocks.
    """
    blocks = state.reshape(components, -1)
    columns = []
    for column in range(components):
        delta = relative_step * jnp.maximum(1., jnp.abs(blocks[column]))
        perturbation = jnp.zeros_like(blocks).at[column].set(delta).reshape(-1)
        derivative = (function(state + perturbation) - function(state - perturbation)).reshape(components, -1)
        columns.append((derivative / (2 * delta)).T)
    return jnp.stack(columns, axis=-1)


def solve_particle_blocks(matrix: Any, residual: Any) -> Any:
    """Solve (N,d,d) systems, retaining component-major residual packing."""
    count = matrix.shape[0]
    rhs = residual.reshape(-1, count).T
    return jnp.linalg.solve(matrix, rhs[..., None])[..., 0].T.reshape(residual.shape)


__all__: list[str] = []
