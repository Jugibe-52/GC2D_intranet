"""Optional device-side nonlinear iteration with shared solver controls."""

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp


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
           backtracking: bool = False, initial_evaluation: Any = None,
           update_returns_correction: bool = True) -> Root:
    """Full Newton with the CPU stopping rule and optional HBVM backtracking.

    Controls are global across the particle batch. Only linear algebra is
    decomposed into independent blocks, so particles share the same iteration
    count and accepted residual criterion as the CPU calculation.
    """
    r: jax.Array
    payload: Any
    r, payload = residual(initial) if initial_evaluation is None else initial_evaluation
    def active(carry: Any) -> Any:
        u: jax.Array
        r: jax.Array
        p: Any
        iteration: jax.Array
        evaluations: jax.Array
        u, r, p, iteration, evaluations = carry
        return (iteration < limit) & (norm(r) > tolerance) & jnp.all(jnp.isfinite(r)) & jnp.all(jnp.isfinite(u))
    def update(carry: Any) -> Any:
        u: jax.Array
        r: jax.Array
        payload: Any
        iteration: jax.Array
        evaluations: jax.Array
        u, r, payload, iteration, evaluations = carry
        proposal: jax.Array = correction(u, r, payload)
        delta: jax.Array = proposal if update_returns_correction else proposal - u
        candidate: jax.Array = u + delta if update_returns_correction or backtracking else proposal
        new_r: jax.Array
        new_payload: Any
        new_r, new_payload = residual(candidate)
        evaluations += 1
        if backtracking:
            def needs_damping(trial: Any) -> Any:
                damping: jax.Array
                candidate: jax.Array
                new_r: jax.Array
                new_payload: Any
                evaluations: jax.Array
                damping, candidate, new_r, new_payload, evaluations = trial
                return ~(norm(new_r) < norm(r)) & (damping > 1. / 128.)
            def damp(trial: Any) -> Any:
                damping: jax.Array
                evaluations: jax.Array
                damping, _, _, _, evaluations = trial
                damping = damping / 2
                candidate = u + damping * delta
                new_r: jax.Array
                new_payload: Any
                new_r, new_payload = residual(candidate)
                return damping, candidate, new_r, new_payload, evaluations + 1
            _, candidate, new_r, new_payload, evaluations = jax.lax.while_loop(
                needs_damping, damp, (jnp.array(1.), candidate, new_r, new_payload, evaluations))
        return candidate, new_r, new_payload, iteration + 1, evaluations
    u: jax.Array
    p: Any
    iterations: jax.Array
    evaluations: jax.Array
    u, r, p, iterations, evaluations = jax.lax.while_loop(
        active, update, (initial, r, payload, jnp.array(0), jnp.array(1)))
    size: jax.Array = norm(r)
    return Root(u, r, p, iterations, evaluations, size,
                (size <= tolerance) & jnp.all(jnp.isfinite(u)))


def broyden(residual: Callable[..., Any], initial: Any, matrix: Any,
            tolerance: Any, limit: int, *, initial_evaluation: Any = None) -> Root:
    """Preserve the global good-Broyden secant update, including cross blocks.

    A per-particle rank-one update would be a different iteration. The full
    matrix is deliberately retained for this explicitly requested solver.
    """
    r: jax.Array
    payload: Any
    r, payload = residual(initial) if initial_evaluation is None else initial_evaluation
    def active(carry: Any) -> Any:
        u: jax.Array
        r: jax.Array
        p: Any
        matrix: jax.Array
        iteration: jax.Array
        valid: jax.Array
        u, r, p, matrix, iteration, valid = carry
        return valid & (iteration < limit) & (infinity_norm(r) > tolerance)
    def update(carry: Any) -> Any:
        u: jax.Array
        r: jax.Array
        p: Any
        matrix: jax.Array
        iteration: jax.Array
        valid: jax.Array
        u, r, p, matrix, iteration, valid = carry
        delta: jax.Array = jnp.linalg.solve(matrix, -r)
        candidate = u + delta
        new_r: jax.Array
        new_p: Any
        new_r, new_p = residual(candidate)
        denominator = delta @ delta
        safe = denominator > jnp.finfo(jnp.float64).tiny
        increment = jnp.outer(new_r - r - matrix @ delta, delta) / jnp.where(safe, denominator, 1.)
        matrix = matrix + jnp.where(safe, increment, 0.)
        valid = jnp.all(jnp.isfinite(new_r)) & jnp.all(jnp.isfinite(candidate)) & (safe | (infinity_norm(new_r) <= tolerance))
        return candidate, new_r, new_p, matrix, iteration + 1, valid
    u: jax.Array
    p: Any
    iterations: jax.Array
    valid: jax.Array
    u, r, p, _, iterations, valid = jax.lax.while_loop(
        active, update, (initial, r, payload, matrix, jnp.array(0),
                         jnp.all(jnp.isfinite(r)) & jnp.all(jnp.isfinite(initial))
                         & jnp.all(jnp.isfinite(matrix))))
    size: jax.Array = infinity_norm(r)
    return Root(u, r, p, iterations, iterations + 1, size, valid & (size <= tolerance))



__all__: list[str] = []
