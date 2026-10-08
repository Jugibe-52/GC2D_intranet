"""Compiled evaluation of the existing tensor-product potential splines.

Only the optional JAX evaluation path imports this module. Spline construction stays
in SciPy; these kernels use its knots and coefficients without refitting data.
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from ._evaluation import normalize_coordinates, reconstruct


def _basis(knots: Any, points: Any, degree: int, derivative: int) -> tuple[Any, Any]:
    """Return local coefficient indices and differentiated B-spline weights.

    The last axis contains the ``degree + 1`` basis functions supported at each
    point. The Cox--de Boor recurrence carries derivatives alongside values;
    knot differences are constant, and each linear factor contributes a signed
    multiple of the preceding derivative. No finite differences are used.
    """
    span = jnp.searchsorted(knots, points, side="right") - 1
    span = jnp.clip(span, degree, knots.size - degree - 2)
    weights = [[jnp.ones_like(points)] + [jnp.zeros_like(points)] * derivative]
    for level in range(1, degree + 1):
        saved = [jnp.zeros_like(points)] * (derivative + 1)
        updated = []
        for index in range(level):
            left = points - knots[span + 1 - level + index]
            right = knots[span + index + 1] - points
            denominator = knots[span + index + 1] - knots[span + 1 - level + index]
            # A repeated knot contributes zero, rather than a 0/0 basis term.
            safe = jnp.where(denominator != 0, denominator, 1.0)
            terms = [jnp.where(denominator != 0, value / safe, 0.0)
                     for value in weights[index]]
            updated.append([
                saved[order] + right * terms[order]
                - (order * terms[order - 1] if order else 0.0)
                for order in range(derivative + 1)
            ])
            saved = [left * terms[order]
                     + (order * terms[order - 1] if order else 0.0)
                     for order in range(derivative + 1)]
        weights = updated + [saved]
    indices = span[..., None] - degree + jnp.arange(degree + 1)
    return indices, jnp.stack([value[derivative] for value in weights], axis=-1)


@partial(jax.jit, static_argnames=("degree", "dx", "dy", "dt"))
def evaluate_splines(
    time: Any, x: Any, y: Any, knots_x: Any, knots_y: Any,
    coefficients: Any, frequencies: Any, origin: Any, period: Any,
    *, degree: int, dx: int, dy: int, dt: int,
) -> Any:
    """Evaluate paired periodic coordinates, preserving JAX array placement.

    ``coefficients`` has shape ``(1 + mode_count, ncoeff_x, ncoeff_y)``.
    The mean and all complex modes share the same spatial basis and lookup.
    """
    time, x, y = jnp.broadcast_arrays(time, x, y)
    x, y = normalize_coordinates(x, y, origin, period)
    ix, wx = _basis(knots_x, x, degree, dx)
    iy, wy = _basis(knots_y, y, degree, dy)
    local = coefficients[:, ix[..., :, None], iy[..., None, :]]
    fields = jnp.sum(local * wx[..., :, None] * wy[..., None, :], axis=(-2, -1))
    return reconstruct(time, fields, frequencies, dt, xp=jnp)


__all__: list[str] = []
