"""Shared evaluation contracts, validation, and backend-independent physics."""

from typing import Any

import numpy as np



def validate_derivatives(degree: int, dx: int, dy: int, dt: int) -> None:
    """Validate the derivative orders supported by the prepared interpolants."""
    if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, (int, np.integer)) or dt not in (0, 1, 2):
        raise ValueError("`dt` must be 0, 1, or 2.")
    for derivative, name in ((dx, "dx"), (dy, "dy")):
        if (isinstance(derivative, (bool, np.bool_))
                or not isinstance(derivative, (int, np.integer)) or derivative < 0):
            raise ValueError(f"`{name}` must be a non-negative integer.")
        if derivative >= degree:
            raise ValueError(f"`{name}` must be at most {degree - 1} for interpolation order {degree}.")


def normalize_coordinates(x: Any, y: Any, origin: Any, period: Any) -> tuple[Any, Any]:
    """Wrap array coordinates without converting their storage or backend."""
    return (x - origin[0]) % period + origin[0], (y - origin[1]) % period + origin[1]


def reconstruct(time: Any, fields: Any, frequencies: Any, dt: int, *, xp: Any) -> Any:
    """Reconstruct one real field from a leading mean/complex-mode axis.

    ``fields`` has shape ``(1 + modes, *coordinate_shape)`` and time broadcasts
    against ``coordinate_shape``. Sequential mode addition preserves the CPU
    summation convention and is unrolled for a fixed mode count under JAX JIT.
    The same formula differentiates time by multiplying each harmonic phase.
    """
    shape = np.broadcast_shapes(fields.shape[1:], time.shape)
    result = xp.zeros(shape, dtype=float)
    if dt == 0:
        result = result + xp.real(fields[0])
    for index in range(frequencies.shape[0]):
        angular = 2.0 * np.pi * frequencies[index]
        phase = xp.exp(1j * angular * time) * (1j * angular) ** dt
        result = result + 2.0 * xp.real(fields[index + 1] * phase)
    # NumPy scalar arithmetic may collapse a zero-dimensional array to a scalar;
    # retain the public array contract for scalar coordinates as well.
    return xp.asarray(result)
