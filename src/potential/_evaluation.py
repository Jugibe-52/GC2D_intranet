"""Shared evaluation contracts, validation, and backend-independent physics."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .grid import Grid
from .prepared import prepared_source


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


class PotentialEvaluator(ABC):
    """Common field API; concrete evaluators own interpolation and placement."""

    grid: Grid
    interpolation_order: int
    namespace: Any

    def __init__(self, source: Any) -> None:
        """Bind both backends to the same prepared data contract."""
        self.prepared = prepared_source(source)
        self.grid = self.prepared.grid
        self.interpolation_order = self.prepared.interpolation_order

    def check_ready(self) -> None:
        """Allow optional runtimes to enforce their precision configuration."""

    @abstractmethod
    def asarray(self, value: Any) -> Any:
        """Normalize one input to the evaluator's array type and device."""

    @abstractmethod
    def _evaluate(self, time: Any, x: Any, y: Any, dx: int, dy: int, dt: int) -> Any:
        """Evaluate validated coordinates and static derivative orders."""

    @abstractmethod
    def _evaluate_grid(self, time: Any, dt: int) -> Any:
        """Reconstruct validated time samples on the stored spatial grid."""

    def evaluate(self, t: Any, x: Any, y: Any, *, dx: int = 0, dy: int = 0, dt: int = 0) -> Any:
        """Evaluate values or derivatives at matching paired coordinates."""
        self.check_ready()
        validate_derivatives(self.interpolation_order, dx, dy, dt)
        time, x, y = self.asarray(t), self.asarray(x), self.asarray(y)
        if x.shape != y.shape:
            raise ValueError("`x` and `y` must have the same shape.")
        return self._evaluate(time, x, y, int(dx), int(dy), int(dt))

    def evaluate_grid(self, t: Any, *, dt: int = 0) -> Any:
        """Return stored grid values with time axes after the spatial axes."""
        self.check_ready()
        validate_derivatives(self.interpolation_order, 0, 0, dt)
        return self._evaluate_grid(self.asarray(t), int(dt))

    def electric_field(self, t: Any, x: Any = None, y: Any = None) -> tuple[Any, Any]:
        """Return ``(-phi_x, -phi_y)`` at paired points or on the full grid."""
        self.check_ready()
        if x is None and y is None:
            x, y = self.namespace.meshgrid(self.asarray(self.grid.x), self.asarray(self.grid.y), indexing="ij")
        elif x is None or y is None:
            raise ValueError("`x` and `y` must be provided together.")
        return -self.evaluate(t, x, y, dx=1), -self.evaluate(t, x, y, dy=1)


__all__: list[str] = []
