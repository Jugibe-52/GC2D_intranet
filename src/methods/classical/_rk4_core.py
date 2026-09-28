"""Backend-independent classical RK4 stages and passive quadrature."""

from collections.abc import Callable
from typing import Any


def physical_step(field: Callable[..., Any], time: Any, state: Any,
                  step: Any) -> tuple[Any, tuple[Any, ...]]:
    """Advance component-major particle arrays without changing array backend."""
    k1 = field(time, state)
    z2 = state + step * k1 / 2
    k2 = field(time + step / 2, z2)
    z3 = state + step * k2 / 2
    k3 = field(time + step / 2, z3)
    z4 = state + step * k3
    k4 = field(time + step, z4)
    return state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6, (state, z2, z3, z4)


def momentum_increment(rate: Callable[..., Any], time: Any, step: Any,
                       stages: tuple[Any, ...]) -> Any:
    """Integrate -partial_t H with the accepted physical RK4 stage states."""
    rates = [rate(time + c * step, z) for c, z in zip((0., .5, .5, 1.), stages)]
    return step * (rates[0] + 2 * rates[1] + 2 * rates[2] + rates[3]) / 6


__all__: list[str] = []
