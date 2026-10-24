"""Backend-independent classical RK4 physical stages."""

from collections.abc import Callable
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from contracts.arrays import BackendArray


def physical_step(field: Callable[..., Any], time: Any, state: Any,
                  step: Any) -> tuple[Any, tuple[Any, ...]]:
    """Advance component-major particle arrays without changing array backend."""
    k1: BackendArray = field(time, state)
    z2 = state + step * k1 / 2
    k2: BackendArray = field(time + step / 2, z2)
    z3 = state + step * k2 / 2
    k3: BackendArray = field(time + step / 2, z3)
    z4 = state + step * k3
    k4: BackendArray = field(time + step, z4)
    return state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6, (state, z2, z3, z4)



__all__: list[str] = []
