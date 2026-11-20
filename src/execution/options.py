"""Immutable backend choices independent of the executor and physical data."""

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    """Select the library and device for field evaluation and integration.

    JAX supports every built-in method with its existing GC/FC compatibility.
    Methods run complete fixed-step loops on the selected device.
    Progress reporting belongs to host execution.
    This contract does not create CPU worker processes. A SciPy CPU
    is the host as a whole; JAX device indices address its detected devices.
    """

    backend: Literal["scipy", "jax"] = "scipy"
    device: Literal["cpu", "gpu"] = "cpu"
    device_index: int = 0
    progress: bool = False
    record_step_statistics: bool = True

    def __setstate__(self, state: list[Any]) -> None:
        """Restore earlier four-field Modal payloads with recording enabled."""
        names = ('backend', 'device', 'device_index', 'progress', 'record_step_statistics')
        if len(state) not in (4, 5):
            raise ValueError('Unsupported serialized execution options.')
        for name, value in zip(names, [*state, True] if len(state) == 4 else state):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Reject unsupported combinations without probing optional hardware."""
        if not isinstance(self.record_step_statistics, bool):
            raise TypeError("`record_step_statistics` must be a boolean.")
        if self.backend not in ("scipy", "jax"):
            raise ValueError("`backend` must be 'scipy' or 'jax'.")
        if self.device not in ("cpu", "gpu"):
            raise ValueError("`device` must be 'cpu' or 'gpu'.")
        if (isinstance(self.device_index, bool)
                or not isinstance(self.device_index, Integral) or self.device_index < 0):
            raise ValueError("`device_index` must be a non-negative integer.")
        object.__setattr__(self, "device_index", int(self.device_index))
        if self.backend == "scipy" and (self.device != "cpu" or self.device_index != 0):
            raise ValueError("SciPy evaluation requires device='cpu' and device_index=0.")


__all__ = ["ExecutionOptions"]
