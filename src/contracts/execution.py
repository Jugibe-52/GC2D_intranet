"""Execution choices independent of physical and numerical parameters."""

from dataclasses import dataclass
from numbers import Integral
from typing import Literal


@dataclass(frozen=True, slots=True)
class Execution:
    """Select the library and device for field evaluation and integration.

    JAX supports every built-in method with its existing GC/FC compatibility.
    Fixed methods run on the device; DOP853/Radau retain SciPy adaptive control and
    transfer their batched field evaluations to the selected JAX device.
    This contract does not create CPU worker processes. A SciPy CPU
    is the host as a whole; JAX device indices address its detected devices.
    """

    backend: Literal["scipy", "jax"] = "scipy"
    device: Literal["cpu", "gpu"] = "cpu"
    device_index: int = 0

    def __post_init__(self) -> None:
        """Reject unsupported combinations without probing optional hardware."""
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


__all__ = ["Execution"]
