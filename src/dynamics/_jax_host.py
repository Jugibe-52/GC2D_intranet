"""Explicit NumPy/JAX boundary for SciPy's adaptive integration algorithms.

SciPy owns adaptive acceptance, Newton/LU and dense interpolation. Field and
energy evaluation batch every particle on the selected JAX device. Transfers
at every evaluation are intentional and reported as hybrid execution.
"""

from functools import lru_cache
from typing import Any, ClassVar

import jax
import numpy as np

from contracts.execution import Execution
from dynamics.protocols import DynamicalSystem
from dynamics._jax import JaxDynamics, bind_dynamics


class _HostGC:
    """Expose compiled field batches through the existing NumPy contracts."""

    state_dimension: ClassVar[int] = 2

    def __init__(self, dynamics: JaxDynamics) -> None:
        self.dynamics = dynamics
        self._field = jax.jit(dynamics.vector_field)
        self._energy = jax.jit(dynamics.hamiltonian)
        self._momentum = jax.jit(dynamics.momentum_rate)

    def _evaluate(self, function: Any, time: Any, state: Any) -> np.ndarray:
        """Synchronize the device result before handing it back to SciPy."""
        evaluator = self.dynamics.evaluator
        evaluator.check_ready()
        return np.asarray(function(evaluator.asarray(time), evaluator.asarray(state)))

    def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
        return self._evaluate(self._field, t, state)

    def hamiltonian(self, t: float | np.ndarray, state: np.ndarray) -> np.ndarray:
        return self._evaluate(self._energy, t, state)

    def extended_momentum_derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        return self._evaluate(self._momentum, t, state)


class _HostFC(_HostGC):
    """The same adapter with the full-cyclotron component count."""

    state_dimension: ClassVar[int] = 4


@lru_cache(maxsize=16)
def _host(dynamics: JaxDynamics) -> _HostGC:
    return (_HostGC if dynamics.dimension == 2 else _HostFC)(dynamics)


def bind_host_dynamics(dynamics: DynamicalSystem, execution: Execution) -> DynamicalSystem:
    """Reuse compiled callables across runs, checking precision on every call."""
    bound = bind_dynamics(dynamics, execution)
    bound.evaluator.check_ready()
    return _host(bound)


__all__: list[str] = []
