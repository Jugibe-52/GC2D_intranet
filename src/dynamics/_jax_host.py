"""Explicit NumPy/JAX boundary for SciPy's adaptive integration algorithms.

SciPy owns adaptive acceptance, Newton/LU and dense interpolation. The ordinary
physical dynamics evaluates every particle on the selected JAX device. Transfers
at every evaluation are intentional and reported as hybrid execution.
"""

from functools import lru_cache
from typing import Any, ClassVar

import jax
import numpy as np

from contracts.execution_options import ExecutionOptions
from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics
from dynamics.protocols import DynamicalSystem
from execution._jax import check_precision, prepare_dynamics, resolve_device


class _HostGC:
	"""Expose compiled physical equations through SciPy's NumPy boundary."""

	state_dimension: ClassVar[int] = 2

	def __init__(self, dynamics: GuidingCenterDynamics | FullCyclotronDynamics, device: Any) -> None:
		self.dynamics = dynamics
		self.device = device
		self._field = jax.jit(dynamics.vector_field)
		self._energy = jax.jit(dynamics.hamiltonian)
		self._momentum = jax.jit(dynamics.extended_momentum_derivative)

	def _evaluate(self, function: Any, time: Any, state: Any) -> np.ndarray:
		"""Place host arguments explicitly and synchronize the SciPy result."""
		check_precision()
		inputs = tuple(jax.device_put(np.asarray(value, dtype=np.float64), self.device)
			for value in (time, state))
		return np.asarray(function(*inputs))

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
def _host(dynamics: GuidingCenterDynamics | FullCyclotronDynamics, device: Any) -> _HostGC:
	"""Bound reuse of compiled host callbacks by physical object and device."""
	return (_HostGC if dynamics.state_dimension == 2 else _HostFC)(dynamics, device)


def bind_host_dynamics(dynamics: DynamicalSystem, execution: ExecutionOptions) -> DynamicalSystem:
	"""Reuse compiled ordinary equations, checking precision on every call."""
	physical = prepare_dynamics(dynamics)
	return _host(physical, resolve_device(execution))


__all__: list[str] = []
