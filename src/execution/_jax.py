"""Optional JAX device preparation shared by fixed and hybrid integrations."""

from functools import lru_cache
from types import ModuleType
from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np

from execution.options import ExecutionOptions
from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics
from dynamics.jax_dynamics import JaxFullCyclotronDynamics, JaxGuidingCenterDynamics
from contracts.dynamics.protocols import DynamicalSystem, HamiltonianSystem


if TYPE_CHECKING:
	from jax import Device


def _import_jax() -> ModuleType:
	"""Keep optional-dependency errors actionable for direct preparation calls."""
	try:
		import jax
	except ImportError as exc:
		raise ImportError(
			"JAX integration requires the optional 'jax' extra. "
			"Install with: python -m pip install -e '.[jax]'"
		) from exc
	return jax


def check_precision() -> None:
	"""Require explicit float64 opt-in without changing global JAX settings."""
	jax: ModuleType = _import_jax()

	if not jax.config.read("jax_enable_x64"):
		raise RuntimeError(
			"JAX potential evaluation requires float64. Enable it before use: "
			"jax.config.update('jax_enable_x64', True)"
		)


def resolve_device(execution: ExecutionOptions) -> "Device":
	"""Validate current precision and resolve the requested integration device."""
	jax: ModuleType = _import_jax()

	check_precision()
	try:
		devices: list[Device] = jax.devices(execution.device)
	except RuntimeError as exc:
		raise RuntimeError(f"Requested JAX device '{execution.device}' is unavailable.") from exc
	if execution.device_index >= len(devices):
		raise ValueError(
			f"JAX device index {execution.device_index} is unavailable for '{execution.device}'."
		)
	return devices[execution.device_index]


def require_builtin_dynamics(dynamics: DynamicalSystem) -> GuidingCenterDynamics | FullCyclotronDynamics:
	"""Keep the original equations and refuse unknown subclass overrides."""
	if (type(dynamics) is GuidingCenterDynamics or type(dynamics) is FullCyclotronDynamics
			or type(dynamics) is JaxGuidingCenterDynamics or type(dynamics) is JaxFullCyclotronDynamics):
		return dynamics
	raise TypeError("JAX execution requires built-in GuidingCenterDynamics or FullCyclotronDynamics, or their explicit JAX classes.")


def prepare_dynamics(dynamics: DynamicalSystem) -> JaxGuidingCenterDynamics | JaxFullCyclotronDynamics:
	"""Delegate physical preparation to the supported component itself."""
	return require_builtin_dynamics(dynamics)._as_jax()


class _HostGC:
	"""Expose compiled physical equations through SciPy's NumPy boundary."""

	state_dimension: ClassVar[int] = 2

	def __init__(self, dynamics: GuidingCenterDynamics | FullCyclotronDynamics, device: "Device") -> None:
		self.dynamics = dynamics
		self.device = device
		jax: ModuleType = _import_jax()
		self._field = jax.jit(dynamics.vector_field)
		self._energy = jax.jit(dynamics.hamiltonian)
		self._momentum = jax.jit(dynamics.extended_momentum_derivative)

	def _evaluate(self, function: Any, time: Any, state: Any) -> np.ndarray:
		"""Place host arguments explicitly and synchronize the SciPy result."""
		check_precision()
		jax: ModuleType = _import_jax()
		inputs: tuple[Any, ...] = tuple(jax.device_put(np.asarray(value, dtype=np.float64), self.device)
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
def _host(dynamics: GuidingCenterDynamics | FullCyclotronDynamics, device: "Device") -> _HostGC:
	"""Bound reuse of compiled host callbacks by physical object and device."""
	return (_HostGC if dynamics.state_dimension == 2 else _HostFC)(dynamics, device)


def bind_host_dynamics(dynamics: GuidingCenterDynamics | FullCyclotronDynamics, device: "Device") -> HamiltonianSystem:
	"""Reuse compiled ordinary equations, checking precision on every call."""
	return _host(dynamics, device)


__all__: list[str] = []
