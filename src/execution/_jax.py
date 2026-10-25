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
from contracts.methods.base import NumericalMethod
from potential.potential import Potential
from potential.jax_potential import JaxPotential


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


def _jax_potential(potential: Potential) -> JaxPotential:
	"""Share immutable physical data and fitted splines without rebuilding them."""
	if type(potential) is JaxPotential:
		return potential
	if type(potential) is not Potential:
		raise TypeError("JAX execution requires a built-in Potential or JaxPotential; subclass overrides are not compiled.")
	# Bypass construction only for the known base representation. Keep device
	# caches local while sharing read-only samples and already fitted splines.
	prepared: JaxPotential = object.__new__(JaxPotential)
	prepared.__dict__.update(potential.__dict__)
	prepared._jax_buffers = {}
	return prepared


def prepare_dynamics(dynamics: DynamicalSystem) -> JaxGuidingCenterDynamics | JaxFullCyclotronDynamics:
	"""Validate the physical type before entering the identity-based cache."""
	return _prepare_dynamics(require_builtin_dynamics(dynamics))


@lru_cache(maxsize=16)
def _prepare_dynamics(
	physical: GuidingCenterDynamics | FullCyclotronDynamics,
) -> JaxGuidingCenterDynamics | JaxFullCyclotronDynamics:
	"""Bind shared GC/FC equations to JAX potentials without mutating the source."""
	potential: JaxPotential = _jax_potential(physical.potential)
	# Only the supported representation is copied. Calling the GC constructor
	# again would repeat gyroaveraging and fit the effective field a second time.
	prepared: JaxGuidingCenterDynamics | JaxFullCyclotronDynamics
	if isinstance(physical, GuidingCenterDynamics):
		effective: JaxPotential = potential if physical.effective_potential is physical.potential else _jax_potential(physical.effective_potential)
		if isinstance(physical, JaxGuidingCenterDynamics) and potential is physical.potential and effective is physical.effective_potential:
			return physical
		prepared = object.__new__(JaxGuidingCenterDynamics)
		prepared.__dict__.update(physical.__dict__)
		object.__setattr__(prepared, "effective_potential", effective)
	else:
		if isinstance(physical, JaxFullCyclotronDynamics) and potential is physical.potential:
			return physical
		prepared = object.__new__(JaxFullCyclotronDynamics)
		prepared.__dict__.update(physical.__dict__)
	object.__setattr__(prepared, "potential", potential)
	return prepared


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


def prepare_environment(
	dynamics: DynamicalSystem, method: NumericalMethod, options: ExecutionOptions,
) -> tuple[HamiltonianSystem, "Device"]:
	"""Resolve a simulation's device and physical evaluation mode once.

	Fixed maps retain device-native equations. Adaptive SciPy solvers receive
	the same equations through an explicit, cached host boundary.
	"""
	mode = method.jax_mode
	if mode not in ("device_resident", "hybrid_scipy_controller"):
		raise NotImplementedError(f"{type(method).__name__} does not provide JAX execution.")
	device: Device = resolve_device(options)
	prepared: GuidingCenterDynamics | FullCyclotronDynamics = prepare_dynamics(dynamics)
	if mode == "hybrid_scipy_controller":
		from methods._jax_dispatch import validate_hybrid_method
		validate_hybrid_method(method)
		return bind_host_dynamics(prepared, device), device
	return prepared, device


__all__: list[str] = []
