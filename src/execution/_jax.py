"""Optional JAX device preparation for fixed-step integrations."""

from types import ModuleType
from typing import TYPE_CHECKING


from execution.options import ExecutionOptions
from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics
from dynamics.jax_dynamics import JaxFullCyclotronDynamics, JaxGuidingCenterDynamics
from contracts.dynamics.protocols import DynamicalSystem


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


__all__: list[str] = []
