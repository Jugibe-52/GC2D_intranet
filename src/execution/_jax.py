"""Optional JAX device preparation shared by fixed and hybrid integrations."""

from typing import Any

from contracts.execution_options import ExecutionOptions
from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics
from dynamics.protocols import DynamicalSystem


def _import_jax() -> Any:
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
	jax = _import_jax()

	if not jax.config.read("jax_enable_x64"):
		raise RuntimeError(
			"JAX potential evaluation requires float64. Enable it before use: "
			"jax.config.update('jax_enable_x64', True)"
		)


def resolve_device(execution: ExecutionOptions) -> Any:
	"""Validate current precision and resolve the requested integration device."""
	jax = _import_jax()

	check_precision()
	try:
		devices = jax.devices(execution.device)
	except RuntimeError as exc:
		raise RuntimeError(f"Requested JAX device '{execution.device}' is unavailable.") from exc
	if execution.device_index >= len(devices):
		raise ValueError(
			f"JAX device index {execution.device_index} is unavailable for '{execution.device}'."
		)
	return devices[execution.device_index]


def require_builtin_dynamics(dynamics: DynamicalSystem) -> GuidingCenterDynamics | FullCyclotronDynamics:
	"""Keep the original equations and refuse unknown subclass overrides."""
	if type(dynamics) is GuidingCenterDynamics or type(dynamics) is FullCyclotronDynamics:
		return dynamics
	raise TypeError("JAX execution requires built-in GuidingCenterDynamics or FullCyclotronDynamics.")


__all__: list[str] = []
