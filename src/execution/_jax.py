"""Optional JAX device preparation shared by fixed and hybrid integrations."""

from copy import copy
from functools import lru_cache
from typing import Any

from contracts.execution_options import ExecutionOptions
from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics
from dynamics.protocols import DynamicalSystem
from potential.potential import Potential
from potential.jax_potential import JaxPotential


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


def _jax_potential(potential: Potential) -> JaxPotential:
	"""Share immutable physical data and fitted splines without rebuilding them."""
	if type(potential) is JaxPotential:
		return potential
	if type(potential) is not Potential:
		raise TypeError("JAX execution requires a built-in Potential or JaxPotential; subclass overrides are not compiled.")
	# Bypass construction only for the known base representation. Keep device
	# caches local while sharing read-only samples and already fitted splines.
	prepared = object.__new__(JaxPotential)
	prepared.__dict__.update(potential.__dict__)
	prepared._jax_buffers = {}
	return prepared


def prepare_dynamics(dynamics: DynamicalSystem) -> GuidingCenterDynamics | FullCyclotronDynamics:
	"""Validate the physical type before entering the identity-based cache."""
	return _prepare_dynamics(require_builtin_dynamics(dynamics))


@lru_cache(maxsize=16)
def _prepare_dynamics(
	physical: GuidingCenterDynamics | FullCyclotronDynamics,
) -> GuidingCenterDynamics | FullCyclotronDynamics:
	"""Bind shared GC/FC equations to JAX potentials without mutating the source."""
	potential = _jax_potential(physical.potential)
	prepared: GuidingCenterDynamics | FullCyclotronDynamics
	if isinstance(physical, GuidingCenterDynamics):
		effective = potential if physical.effective_potential is physical.potential else _jax_potential(physical.effective_potential)
		if potential is physical.potential and effective is physical.effective_potential:
			return physical
		prepared = copy(physical)
		object.__setattr__(prepared, "effective_potential", effective)
	else:
		if potential is physical.potential:
			return physical
		prepared = copy(physical)
	object.__setattr__(prepared, "potential", potential)
	return prepared


__all__: list[str] = []
