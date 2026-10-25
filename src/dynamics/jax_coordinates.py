"""Explicit JAX arrays for the shared physical coordinate operations."""

from types import ModuleType
from typing import ClassVar

try:
	import jax.numpy as jnp
except ImportError as exc:
	raise ImportError(
		"JAX coordinates require the optional 'jax' extra. "
		"Install with: python -m pip install -e '.[jax]'"
	) from exc

from dynamics.coordinates import PhysicalCoordinates


class JaxPhysicalCoordinates(PhysicalCoordinates):
	"""Reuse immutable physical layouts with JAX conversion and packing.

	Import this module explicitly to load optional JAX. The array module is
	fixed at class definition; names, roles and component operations are inherited.
	"""

	__slots__ = ()

	array_namespace: ClassVar[ModuleType] = jnp


__all__ = ["JaxPhysicalCoordinates"]
