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

from dynamics.coordinates import GCCoordinates, FCCoordinates


class JaxGCCoordinates(GCCoordinates):
	"""Fixed GC coordinates using JAX for all inherited array operations."""

	__slots__ = ()
	array_namespace: ClassVar[ModuleType] = jnp


class JaxFCCoordinates(FCCoordinates):
	"""Fixed FC coordinates using JAX for all inherited array operations."""

	__slots__ = ()
	array_namespace: ClassVar[ModuleType] = jnp


__all__ = ["JaxGCCoordinates", "JaxFCCoordinates"]
