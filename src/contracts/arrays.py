"""Array backend selection without importing the optional JAX runtime eagerly."""

import sys
from typing import Any

import numpy as np


def uses_jax(*values: Any) -> bool:
	"""Recognize concrete JAX arrays and tracers, including sequence inputs."""
	jax = sys.modules.get("jax")
	if jax is None:
		return False
	for value in values:
		if isinstance(value, (jax.Array, jax.core.Tracer)):
			return True
		if isinstance(value, (tuple, list)) and uses_jax(*value):
			return True
	return False


def array_namespace(*values: Any) -> Any:
	"""Use JAX if any argument belongs to it; otherwise retain NumPy behavior."""
	if uses_jax(*values):
		import jax.numpy as jnp
		return jnp
	return np


__all__ = ["array_namespace", "uses_jax"]
