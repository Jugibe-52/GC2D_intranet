"""Apply the shared NumPy/JAX array compatibility contract.

NumPy-only values select NumPy; any JAX array or tracer selects JAX, including
mixed components and nested lists or tuples. These helpers identify array
representations without selecting devices or preparing an execution environment.
The NumPy path does not import the optional JAX runtime.
"""

import sys
from typing import Any, TYPE_CHECKING, TypeAlias, Union

import numpy as np


if TYPE_CHECKING:
	from jax import Array
	from types import ModuleType


# Preserve the concrete numerical backend without imposing a dtype or shape.
BackendArray: TypeAlias = Union[np.ndarray, "Array"]


def uses_jax(*values: Any) -> bool:
	"""Recognize concrete JAX arrays and tracers, including sequence inputs."""
	jax: ModuleType | None = sys.modules.get("jax")
	if jax is None:
		return False
	for value in values:
		if isinstance(value, (jax.Array, jax.core.Tracer)):
			return True
		if isinstance(value, (tuple, list)) and uses_jax(*value):
			return True
	return False


def array_namespace(*values: Any) -> "ModuleType":
	"""Use JAX if any argument belongs to it; otherwise retain NumPy behavior."""
	if uses_jax(*values):
		import jax.numpy as jnp
		return jnp
	return np


__all__ = ["BackendArray", "array_namespace", "uses_jax"]
