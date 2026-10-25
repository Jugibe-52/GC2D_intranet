"""Apply the shared NumPy/JAX array compatibility contract.

NumPy-only values select NumPy; any JAX array or tracer selects JAX, including
mixed components and nested lists or tuples. These helpers identify array
representations without selecting devices or preparing an execution environment.
The NumPy path does not import the optional JAX runtime.
"""

import sys
from typing import Any, TYPE_CHECKING, TypeAlias, Union, TypeVar, cast

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


_Value = TypeVar("_Value")


def checked_finite(value: _Value, message: str, error: type[Exception] = ValueError) -> _Value:
	"""Raise eagerly, or propagate invalid numerical data through compiled work.

	A traced invalid intermediate becomes NaN even if later arithmetic could
	otherwise hide an infinity. The execution boundary rejects the resulting
	invalid state or unconverged solve, without callbacks inside compiled loops.
	"""
	xp: ModuleType = array_namespace(value)
	valid: Any = xp.all(xp.isfinite(value))
	jax: ModuleType | None = sys.modules.get("jax")
	if jax is not None and isinstance(valid, jax.core.Tracer):
		return cast(_Value, xp.where(valid, value, xp.nan))
	if not valid:
		raise error(message)
	return value


def scan_arrays(function: Any, initial: Any, values: Any) -> tuple[Any, Any]:
	"""Fold over leading array axes, returning the carry and stacked outputs.

	The function defines the numerical recurrence once. NumPy executes ordinary
	iterations; traced arrays use a compact compiled loop with the same order.
	Carries and outputs are nested tuples of arrays or scalars.
	"""
	if uses_jax(initial, values):
		from jax import lax
		return lax.scan(function, initial, values)
	carry: Any = initial
	outputs: list[Any] = []
	for index in range(len(values[0])):
		output: Any
		carry, output = function(carry, tuple(value[index] for value in values))
		outputs.append(output)
	def stack(items: list[Any]) -> Any:
		if isinstance(items[0], tuple):
			return tuple(stack([item[index] for item in items]) for index in range(len(items[0])))
		return np.stack(items)
	return carry, stack(outputs)


def select_array_branch(index: Any, branches: tuple[Any, ...], *values: Any) -> Any:
	"""Select one numerical branch with native scalar control."""
	if uses_jax(index):
		from jax import lax
		return lax.switch(index, branches, *values)
	return branches[int(index)](*values)


def map_arrays(function: Any, *values: Any) -> Any:
	"""Apply one array function to corresponding leading-axis entries."""
	if uses_jax(*values):
		from jax import vmap
		return vmap(function)(*values)
	return np.stack([function(*items) for items in zip(*values, strict=True)])
