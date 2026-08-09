"""Private component-major array operations shared by physical dynamics."""

from __future__ import annotations

import numpy as np


def split_components(
	state: np.ndarray,
	*,
	component_count: int,
) -> tuple[np.ndarray, ...]:
	"""Split a packed state without depending on an initial configuration."""
	value = np.asarray(state)
	if (
		value.ndim == 0
		or value.shape[0] == 0
		or value.shape[0] % component_count
	):
		raise ValueError(
			"The leading state axis must contain complete component blocks."
		)
	particle_count = value.shape[0] // component_count
	blocks = value.reshape((component_count, particle_count, *value.shape[1:]))
	return tuple(blocks[index] for index in range(component_count))


def pack_components(*components: np.ndarray) -> np.ndarray:
	"""Pack equally shaped component blocks along the leading state axis."""
	values = tuple(np.asarray(component) for component in components)
	if not values or values[0].ndim == 0 or values[0].shape[0] == 0:
		raise ValueError("State components must be non-empty arrays.")
	if any(value.shape != values[0].shape for value in values[1:]):
		raise ValueError("All state components must have the same shape.")
	blocks = np.stack(values, axis=0)
	return np.asarray(
		blocks.reshape((len(values) * blocks.shape[1], *blocks.shape[2:]))
	)


__all__: list[str] = []
