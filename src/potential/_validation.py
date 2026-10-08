"""Validate and own sampled potential data independently of its backend."""

from typing import Any

import numpy as np

from .grid import Grid


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
	"""Return an owned, immutable array with the requested dtype."""
	array = np.array(values, dtype=dtype, copy=True)
	array.setflags(write=False)
	return array


def _validated_potential_data(
	grid: Grid,
	mean: np.ndarray | None,
	modes: np.ndarray | None,
	frequencies: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Validate and own the mutually dependent runtime potential fields."""
	shape = grid.shape
	if mean is None:
		mean_values = np.zeros(shape, dtype=float)
	else:
		mean_values = np.asarray(mean, dtype=float)
		if mean_values.shape != shape:
			raise ValueError(
				f"The mean field shape is {mean_values.shape}; expected {shape}."
			)
		if not np.all(np.isfinite(mean_values)):
			raise ValueError("The mean field must contain finite values.")

	if frequencies is None:
		frequency_values = np.empty(0, dtype=float)
	elif not isinstance(frequencies, np.ndarray):
		raise TypeError("`frequencies` must be a NumPy array or None.")
	else:
		frequency_values = np.asarray(frequencies, dtype=float)
	if frequency_values.ndim != 1:
		raise ValueError("`frequencies` must be one-dimensional.")
	if not np.all(np.isfinite(frequency_values)) or np.any(frequency_values <= 0):
		raise ValueError("`frequencies` must contain finite positive values.")

	if modes is None:
		mode_values = np.empty((0, *shape), dtype=np.complex128)
	else:
		mode_values = np.asarray(modes, dtype=np.complex128)
		if mode_values.ndim != 3 or mode_values.shape[1:] != shape:
			raise ValueError(
				"`modes` must have shape "
				f"(mode_count, {shape[0]}, {shape[1]})."
			)
		if not np.all(np.isfinite(mode_values)):
			raise ValueError("The mode fields must contain finite values.")
	if mode_values.shape[0] != frequency_values.size:
		raise ValueError("The mode count must equal the frequency count.")

	return (
		_readonly_array(mean_values, dtype=float),
		_readonly_array(mode_values, dtype=np.complex128),
		_readonly_array(frequency_values, dtype=float),
	)


