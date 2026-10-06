"""Animation controls with explicit strict and legacy coercing policies."""

from __future__ import annotations

import numpy as np


def frame_indices(sample_count: int, frames: int | None) -> np.ndarray:
	"""Select strictly increasing saved-state indices for an animation."""
	if frames is None:
		return np.arange(sample_count, dtype=int)
	if (
		isinstance(frames, (bool, np.bool_))
		or not isinstance(frames, (int, np.integer))
		or not 2 <= int(frames) <= sample_count
	):
		raise ValueError("`frames` must be None or an integer from 2 to the sample count.")
	return np.asarray(
		np.unique(np.linspace(0, sample_count - 1, int(frames), dtype=int)),
		dtype=int,
	)


def positive_interval(value: int, *, strict: bool = False) -> int:
	"""Require a positive interval while retaining each animator's input policy."""
	if strict:
		if (
			isinstance(value, (bool, np.bool_))
			or not isinstance(value, (int, np.integer))
			or value <= 0
		):
			raise ValueError("`interval` must be a positive integer.")
	elif isinstance(value, (bool, np.bool_)) or int(value) <= 0:
		raise ValueError("`interval` must be a positive integer.")
	return value


def boolean_control(value: bool, name: str, *, numpy_scalar: bool = True) -> bool:
	"""Check a display flag without changing its accepted scalar representation."""
	accepted = (bool, np.bool_) if numpy_scalar else (bool,)
	if not isinstance(value, accepted):
		raise TypeError(f"`{name}` must be a boolean.")
	return value


def unbounded_frame_count(frames: int | None) -> int | None:
	"""Validate animators that cap an optional frame count to available samples."""
	if frames is not None and (
		isinstance(frames, (bool, np.bool_))
		or not isinstance(frames, (int, np.integer))
		or frames < 2
	):
		raise ValueError("`frames` must be None or an integer of at least 2.")
	return frames
