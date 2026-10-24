"""Signed polygon area for ordered guiding-centre boundary vertices."""

from __future__ import annotations

import numpy as np
from contracts.dynamics.coordinates import StateLayout

def _positive_finite(value: float, name: str) -> float:
	"""Normalize a scalar geometric parameter and enforce positivity."""
	try:
		result: float = float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"`{name}` must be positive and finite.") from exc
	if not np.isfinite(result) or result <= 0:
		raise ValueError(f"`{name}` must be positive and finite.")
	return result


def calculate_area(
	state: np.ndarray,
	*,
	layout: StateLayout,
	period: float | None = None,
) -> np.ndarray:
	"""Calculate the signed polygon area for one state or a time series.

	When ``period`` is provided, consecutive vertices are unwrapped with the
	minimum-image convention before applying the shoelace formula; ``period``
	must therefore use the same coordinate normalization as ``x`` and ``y``.
	A single state returns a scalar array.  An input with shape
	``(2 * N, *sample_axes)`` returns an area array of shape ``sample_axes``.
	Counter-clockwise contours have positive area.
	"""
	if layout.state_dimension != 2:
		raise ValueError("Area calculation requires a two-component position layout.")
	value: np.ndarray = np.asarray(layout.validate_packed_state_layout(state), dtype=float)
	if not np.all(np.isfinite(value)):
		raise ValueError("The area state must contain only finite values.")
	# Here axis zero enumerates successive boundary vertices; any trailing
	# axes are independent contour snapshots, commonly integration times.
	x: np.ndarray
	y: np.ndarray
	x, y = layout.positions(value)
	if x.shape[0] < 3:
		raise ValueError("An area boundary requires at least three points.")
	# Periodic unwrapping edits coordinates in place.  Copies protect both the
	# stored initial condition and integration results supplied by callers.
	x = np.asarray(x, dtype=float).copy()
	y = np.asarray(y, dtype=float).copy()

	if period is not None:
		period = _positive_finite(period, "period")
		# Reconstruct a locally continuous polygon independently at every time
		# sample, choosing the nearest periodic image of each next vertex.
		for vertex in range(1, x.shape[0]):
			delta_x = x[vertex] - x[vertex - 1]
			delta_y = y[vertex] - y[vertex - 1]
			delta_x -= period * np.round(delta_x / period)
			delta_y -= period * np.round(delta_y / period)
			x[vertex] = x[vertex - 1] + delta_x
			y[vertex] = y[vertex - 1] + delta_y

	# The rolled arrays pair every vertex with its successor and implicitly
	# include the closing edge from the final point to the first.
	return np.asarray(
		0.5
		* np.sum(
			x * np.roll(y, -1, axis=0) - y * np.roll(x, -1, axis=0),
			axis=0,
		)
	)


__all__ = ["calculate_area"]
