"""Planar particle distances for packed guiding-center trajectories."""

from __future__ import annotations

from typing import Literal

import numpy as np


DistanceConvention = Literal["euclidean", "periodic"]
DISTANCE_CONVENTIONS: tuple[DistanceConvention, ...] = ("euclidean", "periodic")


def normalized_distance_convention(value: str) -> DistanceConvention:
	"""Validate one supported trajectory-distance convention."""
	if value not in DISTANCE_CONVENTIONS:
		raise ValueError("`distance_convention` must be 'euclidean' or 'periodic'.")
	return value


def particle_distances(
	states: np.ndarray,
	reference_states: np.ndarray,
	*,
	distance_convention: DistanceConvention,
	period: float | None = None,
) -> np.ndarray:
	"""Return planar distances with shape ``(particles, samples)``.

	Both histories use the component-major layout
	``[x_1, ..., x_N, y_1, ..., y_N]``. Periodic distances apply the
	minimum-image convention independently to both coordinates; Euclidean
	distances retain the physical displacement represented by the histories.
	"""
	first = np.asarray(states, dtype=float)
	second = np.asarray(reference_states, dtype=float)
	if first.shape != second.shape or first.ndim != 2 or first.shape[0] % 2:
		raise ValueError("Compared GC histories must have one matching packed shape.")
	convention = normalized_distance_convention(distance_convention)
	difference = first - second
	if convention == "periodic":
		if period is None or not np.isfinite(period) or period <= 0.0:
			raise ValueError("`period` must be positive and finite for periodic distances.")
		difference = (difference + period / 2.0) % period - period / 2.0
	particle_count = first.shape[0] // 2
	return np.asarray(
		np.hypot(difference[:particle_count], difference[particle_count:]),
		dtype=float,
	)


def periodic_particle_distances(
	states: np.ndarray,
	reference_states: np.ndarray,
	*,
	period: float,
) -> np.ndarray:
	"""Return minimum-image planar distances for periodic trajectories."""
	return particle_distances(
		states,
		reference_states,
		distance_convention="periodic",
		period=period,
	)


__all__: list[str] = []
