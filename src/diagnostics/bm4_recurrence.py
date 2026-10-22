"""Periodic initial-return distances for independent BM4 trajectories."""

import numpy as np

from contracts._study_validation import positive_finite
from contracts.study_results import ParallelBM4RecurrenceResult


def recurrence_distances(result: ParallelBM4RecurrenceResult, *, period: float) -> np.ndarray:
	"""Return read-only minimum-image distances with shape ``(particles, samples)``.

	Positions use ``(particles, 2, samples)``. The cell period and returned
	distances use the same spatial units as the stored initial positions.
	"""
	cell_period: float = positive_finite(period, "period")
	displacement: np.ndarray = result.positions - result.initial_positions[:, :, None]
	periodic: np.ndarray = (
		(displacement + 0.5 * cell_period) % cell_period - 0.5 * cell_period
	)
	distances: np.ndarray = np.asarray(np.linalg.norm(periodic, axis=1), dtype=float)
	distances[:, 0] = 0.0
	distances.setflags(write=False)
	return distances


__all__ = ["recurrence_distances"]
