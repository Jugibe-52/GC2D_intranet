"""Shared contracts for sampled planar solutions used by comparison plots."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from solution import Solution


def validated_solutions(
	solutions: Mapping[str, Solution],
	*,
	expected_count: int | None = None,
	alignment_error: str = "All solutions must share times and particle count.",
) -> tuple[tuple[str, ...], np.ndarray, int]:
	"""Validate an aligned non-empty collection of planar solutions."""
	if not solutions:
		raise ValueError("At least one labeled solution is required.")
	labels = tuple(solutions)
	if expected_count is not None and len(labels) != expected_count:
		raise ValueError(f"The plot requires exactly {expected_count} solutions.")
	reference_times: np.ndarray | None = None
	particle_count: int | None = None
	for label, solution in solutions.items():
		if not isinstance(label, str) or not label:
			raise ValueError("Solution labels must be non-empty strings.")
		if not isinstance(solution, Solution):
			raise TypeError("Every comparison value must be a Solution.")
		times = np.asarray(solution.t, dtype=float)
		x, y = solution.positions()
		if x.shape != y.shape or x.ndim != 2 or x.shape[1] != times.size:
			raise ValueError("Every solution must contain aligned planar trajectories.")
		if reference_times is None:
			reference_times = times
			particle_count = x.shape[0]
		elif not np.array_equal(times, reference_times) or x.shape[0] != particle_count:
			raise ValueError(alignment_error)
	assert reference_times is not None and particle_count is not None
	return labels, reference_times, particle_count

