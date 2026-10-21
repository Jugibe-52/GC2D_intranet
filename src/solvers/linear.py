"""NumPy-version-independent linear algebra for independent particle blocks."""

from __future__ import annotations

import numpy as np


def _solve_particle_systems(
	matrix: np.ndarray, right_side: np.ndarray, *, singular_message: str,
) -> np.ndarray:
	"""Solve ``(N,d,d)`` matrices against ``(N,d)`` particle vectors.

	An explicit singleton right-hand-side axis has the same meaning in NumPy
	1.x and 2.x. Only that axis is removed; particle and coordinate axes remain.
	"""
	if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2] or right_side.shape != matrix.shape[:2]:
		raise ValueError("Particle systems require matrices (N,d,d) and vectors (N,d).")
	try:
		return np.asarray(np.linalg.solve(matrix, right_side[..., None])[..., 0])
	except np.linalg.LinAlgError as exc:
		raise RuntimeError(singular_message) from exc


__all__: list[str] = []
