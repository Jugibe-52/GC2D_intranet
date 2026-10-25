"""NumPy-version-independent linear algebra for independent particle blocks."""

from __future__ import annotations

import numpy as np
from typing import cast
from contracts.arrays import BackendArray
from contracts.arrays import array_namespace
from types import ModuleType


def _solve_particle_systems(
	matrix: BackendArray, right_side: BackendArray, *, singular_message: str,
) -> BackendArray:
	"""Solve ``(N,d,d)`` matrices against ``(N,d)`` particle vectors.

	An explicit singleton right-hand-side axis has the same meaning in NumPy
	1.x and 2.x. Only that axis is removed; particle and coordinate axes remain.
	"""
	if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2] or right_side.shape != matrix.shape[:2]:
		raise ValueError("Particle systems require matrices (N,d,d) and vectors (N,d).")
	xp: ModuleType = array_namespace(matrix, right_side)
	try:
		return cast(BackendArray, xp.asarray(xp.linalg.solve(matrix, right_side[..., None])[..., 0]))
	except np.linalg.LinAlgError as exc:
		raise RuntimeError(singular_message) from exc


__all__: list[str] = []


def solve_linear(matrix: BackendArray, right_side: BackendArray, *, singular_message: str) -> BackendArray:
	"""Solve one dense system, preserving native arrays and host error context."""
	xp: ModuleType = array_namespace(matrix, right_side)
	try:
		return cast(BackendArray, xp.linalg.solve(matrix, right_side))
	except np.linalg.LinAlgError as exc:
		raise RuntimeError(singular_message) from exc
