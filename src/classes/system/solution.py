"""Numerical result returned by a system simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Solution:
	"""Saved times, states, and optional energy diagnostics."""

	t: np.ndarray
	y: np.ndarray
	n_steps: int
	k: np.ndarray | None = None
	err: float | None = None


__all__ = ["Solution"]
