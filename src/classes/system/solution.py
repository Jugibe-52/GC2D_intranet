"""Numerical result returned by a system simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Solution:
	"""Saved integration output.

	``y`` stores state components by row and saved times by column. ``k`` and
	``err`` are populated only when generalized-energy tracking is requested.
	"""

	t: np.ndarray
	y: np.ndarray
	n_steps: int
	k: np.ndarray | None = None
	err: float | None = None


__all__ = ["Solution"]
