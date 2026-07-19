"""Numerical result returned by a system simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Solution:
	"""Saved integration output.

	``t`` has shape ``(n_save_step,)`` and contains the uniformly requested
	output times. ``y`` has shape ``(physical_state_size, n_save_step)``: its
	rows retain the trajectory's component-major layout and its columns are
	instantaneous states. ``n_steps`` counts complete internal BM4 advances.

	``k`` has one momentum-conjugate-to-time history per particle, with shape
	``(particle_count, n_save_step)``, and ``err`` is the maximum absolute drift
	of ``H + k``. Both remain ``None`` unless energy checking was requested.
	"""

	t: np.ndarray
	y: np.ndarray
	n_steps: int
	k: np.ndarray | None = None
	err: float | None = None


__all__ = ["Solution"]
