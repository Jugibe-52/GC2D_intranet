"""Numerical result returned by a system simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from classes.trajectory.trajectory import Trajectory


@dataclass(slots=True)
class Solution:
	"""Saved integration output.

	``t`` has shape ``(n_output_samples,)`` and contains the uniformly requested
	output times. ``y`` has shape ``(physical_state_size, n_output_samples)``: its
	rows retain the trajectory's component-major layout and its columns are
	instantaneous states. ``n_steps`` counts complete BM4 cycles.

	``trajectory`` records the layout that produced ``y`` so callers can recover
	named physical components through :meth:`components` without repeating that
	context. ``k`` has one momentum-conjugate-to-time history per particle, with
	shape ``(particle_count, n_output_samples)``, and ``err`` is the maximum
	absolute drift of ``H + k``. Both remain ``None`` unless energy checking was
	requested.
	"""

	t: np.ndarray
	y: np.ndarray
	n_steps: int
	k: np.ndarray | None = None
	err: float | None = None
	trajectory: Trajectory | None = None

	def components(
		self,
		trajectory: Trajectory | None = None,
	) -> tuple[np.ndarray, ...]:
		"""Return named physical components using the solution's state layout.

		An explicit trajectory remains accepted for manually constructed solutions.
		When integration already attached one, a trajectory with a different state
		dimension is rejected to prevent interpreting GC output as FC or vice versa.
		"""
		layout = self.trajectory if trajectory is None else trajectory
		if layout is None:
			raise ValueError(
				"`trajectory` is required because this solution has no attached layout."
			)
		if (
			self.trajectory is not None
			and layout.state_dimension != self.trajectory.state_dimension
		):
			raise TypeError("The supplied trajectory is incompatible with this solution.")
		return layout.split(self.y)


__all__ = ["Solution"]
