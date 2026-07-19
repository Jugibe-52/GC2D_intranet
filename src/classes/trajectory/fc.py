"""Full-cyclotron trajectories and their position/velocity state layout."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .trajectory import Trajectory


class FCState(NamedTuple):
	"""Named position and velocity blocks with matching dimensions."""

	x: np.ndarray
	y: np.ndarray
	vx: np.ndarray
	vy: np.ndarray


class TrajectoryFC(Trajectory):
	"""Full-cyclotron state stored as component-major ``[x, y, vx, vy]``."""

	state_dimension = 4

	def __init__(
		self,
		state: np.ndarray | None = None,
		*,
		rho: float,
		eta: float,
	) -> None:
		"""Create a full-cyclotron trajectory with finite, non-zero scales."""
		eta = float(eta)
		if not np.isfinite(eta):
			raise ValueError("`eta` must be finite.")
		# The derived velocity, electric and frequency scales divide by these
		# parameters, so the FC model is undefined when either is zero.
		if float(rho) == 0 or eta == 0:
			raise ValueError("TrajectoryFC requires non-zero `rho` and `eta`.")
		self.eta = eta
		super().__init__(state, rho=rho)

	@property
	def velocity_scale(self) -> float:
		"""Return the characteristic speed used by the FC equations."""
		return self.rho / (2 * abs(self.eta))

	@property
	def electric_scale(self) -> float:
		"""Return the signed electric-force scale."""
		return float(np.sign(self.eta) / self.rho)

	@property
	def larmor_frequency(self) -> float:
		"""Return the signed Larmor angular frequency."""
		return 1 / (2 * self.eta)

	def velocities(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Return the ``vx`` and ``vy`` blocks of a state or time series."""
		components = self.split(state)
		return components.vx, components.vy

	def split(self, state: np.ndarray) -> FCState:
		"""Return named position and velocity blocks, preserving trailing axes."""
		x, y, vx, vy = super().split(state)
		return FCState(x, y, vx, vy)


__all__ = ["FCState", "TrajectoryFC"]
