"""Full-cyclotron trajectory."""

from __future__ import annotations

import numpy as np

from .trajectory import Trajectory


class TrajectoryFC(Trajectory):
	"""Full-cyclotron state stored as ``[x, y, vx, vy]``."""

	state_dimension = 4

	def __init__(
		self,
		state: np.ndarray | None = None,
		*,
		rho: float,
		eta: float,
	) -> None:
		eta = float(eta)
		if not np.isfinite(eta):
			raise ValueError("`eta` must be finite.")
		if float(rho) == 0 or eta == 0:
			raise ValueError("TrajectoryFC requires non-zero `rho` and `eta`.")
		self.eta = eta
		super().__init__(state, rho=rho)

	@property
	def velocity_scale(self) -> float:
		return self.rho / (2 * abs(self.eta))

	@property
	def electric_scale(self) -> float:
		return float(np.sign(self.eta) / self.rho)

	@property
	def larmor_frequency(self) -> float:
		return 1 / (2 * self.eta)

	def velocities(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		_, _, vx, vy = self.split(state)
		return vx, vy


__all__ = ["TrajectoryFC"]
