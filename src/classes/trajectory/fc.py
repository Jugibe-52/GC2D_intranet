"""Full-cyclotron trajectories and their position/velocity state layout."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .trajectory import Trajectory


class FCState(NamedTuple):
	"""FC position and velocity coordinates with matching dimensions.

	Every field has shape ``(N, *sample_axes)``.  ``vx`` and ``vy`` are the
	velocity coordinates stored by the normalized model; multiplying them by
	:meth:`TrajectoryFC.velocity_scale` gives the corresponding position rates.
	"""

	x: np.ndarray
	y: np.ndarray
	vx: np.ndarray
	vy: np.ndarray


class TrajectoryFC(Trajectory):
	"""Full-cyclotron state in component-major ``[x, y, vx, vy]`` order.

	For ``N`` particles, the initial state contains all ``x`` values, then all
	``y``, ``vx`` and ``vy`` values.  ``rho`` is the positive normalized Larmor
	radius.  ``eta`` is a signed normalized model parameter: its magnitude sets
	the velocity and cyclotron-frequency scales, while its sign fixes the
	rotation and electric-response orientation.
	"""

	state_dimension = 4

	def __init__(
		self,
		state: np.ndarray | None = None,
		*,
		rho: float,
		eta: float,
	) -> None:
		"""Create an FC trajectory with finite, non-zero ``rho`` and ``eta``."""
		eta = float(eta)
		if not np.isfinite(eta):
			raise ValueError("`eta` must be finite.")
		# The derived velocity, electric and frequency scales divide by these
		# parameters, so the FC model is undefined when either is zero.
		if float(rho) == 0 or eta == 0:
			raise ValueError("TrajectoryFC requires non-zero `rho` and `eta`.")
		# Like ``rho``, ``eta`` belongs to the full ensemble and is not repeated
		# inside each particle state.
		self.eta = eta
		super().__init__(state, rho=rho)

	@property
	def velocity_scale(self) -> float:
		"""Return the factor mapping ``vx, vy`` to ``dx/dt, dy/dt``."""
		return self.rho / (2 * abs(self.eta))

	@property
	def electric_scale(self) -> float:
		"""Return the signed factor mapping electric field to acceleration."""
		return float(np.sign(self.eta) / self.rho)

	@property
	def larmor_frequency(self) -> float:
		"""Return the signed angular rate of velocity-space rotation."""
		return 1 / (2 * self.eta)

	def velocities(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Return normalized ``vx`` and ``vy`` coordinate blocks."""
		components = self.split(state)
		return components.vx, components.vy

	def split(self, state: np.ndarray) -> FCState:
		"""Return named position/velocity blocks, preserving sample axes."""
		x, y, vx, vy = super().split(state)
		return FCState(x, y, vx, vy)


__all__ = ["FCState", "TrajectoryFC"]
