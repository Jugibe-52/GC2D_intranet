"""Full-cyclotron trajectory entity."""

from __future__ import annotations

from typing import Any

import numpy as np

from contracts import Array

from .trajectory import Trajectory


class TrajectoryFC(Trajectory):
	"""Full-cyclotron state layout ``[x, y, vx, vy]``."""

	kind = "fc"
	state_dimension = 4
	degrees_of_freedom = 2

	def __init__(self, **kwargs: Any) -> None:
		super().__init__(**kwargs)
		if self.rho == 0 or self.eta == 0:
			raise ValueError("TrajectoryFC requires non-zero `rho` and `eta`.")

	@property
	def velocity_scale(self) -> float:
		return self.rho / (2 * abs(self.eta))

	@property
	def electric_scale(self) -> float:
		return float(np.sign(self.eta) / self.rho)

	@property
	def larmor_frequency(self) -> float:
		return 1 / (2 * self.eta)

	def get_velocities(self, state: Array) -> tuple[Array, Array]:
		_, _, vx, vy = self.split_state(state)
		return vx, vy

	def _complete_initial_state(
		self,
		x: Array,
		y: Array,
		rng: np.random.Generator | np.random.RandomState,
	) -> Array:
		gyro_angle = rng.uniform(0.0, 2 * np.pi, x.size)
		return np.concatenate((x, y, np.cos(gyro_angle), np.sin(gyro_angle)))

	def to_guiding_center(self, state: Array) -> tuple[Array, Array]:
		"""Convert full-cyclotron state variables to guiding-centre positions."""
		x, y = self.get_positions(state)
		vx, vy = self.get_velocities(state)
		velocity = vy + 1j * vx
		angle = np.pi + np.angle(velocity)
		radius = self.rho * np.abs(velocity)
		return x - radius * np.cos(angle), y + radius * np.sin(angle)


__all__ = ["TrajectoryFC"]
