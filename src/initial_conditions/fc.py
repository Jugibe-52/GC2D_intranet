"""Full-cyclotron initial configurations and their state layout."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .base import StateConfiguration


class FCState(NamedTuple):
	"""FC position and velocity coordinates with matching dimensions.

	Every field has shape ``(N, *sample_axes)``.  ``vx`` and ``vy`` are the
	velocity coordinates stored by the normalized model. Their physical scaling
	is owned by :class:`dynamics.FullCyclotronDynamics`.
	"""

	x: np.ndarray
	y: np.ndarray
	vx: np.ndarray
	vy: np.ndarray


class FCInitialConfiguration(StateConfiguration):
	"""Full-cyclotron state in component-major ``[x, y, vx, vy]`` order.

	For ``N`` particles, the initial state contains all ``x`` values, then all
	``y``, ``vx`` and ``vy`` values. Physical ``rho`` and ``eta`` parameters are
	owned exclusively by the corresponding dynamics object.
	"""

	state_dimension = 4

	@classmethod
	def from_components(
		cls,
		*,
		x: np.ndarray,
		y: np.ndarray,
		vx: np.ndarray,
		vy: np.ndarray,
	) -> FCInitialConfiguration:
		"""Create an FC configuration without exposing packed state order."""
		state = cls.pack_components(x, y, vx, vy)
		return cls(state)

	def velocities(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Return normalized ``vx`` and ``vy`` coordinate blocks."""
		components = self.split(state)
		return components.vx, components.vy

	def split(self, state: np.ndarray) -> FCState:
		"""Return named position/velocity blocks, preserving sample axes."""
		x, y, vx, vy = super().split(state)
		return FCState(x, y, vx, vy)


class TrajectoryFC(FCInitialConfiguration):
	"""Deprecated FC configuration carrying former physical metadata."""

	def __init__(
		self,
		state: np.ndarray | None = None,
		*,
		rho: float,
		eta: float,
	) -> None:
		"""Create a legacy FC trajectory with finite non-zero parameters."""
		radius = float(rho)
		scale = float(eta)
		if not np.isfinite(radius) or not np.isfinite(scale):
			raise ValueError("`rho` and `eta` must be finite.")
		if radius <= 0 or scale == 0:
			raise ValueError(
				"TrajectoryFC requires positive `rho` and non-zero `eta`."
			)
		self.rho = radius
		self.eta = scale
		super().__init__(state)

	@classmethod
	def from_components(
		cls,
		*,
		x: np.ndarray,
		y: np.ndarray,
		vx: np.ndarray,
		vy: np.ndarray,
		rho: float | None = None,
		eta: float | None = None,
	) -> TrajectoryFC:
		"""Create a legacy FC trajectory from named state components."""
		if rho is None or eta is None:
			raise TypeError("Legacy TrajectoryFC requires both `rho` and `eta`.")
		return cls(cls.pack_components(x, y, vx, vy), rho=rho, eta=eta)

	@property
	def velocity_scale(self) -> float:
		"""Compatibility view of the former velocity scale."""
		return self.rho / (2 * abs(self.eta))

	@property
	def electric_scale(self) -> float:
		"""Compatibility view of the former electric scale."""
		return float(np.sign(self.eta) / self.rho)

	@property
	def larmor_frequency(self) -> float:
		"""Compatibility view of the former cyclotron frequency."""
		return 1 / (2 * self.eta)


__all__ = ["FCInitialConfiguration", "FCState", "TrajectoryFC"]
