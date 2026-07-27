"""Full-cyclotron physical dynamics independent of initial conditions."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from classes.potential import Potential

from ._layout import pack_components, split_components


class FullCyclotronDynamics:
	"""Full-cyclotron equations for fixed physical parameters."""

	state_dimension: ClassVar[int] = 4

	def __init__(self, potential: Potential, *, rho: float, eta: float) -> None:
		"""Create FC dynamics with finite, non-zero ``rho`` and ``eta``."""
		if not isinstance(potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		rho = float(rho)
		eta = float(eta)
		if not np.isfinite(rho) or not np.isfinite(eta) or rho == 0 or eta == 0:
			raise ValueError("FullCyclotronDynamics requires finite, non-zero parameters.")
		if rho < 0:
			raise ValueError("`rho` must be positive.")
		self.potential = potential
		self.rho = rho
		self.eta = eta

	@property
	def velocity_scale(self) -> float:
		"""Map normalized velocity coordinates to position rates."""
		return self.rho / (2 * abs(self.eta))

	@property
	def electric_scale(self) -> float:
		"""Map electric-field components to signed acceleration."""
		return float(np.sign(self.eta) / self.rho)

	@property
	def larmor_frequency(self) -> float:
		"""Return the signed angular rate of velocity-space rotation."""
		return 1 / (2 * self.eta)

	def electric_acceleration(
		self,
		t: float,
		x: np.ndarray,
		y: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Evaluate electric acceleration at paired positions."""
		ex, ey = self.potential.electric_field(t, x, y)
		return self.electric_scale * ex, self.electric_scale * ey

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Evaluate FC equations in packed ``[x, y, vx, vy]`` order."""
		x, y, vx, vy = split_components(
			state,
			component_count=self.state_dimension,
		)
		acceleration_x, acceleration_y = self.electric_acceleration(
			t,
			x,
			y,
		)
		return pack_components(
			vx * self.velocity_scale,
			vy * self.velocity_scale,
			acceleration_x + vy * self.larmor_frequency,
			acceleration_y - vx * self.larmor_frequency,
		)

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate kinetic plus scaled electrostatic energy."""
		x, y, vx, vy = split_components(
			state,
			component_count=self.state_dimension,
		)
		kinetic_scale = self.rho / (4 * abs(self.eta))
		return np.asarray(
			kinetic_scale * (vx**2 + vy**2)
			+ self.electric_scale
			* self.potential.evaluate(t, x, y)
		)

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate the time-conjugate momentum derivative."""
		x, y, *_ = split_components(
			state,
			component_count=self.state_dimension,
		)
		return np.asarray(
			-self.electric_scale
			* self.potential.evaluate(t, x, y, dt=1)
		)


__all__ = ["FullCyclotronDynamics"]
