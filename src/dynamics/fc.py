"""Full-cyclotron physical dynamics independent of initial conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from potential.potential import Potential

from contracts.arrays import array_namespace
from contracts.state_layout import FCStateLayout
from ._equations import fc_velocity, fc_hamiltonian


@dataclass(frozen=True, init=False, eq=False)
class FullCyclotronDynamics:
	"""FC equations for fixed physical parameters, for NumPy or JAX arrays."""

	state_dimension: ClassVar[int] = 4

	potential: Potential
	rho: float
	eta: float

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
		object.__setattr__(self, "potential", potential)
		object.__setattr__(self, "rho", rho)
		object.__setattr__(self, "eta", eta)

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
		t: Any,
		x: Any,
		y: Any,
	) -> tuple[Any, Any]:
		"""Evaluate electric acceleration at paired positions."""
		ex, ey = self.potential.electric_field(t, x, y)
		return self.electric_scale * ex, self.electric_scale * ey

	def vector_field(self, t: Any, state: Any) -> Any:
		"""Evaluate FC equations in packed ``[x, y, vx, vy]`` order."""
		x, y, vx, vy = FCStateLayout().split(state)
		acceleration_x, acceleration_y = self.electric_acceleration(t, x, y)
		return FCStateLayout.pack_components(*fc_velocity(
			acceleration_x, acceleration_y, vx, vy, velocity_scale=self.velocity_scale,
			larmor_frequency=self.larmor_frequency,
		))

	def hamiltonian(
		self,
		t: Any,
		state: Any,
	) -> Any:
		"""Evaluate kinetic plus scaled electrostatic energy."""
		x, y, vx, vy = FCStateLayout().split(state)
		return array_namespace(t, state).asarray(fc_hamiltonian(
			self.potential.evaluate(t, x, y), vx, vy,
			velocity_scale=self.velocity_scale, electric_scale=self.electric_scale,
		))

	def extended_momentum_derivative(
		self,
		t: Any,
		state: Any,
	) -> Any:
		"""Evaluate the time-conjugate momentum derivative."""
		x, y, *_ = FCStateLayout().split(state)
		return array_namespace(t, state).asarray(
			-self.electric_scale
			* self.potential.evaluate(t, x, y, dt=1)
		)


__all__ = ["FullCyclotronDynamics"]
