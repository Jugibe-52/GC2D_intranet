"""Full-cyclotron physical dynamics independent of initial conditions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np

from potential.potential import Potential

from dynamics.coordinates import FCCoordinates
from ._equations import fc_velocity, fc_hamiltonian


if TYPE_CHECKING:
	from dynamics.jax_dynamics import JaxFullCyclotronDynamics
	from contracts.arrays import BackendArray
	from types import ModuleType


def _validated_parameters(potential: Potential, rho: float, eta: float) -> tuple[float, float]:
	"""Normalize physical parameters in their established validation order."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	rho = float(rho)
	eta = float(eta)
	if not np.isfinite(rho) or not np.isfinite(eta) or rho == 0 or eta == 0:
		raise ValueError("FullCyclotronDynamics requires finite, non-zero parameters.")
	if rho < 0:
		raise ValueError("`rho` must be positive.")
	return rho, eta


@dataclass(frozen=True, init=False, eq=False)
class FullCyclotronDynamics:
	"""FC equations for fixed physical parameters and declared coordinates."""

	coordinates: ClassVar[FCCoordinates] = FCCoordinates()
	state_dimension: ClassVar[int] = 4

	potential: Potential
	rho: float
	eta: float

	@lru_cache(maxsize=16)
	def _as_jax(self) -> JaxFullCyclotronDynamics:
		"""Prepare physical components once without rebuilding or mutating them."""
		from dynamics.jax_dynamics import JaxFullCyclotronDynamics
		if type(self) not in (FullCyclotronDynamics, JaxFullCyclotronDynamics):
			raise TypeError("JAX preparation requires built-in dynamics; subclass overrides are not compiled.")
		potential: Potential = self.potential._as_jax()
		if type(self) is JaxFullCyclotronDynamics and potential is self.potential:
			return self
		prepared: JaxFullCyclotronDynamics = object.__new__(JaxFullCyclotronDynamics)
		prepared.__dict__.update(self.__dict__)
		object.__setattr__(prepared, "potential", potential)
		return prepared

	def __init__(self, potential: Potential, *, rho: float, eta: float) -> None:
		"""Create FC dynamics with finite, non-zero ``rho`` and ``eta``."""
		rho, eta = _validated_parameters(potential, rho, eta)
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
		ex: BackendArray
		ey: BackendArray
		ex, ey = self.potential.electric_field(t, x, y)
		return self.electric_scale * ex, self.electric_scale * ey

	def cyclotron_flow(self, t: Any, state: Any, duration: Any) -> Any:
		"""Exact field-free flow in the physical FC normalization."""
		x: BackendArray
		y: BackendArray
		vx: BackendArray
		vy: BackendArray
		x, y, vx, vy = self.coordinates.split(state)
		xp: ModuleType = self.coordinates.array_namespace
		rotation: BackendArray = xp.exp(-1j * duration * self.larmor_frequency)
		position: BackendArray = x + 1j * y + 1j * self.rho * np.sign(self.eta) * (rotation - 1) * (vx + 1j * vy)
		velocity: BackendArray = rotation * (vx + 1j * vy)
		return self.coordinates.pack_components(position.real, position.imag, velocity.real, velocity.imag)

	def electric_kick(self, t: Any, state: Any, duration: Any) -> Any:
		"""Exact electric kick at a fixed time and position."""
		x: BackendArray
		y: BackendArray
		vx: BackendArray
		vy: BackendArray
		x, y, vx, vy = self.coordinates.split(state)
		ax: BackendArray
		ay: BackendArray
		ax, ay = self.electric_acceleration(t, x, y)
		return self.coordinates.pack_components(x, y, vx + duration * ax, vy + duration * ay)

	def vector_field(self, t: Any, state: Any) -> Any:
		"""Evaluate FC equations in packed ``[x, y, vx, vy]`` order."""
		x: BackendArray
		y: BackendArray
		vx: BackendArray
		vy: BackendArray
		x, y, vx, vy = self.coordinates.split(state)
		acceleration_x: BackendArray
		acceleration_y: BackendArray
		acceleration_x, acceleration_y = self.electric_acceleration(t, x, y)
		return self.coordinates.pack_components(*fc_velocity(
			acceleration_x, acceleration_y, vx, vy, velocity_scale=self.velocity_scale,
			larmor_frequency=self.larmor_frequency,
		))

	def hamiltonian(
		self,
		t: Any,
		state: Any,
	) -> Any:
		"""Evaluate kinetic plus scaled electrostatic energy."""
		x: BackendArray
		y: BackendArray
		vx: BackendArray
		vy: BackendArray
		x, y, vx, vy = self.coordinates.split(state)
		return self.coordinates.array_namespace.asarray(fc_hamiltonian(
			self.potential.evaluate(t, x, y), vx, vy,
			velocity_scale=self.velocity_scale, electric_scale=self.electric_scale,
		))

	def extended_momentum_derivative(
		self,
		t: Any,
		state: Any,
	) -> Any:
		"""Evaluate the time-conjugate momentum derivative."""
		x: BackendArray
		y: BackendArray
		x, y, *_ = self.coordinates.split(state)
		return self.coordinates.array_namespace.asarray(
			-self.electric_scale
			* self.potential.evaluate(t, x, y, dt=1)
		)


__all__ = ["FullCyclotronDynamics"]
