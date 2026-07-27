"""Capability contracts implemented by physical dynamical systems."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DynamicalSystem(Protocol):
	"""Physical equations consumable by a general ODE method."""

	state_dimension: ClassVar[int]

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Return a derivative with the same packed layout as ``state``."""


@runtime_checkable
class HamiltonianSystem(Protocol):
	"""Optional capability for evaluating physical energy."""

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return one Hamiltonian value per particle and optional saved time."""


@runtime_checkable
class ExtendedHamiltonianSystem(HamiltonianSystem, Protocol):
	"""Hamiltonian capability needed to track time-conjugate momentum."""

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return minus the explicit time derivative of the Hamiltonian."""


@runtime_checkable
class CyclotronSplitSystem(DynamicalSystem, Protocol):
	"""FC operations and parameters required by the exact split formulation."""

	rho: float
	eta: float

	@property
	def larmor_frequency(self) -> float:
		"""Return the signed angular rate used by the exact cyclotron flow."""

	def electric_acceleration(
		self,
		t: float,
		x: np.ndarray,
		y: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Return electric acceleration at paired particle positions."""


__all__ = [
	"CyclotronSplitSystem",
	"DynamicalSystem",
	"ExtendedHamiltonianSystem",
	"HamiltonianSystem",
]
