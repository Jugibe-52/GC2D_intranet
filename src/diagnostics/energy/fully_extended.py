"""Generalized energy read directly from accepted full ``(z,t,k)`` states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import GuidingCenterDynamics
from simulation import FullyExtendedImplicitIntegrationStep, IntegrationStep



@dataclass(frozen=True, slots=True)
class GCFullyExtendedEnergyRecord:
	"""Direct physical and autonomous energy values at one accepted node."""

	step_index: int
	time: float
	duration: float
	hamiltonian: float
	momentum: float
	generalized_energy: float
	energy_error: float
	relative_error: float


def _hamiltonian(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
) -> float:
	"""Return the scalar one-particle Hamiltonian from ``(x,y,t,k)``."""
	value = np.asarray(state, dtype=float)
	if value.shape != (4,) or not np.all(np.isfinite(value)):
		raise ValueError("A full physical extended state must have shape (4,).")
	result = np.asarray(dynamics.hamiltonian(float(value[2]), value[:2]), dtype=float)
	if result.size != 1 or not np.all(np.isfinite(result)):
		raise ValueError("The one-particle Hamiltonian must be finite and scalar.")
	return float(result.reshape(-1)[0])


class GCFullyExtendedEnergyObserver:
	"""Record ``K=h(t,z)+k`` without reconstructing the conjugate momentum."""

	def __init__(
		self,
		dynamics: GuidingCenterDynamics,
		*,
		initial_state: np.ndarray,
	) -> None:
		"""Initialize an accepted-node history from one physical ``R^4`` state."""
		if not isinstance(dynamics, GuidingCenterDynamics):
			raise TypeError("`dynamics` must be GuidingCenterDynamics.")
		value = np.asarray(initial_state, dtype=float)
		if value.shape != (4,) or not np.all(np.isfinite(value)):
			raise ValueError("`initial_state` must be finite with shape (4,).")
		hamiltonian = _hamiltonian(dynamics, value)
		energy = hamiltonian + float(value[3])
		self._dynamics = dynamics
		self._initial_energy = energy
		self._energy_scale = max(abs(energy), float(np.finfo(float).eps))
		self._records: list[GCFullyExtendedEnergyRecord] = [
			GCFullyExtendedEnergyRecord(
				step_index=-1,
				time=float(value[2]),
				duration=0.0,
				hamiltonian=hamiltonian,
				momentum=float(value[3]),
				generalized_energy=energy,
				energy_error=0.0,
				relative_error=0.0,
			)
		]

	@property
	def records(self) -> tuple[GCFullyExtendedEnergyRecord, ...]:
		"""Return immutable records including the initial node."""
		return tuple(self._records)

	def __call__(self, record: IntegrationStep) -> None:
		"""Read ``h``, ``k``, and ``K`` from one accepted projected state."""
		if not isinstance(record, FullyExtendedImplicitIntegrationStep):
			raise TypeError(
				"The observer requires FullyExtendedImplicitIntegrationStep records."
			)
		if record.dynamics is not self._dynamics:
			raise ValueError("The observed step belongs to a different GC system.")
		if record.step_index != len(self._records) - 1:
			raise ValueError("Energy records must arrive sequentially.")
		state = np.asarray(record.state_after, dtype=float)
		hamiltonian = _hamiltonian(self._dynamics, state)
		energy = hamiltonian + float(state[3])
		error = energy - self._initial_energy
		self._records.append(
			GCFullyExtendedEnergyRecord(
				step_index=record.step_index,
				time=float(state[2]),
				duration=float(record.duration),
				hamiltonian=hamiltonian,
				momentum=float(state[3]),
				generalized_energy=energy,
				energy_error=error,
				relative_error=error / self._energy_scale,
			)
		)


__all__ = ["GCFullyExtendedEnergyObserver", "GCFullyExtendedEnergyRecord"]
