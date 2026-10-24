"""Accepted-step reconstruction of the GC time-conjugate momentum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from dynamics import GuidingCenterDynamics
from contracts.methods.extended.abba import ABBA2ImplicitIntegrationStep, ABBA4ImplicitIntegrationStep
from contracts.methods.extended.bm4 import ImplicitBM4IntegrationStep
from contracts.methods.observation import IntegrationStep
from diagnostics.methods.energy import EnergyObserver


def _scalar_value(value: np.ndarray, name: str) -> float:
	"""Return the unique finite scalar produced by a one-particle evaluation."""
	array: np.ndarray = np.asarray(value, dtype=float)
	if array.size != 1 or not np.all(np.isfinite(array)):
		raise ValueError(f"The one-particle {name} must be one finite scalar.")
	return float(array.reshape(-1)[0])



@dataclass(frozen=True, slots=True)
class GCGeneralizedEnergyRecord:
	"""Physical and extended energy values at one accepted main-grid node."""

	step_index: int
	time: float
	duration: float
	hamiltonian: float
	kappa: float
	generalized_energy: float
	energy_error: float
	relative_error: float


class GCGeneralizedEnergyObserver:
	"""Accumulate ``kappa`` and ``K=h+kappa`` from accepted quadrature data.

	The observer is deliberately restricted to one GC particle. The implicit
	methods evolve only the projected physical state, so the conjugate variable is
	integrated by the shared observer from accepted stage data without perturbing the
	trajectory or adding shadow-step observations.
	"""
	requires_quadrature: ClassVar[bool] = True

	def __init__(
		self,
		dynamics: GuidingCenterDynamics,
		*,
		initial_time: float,
		initial_state: np.ndarray,
	) -> None:
		"""Initialize the normalized conjugate momentum with ``kappa(0)=0``."""
		if not isinstance(dynamics, GuidingCenterDynamics):
			raise TypeError("`dynamics` must be GuidingCenterDynamics.")
		state: np.ndarray = np.asarray(initial_state, dtype=float)
		if state.shape != (2,) or not np.all(np.isfinite(state)):
			raise ValueError("Energy reconstruction requires one finite planar state.")
		time: float = float(initial_time)
		if not np.isfinite(time):
			raise ValueError("`initial_time` must be finite.")
		hamiltonian: float = _scalar_value(
			dynamics.hamiltonian(time, state),
			"Hamiltonian",
		)
		self._dynamics = dynamics
		self._initial_energy = hamiltonian
		self._energy_scale = max(abs(hamiltonian), float(np.finfo(float).eps))
		self._energy = EnergyObserver().new_run(dynamics.hamiltonian,
			dynamics.extended_momentum_derivative, particle_count=1, initial_time=time)
		self._records: list[GCGeneralizedEnergyRecord] = [
			GCGeneralizedEnergyRecord(
				step_index=-1,
				time=time,
				duration=0.0,
				hamiltonian=hamiltonian,
				kappa=0.0,
				generalized_energy=hamiltonian,
				energy_error=0.0,
				relative_error=0.0,
			)
		]

	@property
	def records(self) -> tuple[GCGeneralizedEnergyRecord, ...]:
		"""Return immutable snapshots including the initial node."""
		return tuple(self._records)

	def _validate_next_record(self, record: IntegrationStep) -> None:
		"""Check the next record against this energy stream before changing momentum."""
		if not isinstance(record, IntegrationStep):
			raise TypeError("The energy observer requires an IntegrationStep record.")
		if record.dynamics is not self._dynamics:
			raise ValueError("The observed step belongs to a different GC system.")
		if not isinstance(record, (ABBA2ImplicitIntegrationStep, ABBA4ImplicitIntegrationStep, ImplicitBM4IntegrationStep)):
			raise TypeError("Generalized-energy reconstruction supports ABBA2Implicit, "
				"ABBA4Implicit, and BM4Implicit step records.")
		if isinstance(record, ImplicitBM4IntegrationStep):
			if len(record.base_stages) != 12:
				raise ValueError("An accepted implicit BM4 step must expose twelve stages.")
			for stage in record.base_stages:
				if stage.flow_name not in ("flow", "adjoint_flow"):
					raise ValueError(f"Unsupported BM4 flow name {stage.flow_name!r}.")
		if record.quadrature is None:
			raise ValueError("Energy observation requires accepted-stage quadrature.")

	def __call__(self, record: IntegrationStep) -> None:
		"""Consume one sequential accepted integration step without replaying its maps."""
		self._validate_next_record(record)

		assert record.quadrature is not None
		kappa = float(self._energy.accept(record.step_index, record.start_time,
			record.time, record.quadrature)[0])
		hamiltonian: float = _scalar_value(
			self._dynamics.hamiltonian(record.time, record.state_after),
			"Hamiltonian",
		)
		generalized_energy = hamiltonian + kappa
		error = generalized_energy - self._initial_energy
		self._records.append(
			GCGeneralizedEnergyRecord(
				step_index=record.step_index,
				time=float(record.time),
				duration=float(record.duration),
				hamiltonian=hamiltonian,
				kappa=kappa,
				generalized_energy=generalized_energy,
				energy_error=error,
				relative_error=error / self._energy_scale,
			)
		)


__all__ = [
	"GCGeneralizedEnergyObserver",
	"GCGeneralizedEnergyRecord",
]
