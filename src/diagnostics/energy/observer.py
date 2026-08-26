"""Accepted-step reconstruction of the GC time-conjugate momentum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import GuidingCenterDynamics
from simulation import (
	ImplicitABBA4IntegrationStep,
	ImplicitABBAIntegrationStep,
	ImplicitBM4IntegrationStep,
	IntegrationStage,
	IntegrationStep,
	gc_coupling_matrix,
)


def _scalar_value(value: np.ndarray, name: str) -> float:
	"""Return the unique finite scalar produced by a one-particle evaluation."""
	array = np.asarray(value, dtype=float)
	if array.size != 1 or not np.all(np.isfinite(array)):
		raise ValueError(f"The one-particle {name} must be one finite scalar.")
	return float(array.reshape(-1)[0])


def _momentum_derivative(
	dynamics: GuidingCenterDynamics,
	time: float,
	state: np.ndarray,
) -> float:
	"""Evaluate ``-partial_t h`` for one physical GC state."""
	return _scalar_value(
		dynamics.extended_momentum_derivative(float(time), np.asarray(state)),
		"extended-momentum derivative",
	)


def _abba_kappa_increment(
	record: ImplicitABBAIntegrationStep,
	dynamics: GuidingCenterDynamics,
) -> float:
	"""Reconstruct the normalized momentum increment of four ABBA shears."""
	half_step = record.duration / 2.0
	start = record.start_time
	stop = record.time
	doubled_increment = half_step * (
		_momentum_derivative(dynamics, start, record.v_initial)
		+ _momentum_derivative(dynamics, start, record.u_first)
		+ _momentum_derivative(dynamics, stop, record.u_first)
		+ _momentum_derivative(dynamics, stop, record.v_final)
	)
	# The duplicated Hamiltonian contains h(u) + h(v) + k. Projection onto
	# u=v exposes the physical conjugate variable kappa=k/2.
	return doubled_increment / 2.0


def _split_doubled_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Split one duplicated one-particle GC state into its two physical copies."""
	value = np.asarray(state, dtype=float)
	if value.shape != (4,) or not np.all(np.isfinite(value)):
		raise ValueError("A one-particle doubled GC state must have shape (4,).")
	return value[:2], value[2:]


def _bm4_stage_kappa_increment(
	stage: IntegrationStage,
	dynamics: GuidingCenterDynamics,
	*,
	coupling_frequency: float,
) -> float:
	"""Reconstruct ``Delta kappa`` for one direct or adjoint BM4 stage."""
	first, second = _split_doubled_state(stage.state_before)
	duration = stage.duration
	time = stage.time
	if stage.flow_name == "flow":
		second_after = second + duration * dynamics.vector_field(time, first)
		doubled_increment = duration * (
			_momentum_derivative(dynamics, time, first)
			+ _momentum_derivative(dynamics, time, second_after)
		)
	elif stage.flow_name == "adjoint_flow":
		blocks = gc_coupling_matrix(duration, coupling_frequency) @ np.concatenate(
			(first, second)
		)
		coupled_first, coupled_second = blocks[:2], blocks[2:]
		first_after = coupled_first + duration * dynamics.vector_field(
			time,
			coupled_second,
		)
		doubled_increment = duration * (
			_momentum_derivative(dynamics, time, coupled_second)
			+ _momentum_derivative(dynamics, time, first_after)
		)
	else:
		raise ValueError(f"Unsupported BM4 flow name {stage.flow_name!r}.")
	return doubled_increment / 2.0


def _kappa_increment(
	record: IntegrationStep,
	dynamics: GuidingCenterDynamics,
) -> float:
	"""Dispatch the accepted-step momentum reconstruction by record type."""
	if isinstance(record, ImplicitABBA4IntegrationStep):
		return float(
			sum(_abba_kappa_increment(substep, dynamics) for substep in record.substeps)
		)
	if isinstance(record, ImplicitABBAIntegrationStep):
		return _abba_kappa_increment(record, dynamics)
	if isinstance(record, ImplicitBM4IntegrationStep):
		if len(record.base_stages) != 12:
			raise ValueError("An accepted implicit BM4 step must expose twelve stages.")
		return float(
			sum(
				_bm4_stage_kappa_increment(
					stage,
					dynamics,
					coupling_frequency=record.coupling_frequency,
				)
				for stage in record.base_stages
			)
		)
	raise TypeError(
		"Generalized-energy reconstruction supports ImplicitABBA1, "
		"ABBA4Implicit1, and BM4Implicit1 step records."
	)


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
	"""Accumulate ``kappa`` and ``K=h+kappa`` from accepted implicit stages.

	The observer is deliberately restricted to one GC particle. The implicit
	methods evolve only the projected physical state, so the conjugate variable is
	reconstructed from the exact accepted stage snapshots without perturbing the
	trajectory or adding shadow-step observations.
	"""

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
		state = np.asarray(initial_state, dtype=float)
		if state.shape != (2,) or not np.all(np.isfinite(state)):
			raise ValueError("Energy reconstruction requires one finite planar state.")
		time = float(initial_time)
		if not np.isfinite(time):
			raise ValueError("`initial_time` must be finite.")
		hamiltonian = _scalar_value(
			dynamics.hamiltonian(time, state),
			"Hamiltonian",
		)
		self._dynamics = dynamics
		self._initial_energy = hamiltonian
		self._energy_scale = max(abs(hamiltonian), float(np.finfo(float).eps))
		self._kappa = 0.0
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

	def __call__(self, record: IntegrationStep) -> None:
		"""Consume one sequential accepted implicit integration step."""
		if not isinstance(record, IntegrationStep):
			raise TypeError("The energy observer requires an IntegrationStep record.")
		if record.dynamics is not self._dynamics:
			raise ValueError("The observed step belongs to a different GC system.")
		expected_index = len(self._records) - 1
		if record.step_index != expected_index:
			raise ValueError("Energy records must arrive in sequential step order.")
		previous_time = self._records[-1].time
		tolerance = float(64.0 * np.finfo(float).eps) * max(
			1.0,
			abs(previous_time),
			abs(record.start_time),
		)
		if not np.isclose(
			record.start_time,
			previous_time,
			rtol=0.0,
			atol=tolerance,
		):
			raise ValueError("Energy records must describe a continuous time grid.")

		self._kappa += _kappa_increment(record, self._dynamics)
		hamiltonian = _scalar_value(
			self._dynamics.hamiltonian(record.time, record.state_after),
			"Hamiltonian",
		)
		generalized_energy = hamiltonian + self._kappa
		error = generalized_energy - self._initial_energy
		self._records.append(
			GCGeneralizedEnergyRecord(
				step_index=record.step_index,
				time=float(record.time),
				duration=float(record.duration),
				hamiltonian=hamiltonian,
				kappa=self._kappa,
				generalized_energy=generalized_energy,
				energy_error=error,
				relative_error=error / self._energy_scale,
			)
		)


__all__ = [
	"GCGeneralizedEnergyObserver",
	"GCGeneralizedEnergyRecord",
]
