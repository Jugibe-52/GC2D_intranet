"""Extended ``(u, v, t, k)`` symplecticity for accepted implicit splittings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from dynamics import GuidingCenterDynamics
from simulation import (
	ImplicitABBA4IntegrationStep,
	ImplicitABBAIntegrationStep,
	ImplicitBM4IntegrationStep,
	IntegrationStep,
	gc_coupling_matrix,
)

from diagnostics.jacobians import central_difference_jacobian

from .observer import (
	gc_reduced_time_extended_symplectic_form,
	gc_time_extended_symplectic_form,
)


ExtendedMap = Callable[[np.ndarray], np.ndarray]
ReducedStepMap = Callable[[float, np.ndarray, float], tuple[np.ndarray, float]]


def _finite_extended_state(state: np.ndarray) -> np.ndarray:
	"""Return a copied finite one-particle state in ``(u, v, t, k)`` order."""
	value = np.asarray(state, dtype=float)
	if value.shape != (6,) or not np.all(np.isfinite(value)):
		raise ValueError("A time-extended one-particle state must have shape (6,).")
	return value.copy()


def _momentum_derivative(
	dynamics: GuidingCenterDynamics,
	time: float,
	state: np.ndarray,
) -> float:
	"""Return the scalar ``-partial_t h`` for one physical GC copy."""
	value = np.asarray(
		dynamics.extended_momentum_derivative(float(time), state),
		dtype=float,
	)
	if value.size != 1 or not np.all(np.isfinite(value)):
		raise ValueError("The one-particle momentum derivative must be finite.")
	return float(value.reshape(-1)[0])


def _abba_extended_map(
	record: ImplicitABBAIntegrationStep,
	dynamics: GuidingCenterDynamics,
) -> ExtendedMap:
	"""Return the accepted unprojected ABBA splitting on ``R^6``."""
	duration = float(record.duration)
	half_step = duration / 2.0

	def map_state(candidate: np.ndarray) -> np.ndarray:
		value = _finite_extended_state(candidate)
		u = value[:2].copy()
		v = value[2:4].copy()
		time = float(value[4])
		momentum = float(value[5])

		u += half_step * dynamics.vector_field(time, v)
		momentum += half_step * _momentum_derivative(dynamics, time, v)
		v += half_step * dynamics.vector_field(time, u)
		momentum += half_step * _momentum_derivative(dynamics, time, u)
		time += duration
		v += half_step * dynamics.vector_field(time, u)
		momentum += half_step * _momentum_derivative(dynamics, time, u)
		u += half_step * dynamics.vector_field(time, v)
		momentum += half_step * _momentum_derivative(dynamics, time, v)
		return np.concatenate((u, v, (time, momentum)))

	return map_state


def _bm4_extended_stage(
	candidate: np.ndarray,
	*,
	flow_name: str,
	duration: float,
	coupling_frequency: float,
	dynamics: GuidingCenterDynamics,
) -> np.ndarray:
	"""Apply one direct or adjoint BM4 base stage on ``(u, v, t, k)``."""
	value = _finite_extended_state(candidate)
	u = value[:2].copy()
	v = value[2:4].copy()
	time = float(value[4])
	momentum = float(value[5])

	if flow_name == "flow":
		time += duration
		v += duration * dynamics.vector_field(time, u)
		momentum += duration * _momentum_derivative(dynamics, time, u)
		u += duration * dynamics.vector_field(time, v)
		momentum += duration * _momentum_derivative(dynamics, time, v)
		blocks = gc_coupling_matrix(duration, coupling_frequency) @ np.concatenate(
			(u, v)
		)
		u, v = blocks[:2], blocks[2:]
	elif flow_name == "adjoint_flow":
		blocks = gc_coupling_matrix(duration, coupling_frequency) @ np.concatenate(
			(u, v)
		)
		u, v = blocks[:2], blocks[2:]
		u += duration * dynamics.vector_field(time, v)
		momentum += duration * _momentum_derivative(dynamics, time, v)
		v += duration * dynamics.vector_field(time, u)
		momentum += duration * _momentum_derivative(dynamics, time, u)
		time += duration
	else:
		raise ValueError(f"Unsupported BM4 flow name {flow_name!r}.")
	return np.concatenate((u, v, (time, momentum)))


def _bm4_extended_map(
	record: ImplicitBM4IntegrationStep,
	dynamics: GuidingCenterDynamics,
) -> ExtendedMap:
	"""Return the complete accepted unprojected twelve-stage BM4 cycle."""
	stage_specs = tuple(
		(stage.flow_name, float(stage.duration)) for stage in record.base_stages
	)
	frequency = float(record.coupling_frequency)

	def map_state(candidate: np.ndarray) -> np.ndarray:
		value = _finite_extended_state(candidate)
		for flow_name, duration in stage_specs:
			value = _bm4_extended_stage(
				value,
				flow_name=flow_name,
				duration=duration,
				coupling_frequency=frequency,
				dynamics=dynamics,
			)
		return value

	return map_state


@dataclass(frozen=True, slots=True)
class GCTimeExtendedSymplecticityRecord:
	"""Maximum ``R^6`` splitting-map defect associated with one accepted step."""

	step_index: int
	time: float
	duration: float
	base_map_count: int
	scope: str
	maximum_relative_defect: float
	mean_relative_defect: float
	maximum_defect_frobenius: float
	maximum_abs_defect: float
	maximum_determinant_error: float
	maximum_condition_number: float


def _measure_symplectic_map(
	map_state: ExtendedMap,
	state: np.ndarray,
	*,
	form: np.ndarray,
	relative_step: float | None,
) -> tuple[float, float, float, float, float]:
	"""Return defect, determinant, and conditioning metrics for one map."""
	jacobian = central_difference_jacobian(
		map_state,
		state,
		relative_step=relative_step,
	)
	defect = jacobian.T @ form @ jacobian - form
	defect_frobenius = float(np.linalg.norm(defect, ord="fro"))
	return (
		defect_frobenius / float(np.linalg.norm(form, ord="fro")),
		defect_frobenius,
		float(np.max(np.abs(defect))),
		abs(float(np.linalg.det(jacobian)) - 1.0),
		float(np.linalg.cond(jacobian)),
	)


class GCTimeExtendedSymplecticityObserver:
	"""Measure the accepted splitting before diagonal Hairer projection.

	For ``ImplicitABBA1`` and ``BM4Implicit1`` one record measures the complete
	unprojected base cycle. ``ABBA4Implicit1`` projects between its three signed
	ABBA substeps, so its record aggregates the three legitimate ``R^6`` base-map
	measurements instead of pretending that the dimension-reducing projection is
	itself a map from ``R^6`` to ``R^6``.
	"""

	def __init__(
		self,
		dynamics: GuidingCenterDynamics,
		*,
		relative_step: float | None = None,
	) -> None:
		"""Configure sequential accepted-step numerical differentiation."""
		if not isinstance(dynamics, GuidingCenterDynamics):
			raise TypeError("`dynamics` must be GuidingCenterDynamics.")
		if relative_step is not None and (
			not np.isfinite(float(relative_step)) or float(relative_step) <= 0.0
		):
			raise ValueError("`relative_step` must be positive and finite.")
		self._dynamics = dynamics
		self.relative_step = relative_step
		self._records: list[GCTimeExtendedSymplecticityRecord] = []

	@property
	def records(self) -> tuple[GCTimeExtendedSymplecticityRecord, ...]:
		"""Return immutable scalar records for all accepted steps."""
		return tuple(self._records)

	def __call__(self, record: IntegrationStep) -> None:
		"""Measure the supported splitting map or maps represented by ``record``."""
		if not isinstance(record, IntegrationStep):
			raise TypeError("The observer requires an IntegrationStep record.")
		if record.dynamics is not self._dynamics:
			raise ValueError("The observed step belongs to a different GC system.")
		if record.step_index != len(self._records):
			raise ValueError("Extended symplecticity records must be sequential.")

		maps_and_states: list[tuple[ExtendedMap, np.ndarray]] = []
		if isinstance(record, ImplicitABBA4IntegrationStep):
			for substep in record.substeps:
				map_state = _abba_extended_map(substep, self._dynamics)
				state = np.concatenate(
					(substep.u_initial, substep.v_initial, (substep.start_time, 0.0))
				)
				maps_and_states.append((map_state, state))
			scope = "three accepted ABBA base maps; inter-substep projections excluded"
		elif isinstance(record, ImplicitABBAIntegrationStep):
			map_state = _abba_extended_map(record, self._dynamics)
			state = np.concatenate(
				(record.u_initial, record.v_initial, (record.start_time, 0.0))
			)
			maps_and_states.append((map_state, state))
			scope = "complete accepted ABBA base map; final projection excluded"
		elif isinstance(record, ImplicitBM4IntegrationStep):
			if len(record.base_stages) != 12:
				raise ValueError("An accepted implicit BM4 step must expose twelve stages.")
			map_state = _bm4_extended_map(record, self._dynamics)
			state = np.concatenate(
				(
					record.base_stages[0].state_before,
					(record.start_time, 0.0),
				)
			)
			maps_and_states.append((map_state, state))
			scope = "complete accepted twelve-stage BM4 base cycle; projection excluded"
		else:
			raise TypeError(
				"The observer supports ImplicitABBA1, ABBA4Implicit1, and "
				"BM4Implicit1 step records."
			)

		metrics = np.asarray(
			[
				_measure_symplectic_map(
					map_state,
					state,
					form=gc_time_extended_symplectic_form(),
					relative_step=self.relative_step,
				)
				for map_state, state in maps_and_states
			],
			dtype=float,
		)
		self._records.append(
			GCTimeExtendedSymplecticityRecord(
				step_index=record.step_index,
				time=float(record.time),
				duration=float(record.duration),
				base_map_count=len(maps_and_states),
				scope=scope,
				maximum_relative_defect=float(np.max(metrics[:, 0])),
				mean_relative_defect=float(np.mean(metrics[:, 0])),
				maximum_defect_frobenius=float(np.max(metrics[:, 1])),
				maximum_abs_defect=float(np.max(metrics[:, 2])),
				maximum_determinant_error=float(np.max(metrics[:, 3])),
				maximum_condition_number=float(np.max(metrics[:, 4])),
			)
		)


@dataclass(frozen=True, slots=True)
class GCReducedTimeExtendedSymplecticityRecord:
	"""``R^4`` defect for one complete projected physical GC step."""

	step_index: int
	time: float
	duration: float
	scope: str
	relative_defect: float
	defect_frobenius: float
	max_abs_defect: float
	determinant_error: float
	condition_number: float


class GCReducedTimeExtendedSymplecticityObserver:
	"""Measure the projected method on physical ``(x, y, t, kappa)``.

	``step_map`` must repeat the complete implicit projected step for arbitrary
	physical state and start time and return both the final physical state and
	the accepted discrete ``Delta kappa``. Differentiating that callable includes
	the state/time dependence of the nonlinear projection multiplier.
	"""

	def __init__(
		self,
		dynamics: GuidingCenterDynamics,
		*,
		step_map: ReducedStepMap,
		relative_step: float | None = None,
	) -> None:
		"""Configure the complete projected-step differentiation."""
		if not isinstance(dynamics, GuidingCenterDynamics):
			raise TypeError("`dynamics` must be GuidingCenterDynamics.")
		if not callable(step_map):
			raise TypeError("`step_map` must be callable.")
		if relative_step is not None and (
			not np.isfinite(float(relative_step)) or float(relative_step) <= 0.0
		):
			raise ValueError("`relative_step` must be positive and finite.")
		self._dynamics = dynamics
		self._step_map = step_map
		self.relative_step = relative_step
		self._records: list[GCReducedTimeExtendedSymplecticityRecord] = []

	@property
	def records(self) -> tuple[GCReducedTimeExtendedSymplecticityRecord, ...]:
		"""Return immutable records for all accepted projected steps."""
		return tuple(self._records)

	def __call__(self, record: IntegrationStep) -> None:
		"""Differentiate one complete projected map in four dimensions."""
		if not isinstance(record, IntegrationStep):
			raise TypeError("The observer requires an IntegrationStep record.")
		if record.dynamics is not self._dynamics:
			raise ValueError("The observed step belongs to a different GC system.")
		if record.step_index != len(self._records):
			raise ValueError("Reduced symplecticity records must be sequential.")
		state_before = np.asarray(record.state_before, dtype=float)
		if state_before.shape != (2,) or not np.all(np.isfinite(state_before)):
			raise ValueError("Reduced symplecticity requires one planar GC state.")
		duration = float(record.duration)

		def map_state(candidate: np.ndarray) -> np.ndarray:
			value = np.asarray(candidate, dtype=float)
			if value.shape != (4,) or not np.all(np.isfinite(value)):
				raise ValueError(
					"A reduced time-extended GC state must have shape (4,)."
				)
			physical_after, kappa_increment = self._step_map(
				float(value[2]),
				value[:2],
				duration,
			)
			physical = np.asarray(physical_after, dtype=float)
			increment = float(kappa_increment)
			if (
				physical.shape != (2,)
				or not np.all(np.isfinite(physical))
				or not np.isfinite(increment)
			):
				raise ValueError("The reduced projected step returned invalid values.")
			return np.concatenate(
				(physical, (float(value[2]) + duration, float(value[3]) + increment))
			)

		initial = np.concatenate((state_before, (record.start_time, 0.0)))
		if not self._records:
			mapped_initial = map_state(initial)
			if not np.allclose(
				mapped_initial[:2],
				record.state_after,
				rtol=2e-12,
				atol=2e-13,
			):
				raise RuntimeError(
					"The replayed reduced map differs from the accepted physical step."
				)
		metrics = _measure_symplectic_map(
			map_state,
			initial,
			form=gc_reduced_time_extended_symplectic_form(),
			relative_step=self.relative_step,
		)
		self._records.append(
			GCReducedTimeExtendedSymplecticityRecord(
				step_index=record.step_index,
				time=float(record.time),
				duration=duration,
				scope=(
					"complete projected physical map on (x, y, t, kappa); "
					"implicit projection included"
				),
				relative_defect=metrics[0],
				defect_frobenius=metrics[1],
				max_abs_defect=metrics[2],
				determinant_error=metrics[3],
				condition_number=metrics[4],
			)
		)
__all__ = [
	"GCTimeExtendedSymplecticityObserver",
	"GCTimeExtendedSymplecticityRecord",
	"GCReducedTimeExtendedSymplecticityObserver",
	"GCReducedTimeExtendedSymplecticityRecord",
]
