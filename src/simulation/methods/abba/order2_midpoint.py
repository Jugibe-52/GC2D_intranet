"""Midpoint ABBA integration with arithmetic-mean diagonal projection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics import DynamicalSystem, ExtendedHamiltonianSystem, GuidingCenterDynamics

from .maps.extended import _abba_base_map, _checked_extended_state, _synchronized_extended_time
from .state import _fully_extended_energy_diagnostics
from ...integration import IntegrationMethod, StepInfo, StepResult
from ..._result import DiagnosticValue
from ...observation import IntegrationStep, StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from ._configuration import (
	StateExtension,
	_resolved_track_energy,
	_state_dimension_diagnostics,
	_validate_state_extension,
)
from .maps.physical import _ABBAStages, _evaluate_unprojected_stages
from ._energy import (
	_conjugate_momentum_increment_from_stages,
	_energy_tracking_diagnostics,
	_energy_tracking_initial_state,
	_pack_energy_tracking_state,
	_unpack_energy_tracking_state,
	_validate_energy_tracking,
)


@dataclass(frozen=True, slots=True)
class _ABBA2MidpointStep:
	"""Physical average and off-diagonal copy separation after one ABBA map."""

	state: np.ndarray
	copy_separation_norm: float
	stages: _ABBAStages


def _midpoint_abba_step(
	dynamics: DynamicalSystem,
	t: float,
	state: np.ndarray,
	step: float,
) -> _ABBA2MidpointStep:
	"""Apply endpoint-time A-B-B-A and project both copies by their mean."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The ABBA physical state must be a finite, non-empty vector.")
	# Midpoint projection starts both copies on the physical diagonal, then uses
	# exactly the same endpoint-time A-B-B-A map as the implicit formulations.
	stages = _evaluate_unprojected_stages(
		dynamics,
		t,
		value,
		value,
		step,
	)
	return _ABBA2MidpointStep(
		state=np.asarray((stages.u_final + stages.v_final) / 2.0),
		copy_separation_norm=float(
			np.linalg.norm(stages.residual, ord=np.inf)
		),
		stages=stages,
	)


@dataclass(slots=True)
class ABBA2Midpoint(IntegrationMethod[_ABBA2MidpointStep | None]):
	"""Second-order midpoint ABBA with optional physical energy tracking.

	The method duplicates the selected physical or fully extended state, applies
	the endpoint-time A-B-B-A shears, and averages the two final copies. Tracking
	the physical conjugate momentum is an auxiliary triangular update that does
	not feed back into this map. Midpoint has no residual-formulation or
	nonlinear-solver axis.
	"""

	state_extension: StateExtension = "physical"
	progress: bool = False
	step_observer: StepObserver | None = None
	track_energy: bool = False

	# Runtime resources are initialized once by new_run, never constructor inputs.
	dynamics: DynamicalSystem = field(init=False, repr=False, compare=False)
	physical_size: int = field(init=False, repr=False, compare=False)
	particle_count: int = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Validate the state strategy and resolve inherent energy tracking."""
		self.state_extension = _validate_state_extension(self.state_extension)
		self.track_energy = _resolved_track_energy(self.track_energy, self.state_extension)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Initialize one physical or fully extended arithmetic-projection run."""
		self.dynamics = problem.dynamics
		self.physical_size = problem.initial_state.size
		self.particle_count = self.physical_size // 2
		if self.state_extension == "fully_extended":
			if not isinstance(self.dynamics, GuidingCenterDynamics):
				raise TypeError("ABBA2Midpoint fully extended mode requires GuidingCenterDynamics.")
			if problem.initial_state.shape != (2,):
				raise ValueError("ABBA2Midpoint fully extended mode requires exactly one GC particle.")
			self.initial_state = np.concatenate((problem.initial_state, (request.t_span[0], 0.0)))
		else:
			if not isinstance(self.dynamics, DynamicalSystem) or self.dynamics.state_dimension != 2:
				raise TypeError("ABBA2Midpoint requires planar two-component dynamics.")
			_validate_energy_tracking(self.dynamics, enabled=self.track_energy, method_name=self.method_name)
			self.initial_state = _energy_tracking_initial_state(
				problem.initial_state, particle_count=self.particle_count, enabled=self.track_energy,
			)
		metadata: dict[str, DiagnosticValue] = {
			"projection_kind": "arithmetic_mean", "state_extension": self.state_extension,
			"track_energy": self.track_energy, "nonlinear_unknown_dimension": 0,
			"vector_field_evaluations_per_step": 4,
		}
		metadata.update(_state_dimension_diagnostics(self.state_extension, particle_count=self.particle_count))
		if self.state_extension == "fully_extended":
			metadata["unprojected_abba_maps_per_step"] = 1
		self.metadata = metadata

	def advance(self, t: float, state: np.ndarray, h: float) -> StepResult[_ABBA2MidpointStep | None]:
		"""Finish one arithmetic projection and its auxiliary energy update."""
		if self.state_extension == "fully_extended":
			value = _synchronized_extended_time(state, t, context="The internal state")
			after, separation = self._extended_step(value, h)
			return StepResult(after, {"copy_separation_norms": separation}, None)
		physical, momentum = _unpack_energy_tracking_state(
			state, physical_size=self.physical_size, particle_count=self.particle_count, enabled=self.track_energy,
		)
		result = _midpoint_abba_step(self.dynamics, t, physical, h)
		after_momentum = momentum
		if momentum is not None:
			assert isinstance(self.dynamics, ExtendedHamiltonianSystem)
			after_momentum = momentum + _conjugate_momentum_increment_from_stages(
				self.dynamics, t, h, result.stages, particle_count=self.particle_count,
			)
		after = _pack_energy_tracking_state(result.state, after_momentum)
		return StepResult(after, {"copy_separation_norms": result.copy_separation_norm}, result)

	def _extended_step(self, state: np.ndarray, h: float) -> tuple[np.ndarray, float]:
		"""Apply the selected base map and average both complete (x,y,t,k) copies."""
		assert isinstance(self.dynamics, GuidingCenterDynamics)
		value = _checked_extended_state(state, duplicated=False)
		mapped = np.asarray(
			_abba_base_map(self.dynamics, h).map_state(np.concatenate((value, value))), dtype=float,
		)
		accepted = _synchronized_extended_time(
			np.asarray((mapped[:4] + mapped[4:]) / 2.0), float(value[2] + h),
			context="The fully extended midpoint map",
		)
		return accepted, float(np.linalg.norm(mapped[:4] - mapped[4:], ord=np.inf))

	def build_observation(self, info: StepInfo, step: StepResult[_ABBA2MidpointStep | None]) -> IntegrationStep:
		"""Expose independent snapshots in the selected observation domain."""
		if self.state_extension == "fully_extended":
			before = _synchronized_extended_time(info.state_before, info.time, context="The internal state")
			after = step.state
			def map_state(candidate: np.ndarray) -> np.ndarray:
				return self._extended_step(candidate, info.duration)[0]
		else:
			before, momentum = _unpack_energy_tracking_state(
				info.state_before, physical_size=self.physical_size,
				particle_count=self.particle_count, enabled=self.track_energy,
			)
			assert step.details is not None
			after = step.details.state
			def map_state(candidate: np.ndarray) -> np.ndarray:
				return _midpoint_abba_step(self.dynamics, info.time, candidate, info.duration).state
		return IntegrationStep(
			dynamics_name=type(self.dynamics).__name__, method_name=self.method_name,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=before.copy(), state_after=after.copy(),
			map_state=map_state, dynamics=self.dynamics,
		)

	def export_history(self, times: np.ndarray, history: np.ndarray) -> tuple[np.ndarray, dict[str, DiagnosticValue]]:
		"""Extract physical samples and energy diagnostics in their existing units."""
		if self.state_extension == "fully_extended":
			assert isinstance(self.dynamics, GuidingCenterDynamics)
			history[2] = times
			return np.asarray(history[:2]), _fully_extended_energy_diagnostics(self.dynamics, history)
		states = np.asarray(history[:self.physical_size])
		momentum = np.asarray(history[self.physical_size:]) if self.track_energy else None
		return states, _energy_tracking_diagnostics(times, states, momentum, self.dynamics)


__all__ = ["ABBA2Midpoint"]
