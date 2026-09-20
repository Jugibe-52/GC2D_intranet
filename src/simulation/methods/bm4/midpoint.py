"""BM4 integration with one arithmetic-mean projection per complete cycle."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics import DynamicalSystem, GuidingCenterDynamics

from ...integration import IntegrationMethod, StepInfo, StepResult
from ..._result import DiagnosticValue
from ...formulations import GCExtendedFormulation
from ...formulations.base import PreparedDirectAdjointFormulation
from ...observation import IntegrationStep, StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from ..abba._configuration import (
	StateExtension, _resolved_track_energy, _state_dimension_diagnostics,
	_validate_state_extension,
)
from ..abba._energy import (
	_energy_tracking_diagnostics, _energy_tracking_initial_state,
	_pack_energy_tracking_state, _unpack_energy_tracking_state,
)
from ._core import _advance_composition
from .midpoint_extended import _PreparedExtendedBM4, _midpoint_extended_bm4_step
from ..abba.maps.extended import _synchronized_extended_time
from ..abba.state import _fully_extended_energy_diagnostics


@dataclass(frozen=True, slots=True)
class _BM4MidpointStep:
	"""Accepted physical mean, copy separation and optional projected momentum."""

	state: np.ndarray
	copy_separation_norm: float
	momentum: np.ndarray | None


def _midpoint_bm4_step(
	prepared: PreparedDirectAdjointFormulation,
	t: float,
	state: np.ndarray,
	step: float,
	momentum: np.ndarray | None = None,
) -> _BM4MidpointStep:
	"""Duplicate z, advance all twelve stages, then accept their mean once."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The BM4 physical state must be a finite, non-empty vector.")
	# Prepared GC stages evolve the summed momentum k=2*kappa. Its update is
	# triangular and cannot affect either physical copy or their coupling.
	internal = np.concatenate((value, value))
	if momentum is not None:
		internal = np.concatenate((internal, 2.0 * momentum))
	mapped = _advance_composition(
		prepared, t, internal, step, step_index=0, stage_observer=None,
		formulation_name="GCExtendedFormulation", method_name="BM4Midpoint",
	)
	if not np.all(np.isfinite(mapped)):
		raise ValueError("The BM4 midpoint map became non-finite.")
	first, second = mapped[:value.size], mapped[value.size:2 * value.size]
	return _BM4MidpointStep(
		(first + second) / 2.0,
		float(np.linalg.norm(first - second, ord=np.inf)),
		mapped[2 * value.size:] / 2.0 if momentum is not None else None,
	)


@dataclass(slots=True)
class BM4Midpoint(IntegrationMethod[_BM4MidpointStep | None]):
	"""Fourth-order BM4 followed by arithmetic-mean diagonal projection.

	Configuration and observer semantics follow ABBA2Midpoint. Physical mode
	accepts packed (x_1,...,x_p,y_1,...,y_p) states with optional auxiliary
	energy tracking. Fully extended mode accepts one internal (x,y,t,k) state
	and inherently tracks energy. Neither mode solves a nonlinear equation.
	The temporary copies use the same harmonic coupling frequency as BM4Implicit.
	Arithmetic projection does not imply exact symplecticity or reversibility.
	"""

	state_extension: StateExtension = "physical"
	progress: bool = False
	step_observer: StepObserver | None = None
	track_energy: bool = False
	coupling_frequency: float = np.pi / 8

	# Runtime resources are initialized once by new_run, never constructor inputs.
	dynamics: DynamicalSystem = field(init=False, repr=False, compare=False)
	physical_size: int = field(init=False, repr=False, compare=False)
	particle_count: int = field(init=False, repr=False, compare=False)
	formulation: PreparedDirectAdjointFormulation = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Validate state strategy and coupling; resolve inherent energy tracking."""
		extension = _validate_state_extension(self.state_extension)
		self.state_extension = extension
		self.track_energy = _resolved_track_energy(self.track_energy, extension)
		frequency = GCExtendedFormulation(self.coupling_frequency).coupling_frequency
		self.coupling_frequency = frequency

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Initialize one physical or fully extended arithmetic-projection run."""
		self.dynamics = problem.dynamics
		self.physical_size = problem.initial_state.size
		self.particle_count = self.physical_size // 2
		if self.state_extension == "fully_extended":
			if not isinstance(self.dynamics, GuidingCenterDynamics):
				raise TypeError("BM4Midpoint fully extended mode requires GuidingCenterDynamics.")
			if problem.initial_state.shape != (2,):
				raise ValueError("BM4Midpoint fully extended mode requires exactly one GC particle.")
			self.initial_state = np.concatenate((problem.initial_state, (request.t_span[0], 0.0)))
			self.formulation = _PreparedExtendedBM4(
				self.dynamics, self.coupling_frequency, np.tile(self.initial_state, 2),
			)
		else:
			if not isinstance(self.dynamics, DynamicalSystem) or self.dynamics.state_dimension != 2:
				raise TypeError("BM4Midpoint requires planar two-component dynamics.")
			self.formulation = GCExtendedFormulation(self.coupling_frequency).prepare(
				problem, track_energy=self.track_energy,
			)
			self.initial_state = _energy_tracking_initial_state(
				problem.initial_state, particle_count=self.particle_count, enabled=self.track_energy,
			)
		metadata: dict[str, DiagnosticValue] = {
			"projection_kind": "arithmetic_mean", "state_extension": self.state_extension,
			"track_energy": self.track_energy, "nonlinear_unknown_dimension": 0,
			"coupling_frequency": self.coupling_frequency, "composition_stage_count": 12,
			"vector_field_evaluations_per_step": 24,
		}
		metadata.update(_state_dimension_diagnostics(self.state_extension, particle_count=self.particle_count))
		self.metadata = metadata

	def advance(self, t: float, state: np.ndarray, h: float) -> StepResult[_BM4MidpointStep | None]:
		"""Finish one arithmetic projection and its auxiliary energy update."""
		if self.state_extension == "fully_extended":
			value = _synchronized_extended_time(state, t, context="The internal state")
			after, separation = self._extended_step(value, h)
			return StepResult(after, {"copy_separation_norms": separation}, None)
		physical, momentum = _unpack_energy_tracking_state(
			state, physical_size=self.physical_size, particle_count=self.particle_count, enabled=self.track_energy,
		)
		result = _midpoint_bm4_step(self.formulation, t, physical, h, momentum)
		after = _pack_energy_tracking_state(result.state, result.momentum)
		return StepResult(after, {"copy_separation_norms": result.copy_separation_norm}, result)

	def _extended_step(self, state: np.ndarray, h: float) -> tuple[np.ndarray, float]:
		"""Apply the selected base map and average both complete (x,y,t,k) copies."""
		assert isinstance(self.formulation, _PreparedExtendedBM4)
		return _midpoint_extended_bm4_step(self.formulation, state, h)

	def build_observation(self, info: StepInfo, step: StepResult[_BM4MidpointStep | None]) -> IntegrationStep:
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
				return _midpoint_bm4_step(self.formulation, info.time, candidate, info.duration, momentum).state
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


__all__ = ["BM4Midpoint"]
