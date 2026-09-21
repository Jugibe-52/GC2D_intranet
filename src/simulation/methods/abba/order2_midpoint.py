"""Midpoint ABBA integration with arithmetic-mean diagonal projection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics import DynamicalSystem, ExtendedHamiltonianSystem, GuidingCenterDynamics

from ...formulations.state import DoubledFormulation
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
from ._energy import _conjugate_momentum_increment_from_stages


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
class ABBA2Midpoint(IntegrationMethod[_ABBA2MidpointStep]):
	"""Second-order midpoint ABBA with optional physical energy tracking.

	The method duplicates only the physical state, applies
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
	state_formulation: DoubledFormulation = field(init=False, repr=False, compare=False)
	dynamics: DynamicalSystem = field(init=False, repr=False, compare=False)
	physical_size: int = field(init=False, repr=False, compare=False)
	particle_count: int = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Validate the state strategy and resolve inherent energy tracking."""
		self.state_extension = _validate_state_extension(self.state_extension)
		self.track_energy = _resolved_track_energy(self.track_energy, self.state_extension)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Initialize one spatial arithmetic-projection run with optional passive energy."""
		self.dynamics = problem.dynamics
		self.physical_size = problem.initial_state.size
		self.particle_count = self.physical_size // 2
		if not isinstance(self.dynamics, DynamicalSystem) or self.dynamics.state_dimension != 2:
			raise TypeError("ABBA2Midpoint requires planar two-component dynamics.")
		self.state_formulation = DoubledFormulation(problem, request.t_span[0], self.track_energy)
		self.initial_state = self.state_formulation.initial_state
		metadata: dict[str, DiagnosticValue] = {
			"projection_kind": "arithmetic_mean", "state_extension": self.state_extension,
			"track_energy": self.track_energy, "nonlinear_unknown_dimension": 0,
			"vector_field_evaluations_per_step": 4,
		}
		metadata.update(_state_dimension_diagnostics(self.state_extension, particle_count=self.particle_count))
		self.metadata = metadata

	def advance(self, t: float, state: np.ndarray, h: float) -> StepResult[_ABBA2MidpointStep]:
		"""Project the spatial copies and independently accumulate their energy balance."""
		result = _midpoint_abba_step(self.dynamics, t, self.state_formulation.physical(state), h)
		increment = None
		if self.track_energy:
			assert isinstance(self.dynamics, ExtendedHamiltonianSystem)
			increment = _conjugate_momentum_increment_from_stages(self.dynamics, t, h, result.stages,
			    particle_count=self.state_formulation.particle_count)
		after = self.state_formulation.finish(state, result.state, t + h, increment)
		return StepResult(after, {"copy_separation_norms": result.copy_separation_norm}, result)

	def build_observation(self, info: StepInfo, step: StepResult[_ABBA2MidpointStep]) -> IntegrationStep:
		"""Observe only the physical map, independently of energy tracking."""
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return _midpoint_abba_step(self.dynamics, info.time, candidate, info.duration).state
		return IntegrationStep(
			dynamics_name=type(self.dynamics).__name__, method_name=self.method_name,
			step_index=info.index, start_time=info.time, time=info.end_time,
			duration=info.duration, state_before=self.state_formulation.physical(info.state_before).copy(),
			state_after=step.details.state.copy(), map_state=map_state, dynamics=self.dynamics)


__all__ = ["ABBA2Midpoint"]
