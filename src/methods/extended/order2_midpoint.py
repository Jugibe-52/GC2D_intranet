"""Midpoint ABBA integration with arithmetic-mean diagonal projection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics import DynamicalSystem
from formulations.gc import GCDoubledMaps
from methods.extended.composition import ABBA2
from methods.extended.midpoint import midpoint_step, MidpointResult as _ABBA2MidpointStep
from methods.extended.abba_maps import uncoupled_maps
from methods.extended.energy import momentum_increment

from formulations.state import DoubledFormulation
from integration.core import IntegrationMethod
from contracts.step import StepInfo, StepResult
from contracts.result import DiagnosticValue
from contracts.observation import IntegrationStep, StepObserver
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from methods.extended.configuration import (
	StateExtension,
	_resolved_track_energy,
	_state_dimension_diagnostics,
	_validate_state_extension,
)
from methods.extended.abba_maps import _ABBAStages, _evaluate_unprojected_stages


def _midpoint_abba_step(dynamics: DynamicalSystem, t: float, state: np.ndarray,
                        step: float) -> _ABBA2MidpointStep:
    """Compatibility entry point using the common arithmetic projection."""
    return midpoint_step(uncoupled_maps(dynamics, state), ABBA2, t, state, step)


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
	formulation: GCDoubledMaps = field(init=False, repr=False, compare=False)
	physical_size: int = field(init=False, repr=False, compare=False)
	particle_count: int = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Validate the state strategy and resolve inherent energy tracking."""
		self.state_extension = _validate_state_extension(self.state_extension)
		self.track_energy = _resolved_track_energy(self.track_energy, self.state_extension)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Initialize one spatial arithmetic-projection run with optional passive energy."""
		self.dynamics = problem.dynamics
		self.formulation = GCDoubledMaps(problem, coupling_frequency=None)
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
		result = midpoint_step(self.formulation, ABBA2, t, self.state_formulation.physical(state), h)
		increment = None
		if self.track_energy:
			increment = momentum_increment(self.state_formulation, result.trace.energy_points)
		after = self.state_formulation.finish(state, result.state, t + h, increment)
		return StepResult(after, {"copy_separation_norms": result.copy_separation_norm}, result)

	def build_observation(self, info: StepInfo, step: StepResult[_ABBA2MidpointStep]) -> IntegrationStep:
		"""Observe only the physical map, independently of energy tracking."""
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return midpoint_step(self.formulation, ABBA2, info.time, candidate, info.duration).state
		return IntegrationStep(
			dynamics_name=type(self.dynamics).__name__, method_name=self.method_name,
			step_index=info.index, start_time=info.time, time=info.end_time,
			duration=info.duration, state_before=self.state_formulation.physical(info.state_before).copy(),
			state_after=step.details.state.copy(), map_state=map_state, dynamics=self.dynamics)


__all__ = ["ABBA2Midpoint"]
