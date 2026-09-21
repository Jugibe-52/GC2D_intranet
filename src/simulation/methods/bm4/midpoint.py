"""BM4 integration with one arithmetic-mean projection per complete cycle."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics import DynamicalSystem, GuidingCenterDynamics

from ...formulations.state import DoubledFormulation
from ...integration import IntegrationMethod, StepInfo, StepResult
from ..._result import DiagnosticValue
from ...formulations.gc import GCDoubledMaps
from ...formulations.base import PreparedDirectAdjointFormulation
from ...observation import IntegrationStep, StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from ..abba._configuration import (
	StateExtension, _resolved_track_energy, _state_dimension_diagnostics,
	_validate_state_extension,
)

from ._core import _advance_composition


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
class BM4Midpoint(IntegrationMethod[_BM4MidpointStep]):
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
	state_formulation: DoubledFormulation = field(init=False, repr=False, compare=False)
	dynamics: DynamicalSystem = field(init=False, repr=False, compare=False)
	physical_size: int = field(init=False, repr=False, compare=False)
	particle_count: int = field(init=False, repr=False, compare=False)
	formulation: GCDoubledMaps = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Validate state strategy and coupling; resolve inherent energy tracking."""
		extension = _validate_state_extension(self.state_extension)
		self.state_extension = extension
		self.track_energy = _resolved_track_energy(self.track_energy, extension)
		frequency = float(self.coupling_frequency)
		if not np.isfinite(frequency) or frequency < 0:
			raise ValueError("`coupling_frequency` must be finite and non-negative.")
		self.coupling_frequency = frequency

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Initialize one spatial arithmetic-projection run with optional passive energy."""
		self.dynamics = problem.dynamics
		self.physical_size = problem.initial_state.size
		self.particle_count = self.physical_size // 2
		self.state_formulation = DoubledFormulation(problem, request.t_span[0], self.track_energy)
		self.formulation = GCDoubledMaps(problem, self.coupling_frequency, track_energy=self.track_energy)
		self.initial_state = self.state_formulation.initial_state
		metadata: dict[str, DiagnosticValue] = {
			"projection_kind": "arithmetic_mean", "state_extension": self.state_extension,
			"track_energy": self.track_energy, "nonlinear_unknown_dimension": 0,
			"coupling_frequency": self.coupling_frequency, "composition_stage_count": 12,
			"vector_field_evaluations_per_step": 24,
		}
		metadata.update(_state_dimension_diagnostics(self.state_extension, particle_count=self.particle_count))
		self.metadata = metadata

	def advance(self, t: float, state: np.ndarray, h: float) -> StepResult[_BM4MidpointStep]:
		"""Apply the BM4 composition with one spatial average and passive momentum."""
		result = _midpoint_bm4_step(self.formulation, t, self.state_formulation.physical(state), h,
		                            self.state_formulation.momentum(state))
		after = self.state_formulation.pack(result.state, t + h, result.momentum)
		return StepResult(after, {"copy_separation_norms": result.copy_separation_norm}, result)

	def build_observation(self, info: StepInfo, step: StepResult[_BM4MidpointStep]) -> IntegrationStep:
		"""Observe the physical map using independent spatial snapshots."""
		momentum = self.state_formulation.momentum(info.state_before)
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return _midpoint_bm4_step(self.formulation, info.time, candidate, info.duration, momentum).state
		return IntegrationStep(
			dynamics_name=type(self.dynamics).__name__, method_name=self.method_name,
			step_index=info.index, start_time=info.time, time=info.end_time,
			duration=info.duration, state_before=self.state_formulation.physical(info.state_before).copy(),
			state_after=step.details.state.copy(), map_state=map_state, dynamics=self.dynamics)


__all__ = ["BM4Midpoint"]
