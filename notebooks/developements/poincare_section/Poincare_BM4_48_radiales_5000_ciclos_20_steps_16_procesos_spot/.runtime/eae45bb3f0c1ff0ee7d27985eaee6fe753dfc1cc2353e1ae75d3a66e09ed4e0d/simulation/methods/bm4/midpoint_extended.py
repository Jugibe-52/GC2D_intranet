"""Fully extended BM4 arithmetic projection with ABBA observer conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dynamics import GuidingCenterDynamics

from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
from ...formulations import gc_coupling_matrix
from ...formulations.base import Projection
from ...observation import IntegrationStep
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from ..abba.maps.extended import (
	_checked_extended_state, _flow_first, _flow_second, _synchronized_extended_time,
)
from ..abba.state import _fully_extended_energy_diagnostics
from ._core import _advance_composition

if TYPE_CHECKING:
	from .midpoint import BM4Midpoint


@dataclass(frozen=True, slots=True)
class _PreparedExtendedBM4:
	"""Direct/adjoint shears of two (x,y,t,k) copies, with spatial coupling."""

	dynamics: GuidingCenterDynamics
	coupling_frequency: float
	initial_internal_state: np.ndarray

	@property
	def dynamics_name(self) -> str:
		return type(self.dynamics).__name__

	def _couple(self, duration: float, state: np.ndarray) -> np.ndarray:
		"""Rotate the spatial copy difference, leaving both times and k unchanged."""
		value = _checked_extended_state(state, duplicated=True)
		indices = np.asarray([0, 1, 4, 5])
		value[indices] = gc_coupling_matrix(duration, self.coupling_frequency) @ value[indices]
		return value

	def direct_map(self, duration: float, t: float, state: np.ndarray) -> np.ndarray:
		"""Apply first and second shears, then the exact coupling flow."""
		value = _flow_first(self.dynamics, state, duration)
		value = _flow_second(self.dynamics, value, duration)
		return self._couple(duration, value)

	def adjoint_map(self, duration: float, t: float, state: np.ndarray) -> np.ndarray:
		"""Reverse the subflow ordering; time is intrinsic to each extended copy."""
		value = self._couple(duration, state)
		value = _flow_second(self.dynamics, value, duration)
		return _flow_first(self.dynamics, value, duration)

	def project(self, internal_history: np.ndarray) -> Projection:
		"""Expose the physical mean for the prepared-formulation contract."""
		return np.asarray((internal_history[:2] + internal_history[4:6]) / 2.0), {}


def _midpoint_extended_bm4_step(
	prepared: _PreparedExtendedBM4, state: np.ndarray, step: float,
) -> tuple[np.ndarray, float]:
	"""Apply the shared complete BM4 cycle and average both full copies."""
	value = _checked_extended_state(state, duplicated=False)
	mapped = _advance_composition(
		prepared, float(value[2]), np.concatenate((value, value)), step,
		step_index=0, stage_observer=None,
		formulation_name="FullyExtendedBM4", method_name="BM4Midpoint",
	)
	accepted = _synchronized_extended_time(
		(mapped[:4] + mapped[4:]) / 2.0, float(value[2] + step),
		context="The fully extended BM4 midpoint map",
	)
	return accepted, float(np.linalg.norm(mapped[:4] - mapped[4:], ord=np.inf))


def _integrate_bm4_fully_extended_midpoint(
	method: BM4Midpoint, problem: InitialValueProblem, request: SimulationRequest,
) -> IntegrationData:
	"""Integrate one GC particle in (x,y,t,k), returning its physical history."""
	dynamics = problem.dynamics
	if not isinstance(dynamics, GuidingCenterDynamics):
		raise TypeError("BM4Midpoint fully extended mode requires GuidingCenterDynamics.")
	if problem.initial_state.shape != (2,):
		raise ValueError("BM4Midpoint fully extended mode requires exactly one GC particle.")
	initial = np.concatenate((problem.initial_state, (request.t_span[0], 0.0)))
	prepared = _PreparedExtendedBM4(dynamics, method.coupling_frequency, np.tile(initial, 2))
	separations: list[float] = []

	def advance(
		t: float, state: np.ndarray, step: float, step_index: int, observe: bool,
	) -> np.ndarray:
		value = _synchronized_extended_time(state, t, context="The internal state")
		after, separation = _midpoint_extended_bm4_step(prepared, value, step)
		if observe:
			separations.append(separation)
			if method.step_observer is not None:
				def map_state(candidate: np.ndarray) -> np.ndarray:
					return _midpoint_extended_bm4_step(prepared, candidate, step)[0]

				method.step_observer(IntegrationStep(
					dynamics_name=type(dynamics).__name__, method_name=type(method).__name__,
					step_index=step_index, start_time=t, time=t + step, duration=step,
					state_before=value.copy(), state_after=after.copy(),
					map_state=map_state, dynamics=dynamics,
				))
		return after

	history, count = integrate_fixed_grid(
		initial, request, advance, progress=bool(method.progress), label=type(method).__name__,
	)
	history[2] = request.output_times
	diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
		"step_count": count, "copy_separation_norms": np.asarray(separations),
		"projection_kind": "arithmetic_mean", "state_extension": "fully_extended",
		"track_energy": True, "coupling_frequency": method.coupling_frequency,
		"accepted_internal_state_dimension": 4, "base_splitting_state_dimension": 8,
		"observer_state_dimension": 4, "observer_state_kind": "accepted_internal_map",
		"nonlinear_unknown_dimension": 0, "composition_stage_count": 12,
		"vector_field_evaluations_per_step": 24,
	}
	diagnostics.update(_fully_extended_energy_diagnostics(dynamics, history))
	return IntegrationData(t=request.output_times, states=np.asarray(history[:2]), diagnostics=diagnostics)


__all__: list[str] = []
