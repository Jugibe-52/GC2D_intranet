"""BM4 integration with one arithmetic-mean projection per complete cycle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import DynamicalSystem

from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
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
from .midpoint_extended import _integrate_bm4_fully_extended_midpoint


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


@dataclass(frozen=True, slots=True)
class BM4Midpoint:
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

	def __post_init__(self) -> None:
		"""Validate state strategy and coupling; resolve inherent energy tracking."""
		extension = _validate_state_extension(self.state_extension)
		object.__setattr__(self, "state_extension", extension)
		object.__setattr__(self, "track_energy", _resolved_track_energy(self.track_energy, extension))
		frequency = GCExtendedFormulation(self.coupling_frequency).coupling_frequency
		object.__setattr__(self, "coupling_frequency", frequency)

	def integrate(
		self, problem: InitialValueProblem, request: SimulationRequest,
	) -> IntegrationData:
		"""Use the shared fixed grid and observe only accepted main-grid steps."""
		if self.state_extension == "fully_extended":
			return _integrate_bm4_fully_extended_midpoint(self, problem, request)
		dynamics = problem.dynamics
		if not isinstance(dynamics, DynamicalSystem) or dynamics.state_dimension != 2:
			raise TypeError("BM4Midpoint requires planar two-component dynamics.")
		prepared = GCExtendedFormulation(self.coupling_frequency).prepare(
			problem, track_energy=self.track_energy,
		)
		physical_size = problem.initial_state.size
		particle_count = physical_size // 2
		separations: list[float] = []

		def advance(
			t: float, state: np.ndarray, step: float, step_index: int, observe: bool,
		) -> np.ndarray:
			physical, momentum = _unpack_energy_tracking_state(
				state, physical_size=physical_size, particle_count=particle_count,
				enabled=self.track_energy,
			)
			result = _midpoint_bm4_step(prepared, t, physical, step, momentum)
			if observe:
				separations.append(result.copy_separation_norm)
				if self.step_observer is not None:
					def map_state(candidate: np.ndarray) -> np.ndarray:
						return _midpoint_bm4_step(prepared, t, candidate, step, momentum).state

					self.step_observer(IntegrationStep(
						dynamics_name=type(dynamics).__name__, method_name=type(self).__name__,
						step_index=step_index, start_time=t, time=t + step, duration=step,
						state_before=physical.copy(), state_after=result.state.copy(),
						map_state=map_state, dynamics=dynamics,
					))
			return _pack_energy_tracking_state(result.state, result.momentum)

		initial = _energy_tracking_initial_state(
			problem.initial_state, particle_count=particle_count, enabled=self.track_energy,
		)
		history, count = integrate_fixed_grid(
			initial, request, advance, progress=bool(self.progress), label=type(self).__name__,
		)
		states = np.asarray(history[:physical_size])
		momentum = np.asarray(history[physical_size:]) if self.track_energy else None
		diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
			"step_count": count, "copy_separation_norms": np.asarray(separations),
			"projection_kind": "arithmetic_mean", "state_extension": self.state_extension,
			"track_energy": self.track_energy, "coupling_frequency": self.coupling_frequency,
			"composition_stage_count": 12, "vector_field_evaluations_per_step": 24,
			"nonlinear_unknown_dimension": 0,
		}
		diagnostics.update(_state_dimension_diagnostics("physical", particle_count=particle_count))
		diagnostics.update(_energy_tracking_diagnostics(request.output_times, states, momentum, dynamics))
		return IntegrationData(t=request.output_times, states=states, diagnostics=diagnostics)


__all__ = ["BM4Midpoint"]
