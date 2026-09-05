"""Midpoint ABBA integration with arithmetic-mean diagonal projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import DynamicalSystem, ExtendedHamiltonianSystem

from .._fully_extended import _integrate_abba_fully_extended_midpoint
from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
from ...observation import IntegrationStep, StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from ._configuration import (
	StateExtension,
	_resolved_track_energy,
	_state_dimension_diagnostics,
	_validate_state_extension,
)
from ._core import _ABBAStages, _evaluate_unprojected_stages
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


@dataclass(frozen=True, slots=True)
class ABBA2Midpoint:
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

	def __post_init__(self) -> None:
		"""Validate the state strategy and resolve inherent energy tracking."""
		object.__setattr__(
			self,
			"state_extension",
			_validate_state_extension(self.state_extension),
		)
		object.__setattr__(
			self,
			"track_energy",
			_resolved_track_energy(self.track_energy, self.state_extension),
		)

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate one planar problem and retain the final copy separation."""
		if self.state_extension == "fully_extended":
			return _integrate_abba_fully_extended_midpoint(self, problem, request)
		dynamics = problem.dynamics
		if not isinstance(dynamics, DynamicalSystem):
			raise TypeError("ABBA2Midpoint requires DynamicalSystem.")
		if dynamics.state_dimension != 2:
			raise TypeError("ABBA2Midpoint requires planar two-component dynamics.")
		_validate_energy_tracking(
			dynamics,
			enabled=self.track_energy,
			method_name=type(self).__name__,
		)
		physical_size = problem.initial_state.size
		particle_count = physical_size // dynamics.state_dimension

		copy_separation_norms: list[float] = []

		def advance(
			t: float,
			state: np.ndarray,
			step: float,
			step_index: int,
			observe: bool,
		) -> np.ndarray:
			def apply_step(candidate: np.ndarray) -> np.ndarray:
				"""Apply the same fixed-time midpoint map to one candidate state."""
				return _midpoint_abba_step(dynamics, t, candidate, step).state

			state_before, momentum_before = _unpack_energy_tracking_state(
				state,
				physical_size=physical_size,
				particle_count=particle_count,
				enabled=self.track_energy,
			)
			result = _midpoint_abba_step(dynamics, t, state_before, step)
			if observe:
				copy_separation_norms.append(result.copy_separation_norm)
				if self.step_observer is not None:
					self.step_observer(
						IntegrationStep(
							dynamics_name=type(dynamics).__name__,
							method_name=type(self).__name__,
							step_index=step_index,
							start_time=t,
							time=t + step,
							duration=step,
							state_before=state_before.copy(),
							state_after=result.state.copy(),
							map_state=apply_step,
							dynamics=dynamics,
						)
					)
			momentum_after = momentum_before
			if momentum_before is not None:
				assert isinstance(dynamics, ExtendedHamiltonianSystem)
				momentum_after = (
					momentum_before
					+ _conjugate_momentum_increment_from_stages(
						dynamics,
						t,
						step,
						result.stages,
						particle_count=particle_count,
					)
				)
			return _pack_energy_tracking_state(result.state, momentum_after)

		initial_state = _energy_tracking_initial_state(
			problem.initial_state,
			particle_count=particle_count,
			enabled=self.track_energy,
		)
		history, step_count = integrate_fixed_grid(
			initial_state,
			request,
			advance,
			progress=bool(self.progress),
			label=type(self).__name__,
		)
		states = np.asarray(history[:physical_size])
		momentum = (
			np.asarray(history[physical_size:]) if self.track_energy else None
		)
		diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
			"step_count": step_count,
			"copy_separation_norms": np.asarray(
				copy_separation_norms,
				dtype=float,
			),
			"projection_kind": "arithmetic_mean",
			"state_extension": self.state_extension,
			"track_energy": self.track_energy,
			"vector_field_evaluations_per_step": 4,
		}
		diagnostics.update(
			_state_dimension_diagnostics(
				self.state_extension,
				particle_count=particle_count,
			)
		)
		diagnostics["nonlinear_unknown_dimension"] = 0
		diagnostics.update(
			_energy_tracking_diagnostics(
				request.output_times,
				states,
				momentum,
				dynamics,
			)
		)
		return IntegrationData(
			t=request.output_times,
			states=states,
			diagnostics=diagnostics,
		)


__all__ = ["ABBA2Midpoint"]
