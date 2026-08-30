"""Midpoint ABBA integration with arithmetic-mean diagonal projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import DynamicalSystem

from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
from ...observation import IntegrationStep, StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from ._core import _evaluate_unprojected_stages


@dataclass(frozen=True, slots=True)
class _MidpointABBAStep:
	"""Physical average and off-diagonal copy separation after one ABBA map."""

	state: np.ndarray
	copy_separation_norm: float


def _midpoint_abba_step(
	dynamics: DynamicalSystem,
	t: float,
	state: np.ndarray,
	step: float,
) -> _MidpointABBAStep:
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
	return _MidpointABBAStep(
		state=np.asarray((stages.u_final + stages.v_final) / 2.0),
		copy_separation_norm=float(
			np.linalg.norm(stages.residual, ord=np.inf)
		),
	)


@dataclass(frozen=True, slots=True)
class MidpointABBA:
	"""Second-order midpoint ABBA method with arithmetic mean projection.

	The method duplicates the physical state, applies the endpoint-time A-B-B-A
	shears, and averages the two final copies. The average is an inexpensive
	Euclidean projection, but it does not guarantee a symplectic physical map.
	"""

	progress: bool = False
	step_observer: StepObserver | None = None

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate one planar physical problem and retain copy separation."""
		dynamics = problem.dynamics
		if not isinstance(dynamics, DynamicalSystem):
			raise TypeError("MidpointABBA requires DynamicalSystem.")
		if dynamics.state_dimension != 2:
			raise TypeError("MidpointABBA requires planar two-component dynamics.")

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

			state_before = np.asarray(state, dtype=float)
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
			return result.state

		history, step_count = integrate_fixed_grid(
			problem.initial_state,
			request,
			advance,
			progress=bool(self.progress),
			label=type(self).__name__,
		)
		return IntegrationData(
			t=request.output_times,
			states=np.asarray(history),
			diagnostics={
				"step_count": step_count,
				"copy_separation_norms": np.asarray(
					copy_separation_norms,
					dtype=float,
				),
				"projection_kind": "arithmetic_mean",
				"vector_field_evaluations_per_step": 4,
			},
		)


__all__ = ["MidpointABBA"]
