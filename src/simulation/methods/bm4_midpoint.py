"""BM4 integration with one arithmetic-mean projection per complete cycle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._fixed import integrate_fixed_grid
from .._result import IntegrationData
from ..formulations import GCStageProjectedFormulation
from ..observation import StageObserver
from ..problem import InitialValueProblem
from ..request import SimulationRequest
from .bm4 import _advance_composition


@dataclass(frozen=True, slots=True)
class MidpointBM4:
	"""Fourth-order uncoupled BM4 with full-cycle midpoint projection.

	Each complete step starts with two equal copies of the physical GC state.
	The method applies all twelve uncoupled BM4 stages before replacing both
	copies by their arithmetic mean. This full-cycle projection is distinct from
	``ProjectedBM4Composition``, which projects after every internal stage.
	"""

	progress: bool = False
	stage_observer: StageObserver | None = None

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate a GC problem and retain pre-projection copy separation."""
		formulation = GCStageProjectedFormulation()
		prepared = formulation.prepare(problem, track_energy=False)
		physical_size = problem.initial_state.size
		copy_separation_norms: list[float] = []

		def advance(
			t: float,
			state: np.ndarray,
			step: float,
			step_index: int,
			observe: bool,
		) -> np.ndarray:
			unprojected = _advance_composition(
				prepared,
				t,
				state,
				step,
				step_index=step_index,
				stage_observer=self.stage_observer if observe else None,
				stage_projection=None,
				formulation_name=type(formulation).__name__,
				method_name=type(self).__name__,
			)
			first = unprojected[:physical_size]
			second = unprojected[physical_size : 2 * physical_size]
			if observe:
				copy_separation_norms.append(
					float(np.linalg.norm(first - second, ord=np.inf))
				)

			# Re-embedding only here preserves one projection per complete BM4
			# cycle, including shadow cycles used to sample off-grid output times.
			projected = np.asarray(prepared.project_internal_state(unprojected))
			if projected.shape != unprojected.shape:
				raise ValueError(
					"The full-cycle midpoint projection changed the internal state shape."
				)
			return projected

		internal_history, step_count = integrate_fixed_grid(
			prepared.initial_internal_state,
			request,
			advance,
			progress=bool(self.progress),
			label=type(self).__name__,
		)
		states, diagnostics = prepared.project(internal_history)
		diagnostics.update(
			{
				"step_count": step_count,
				"copy_separation_norms": np.asarray(
					copy_separation_norms,
					dtype=float,
				),
				"projection_kind": "arithmetic_mean",
				"projection_scope": "complete_bm4_cycle",
				"projections_per_step": 1,
				"vector_field_evaluations_per_step": 24,
			}
		)
		return IntegrationData(
			t=request.output_times,
			states=np.asarray(states),
			diagnostics=diagnostics,
		)


__all__ = ["MidpointBM4"]
