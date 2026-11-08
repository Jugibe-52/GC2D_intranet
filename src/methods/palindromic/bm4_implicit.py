"""Uncoupled BM4 with one reduced Hairer projection per complete cycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
from contracts.methods.palindromic.bm4 import ImplicitBM4IntegrationStep
from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from solvers.nonlinear import AlgebraicSolver, NewtonSolver
from methods.projection import ProjectionChoice
from methods.palindromic.core.records import ProjectedMapResult
from methods.palindromic.observations import stage_events
from methods.palindromic.projected_base import ProjectedPalindromicMethod

from methods.palindromic.bm4_midpoint import BM4Midpoint

if TYPE_CHECKING:
	from contracts.methods.observation import IntegrationStage


@dataclass(slots=True)
class BM4Implicit(ProjectedPalindromicMethod):
	"""Uncoupled BM4 with one reduced Hairer projection per complete cycle.

	The accepted state is physical. Temporary copies and projection equations
	belong to the method; nonlinear controls and observation belong to the solver.
	Broyden retains its initial reduced Jacobian 4I independently of derivative policy.
	"""

	composition_name = 'BM4'
	recipe = BM4Midpoint.recipe
	solver: AlgebraicSolver = field(default_factory=lambda: NewtonSolver(jacobian_method='analytic'), kw_only=True)

	def _projection_metadata(self, projection: ProjectionChoice) -> dict[str, DiagnosticValue]:
		return {**ProjectedPalindromicMethod._projection_metadata(self, projection),
			'projection_solver_formulation': 'bm4_implicit_reduced'}

	def build_observation(self, info: StepInfo, step: StepResult[ProjectedMapResult]) -> ImplicitBM4IntegrationStep:
		"""Reconstruct converged stage snapshots only for a requested event."""
		result = step.details
		base_stages: tuple[IntegrationStage, ...] = stage_events(self.formulation, result.trace,
			step_index=info.index, formulation_name='GCDoubledMaps', method_name=self.method_name)

		def map_state(candidate: np.ndarray) -> np.ndarray:
			return np.asarray(self.project(info.time, candidate, info.duration).state)

		return ImplicitBM4IntegrationStep(
			dynamics_name='', method_name=self.method_name,
			step_index=info.index, start_time=info.time, time=info.time + info.duration, duration=info.duration,
			state_before=info.state_before.copy(), state_after=result.state.copy(), map_state=map_state,
			formulation_name='bm4_implicit_reduced', nonlinear_solver=self.solver.name,
			newton_iterations=result.iterations, residual_evaluations=result.residual_evaluations,
			newton_residual_norm=result.residual_norm, newton_tolerance=result.stats.tolerance,
			projection_multiplier_norm=float(step.statistics['projection_multiplier_norms']), coupling_frequency=0.,
			multiplier=np.asarray(result.multiplier.copy()), base_stages=base_stages,
		)


__all__ = ['BM4Implicit']
