"""BM4 presets and observation adapters on the common palindromic method."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np

from contracts.methods.extended.bm4 import ImplicitBM4IntegrationStep
from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from solvers.nonlinear import AlgebraicSolver, NewtonSolver
from methods.extended.configuration import StateExtension
from methods.projection import ProjectionChoice
from methods.extended.core.operations import A, B, T
from methods.extended.core.records import ProjectedMapResult
from methods.extended.observations import stage_events
from methods.extended.palindromic import ProjectedPalindromicMethod, MidpointPalindromicMethod

if TYPE_CHECKING:
	from contracts.methods.observation import IntegrationStage


_A1: float = 0.0792036964311957
_A2: float = 0.1303114101821663
_A3: float = 0.2228614958676077
_A4: float = -0.3667132690474257
_A5: float = 0.3246481886897062
_A6: float = 0.1096884778767498


@dataclass(slots=True)
class BM4Midpoint(MidpointPalindromicMethod):
	"""Uncoupled fourth-order BM4 palindrome with one final arithmetic mean."""

	state_extension: StateExtension = 'physical'
	progress: bool = False
	composition_name = 'BM4'
	recipe = (
		A(_A1), B(_A1), T(_A1+_A2), B(_A2), A(_A2),
		A(_A3), B(_A3), T(_A3+_A4), B(_A4), A(_A4),
		A(_A5), B(_A5), T(_A5+_A6), B(_A6), A(_A6),
		A(_A6), B(_A6), T(_A6+_A5), B(_A5), A(_A5),
		A(_A4), B(_A4), T(_A4+_A3), B(_A3), A(_A3),
		A(_A2), B(_A2), T(_A2+_A1), B(_A1), A(_A1),
	)

	def _preset_metadata(self) -> dict[str, DiagnosticValue]:
		return {'coupling_frequency': 0.0, 'composition_stage_count': 12}


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
	progress: bool = False

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


__all__ = ['BM4Implicit', 'BM4Midpoint']
