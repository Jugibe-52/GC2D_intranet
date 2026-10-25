"""BM4 presets and observation adapters on the common palindromic method."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np

from contracts.methods.extended.bm4 import ImplicitBM4IntegrationStep
from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from request import SimulationRequest
from solvers.nonlinear import AlgebraicSolver, NewtonSolver
from methods.extended.configuration import StateExtension
from methods.extended.core.composition import BM4, Composition
from methods.extended.core.records import ProjectedMapResult
from methods.extended.observations import stage_events
from methods.base import FixedStepMethod
from methods.extended.palindromic import ProjectedPalindromicMethod, MidpointPalindromicMethod

if TYPE_CHECKING:
	from contracts.methods.observation import IntegrationStage


@dataclass(slots=True)
class BM4Implicit(ProjectedPalindromicMethod):
	"""Uncoupled BM4 with one reduced Hairer projection per complete cycle.

	The accepted state is physical. Temporary copies and projection equations
	belong to the method; nonlinear controls and observation belong to the solver.
	Broyden retains its initial reduced Jacobian 4I independently of derivative policy.
	"""

	recipe: Composition = field(default=BM4, init=False)
	solver: AlgebraicSolver = field(default_factory=lambda: NewtonSolver(jacobian_method='analytic'), kw_only=True)
	progress: bool = False

	def __post_init__(self) -> None:
		FixedStepMethod.__post_init__(self)
		self._validate_solver_controls()

	def initialize(self, request: SimulationRequest) -> None:
		self._prepare_projection(request, 'reduced_multiplier')
		self.metadata = {**self._projection_metadata('reduced_multiplier'),
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


@dataclass(slots=True)
class BM4Midpoint(MidpointPalindromicMethod):
	"""Uncoupled fourth-order BM4 palindrome with one final arithmetic mean."""

	state_extension: StateExtension = 'physical'
	progress: bool = False
	recipe: Composition = field(default=BM4, init=False)

	def _preset_metadata(self) -> dict[str, DiagnosticValue]:
		return {'coupling_frequency': 0.0, 'composition_stage_count': 12}


__all__ = ['BM4Implicit', 'BM4Midpoint']
