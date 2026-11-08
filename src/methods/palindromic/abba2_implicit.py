"""Second-order ABBA pair with one symmetric spatial projection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal
from contracts.methods.observation import IntegrationStep
from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from methods.projection import ProjectionChoice
from solvers.nonlinear import AlgebraicSolver, NewtonSolver
from methods.palindromic.configuration import StateExtension, _validate_projection_formulation, _validate_state_extension
from methods.palindromic.core.operations import A, B, T
from methods.palindromic.core.records import ProjectedMapResult
from methods.palindromic.observations import EventBuilder, bind_event_builder
from methods.palindromic.projected_base import ProjectedPalindromicMethod


@dataclass(slots=True)
class ABBA2Implicit(ProjectedPalindromicMethod):
	"""Second-order ABBA pair with one symmetric spatial projection."""

	solver: AlgebraicSolver = field(default_factory=lambda: NewtonSolver(jacobian_method='analytic'), kw_only=True)
	projection_formulation: ProjectionChoice = 'reduced_multiplier'
	state_extension: StateExtension = 'physical'
	order: ClassVar[Literal[2, 4]] = 2
	composition_name = 'ABBA2'
	recipe = (A(.5), B(.5), T(1.), B(.5), A(.5))

	def __post_init__(self) -> None:
		ProjectedPalindromicMethod.__post_init__(self)
		self.state_extension = _validate_state_extension(self.state_extension)

	def _projection_metadata(self, projection: ProjectionChoice) -> dict[str, DiagnosticValue]:
		return {**ProjectedPalindromicMethod._projection_metadata(self, projection),
			'state_extension': self.state_extension, 'composition_stage_count': 2}

	def build_observation(self, info: StepInfo, step: StepResult[ProjectedMapResult]) -> IntegrationStep:
		"""Expose the accepted four shears only when observation is requested."""
		build: EventBuilder = bind_event_builder(self.method_name, _validate_projection_formulation(self.projection_formulation),
			order=2, coefficients=(1.,), project=self.project)
		return build(info.time, info.duration, info.index, info.state_before, step.details)


__all__ = ['ABBA2Implicit']
