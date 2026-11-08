"""Fourth-order triple jump with one projection around all six stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal
import numpy as np
from contracts.methods.observation import IntegrationStep
from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from methods.projection import ProjectionChoice
from solvers.nonlinear import AlgebraicSolver, NewtonSolver
from methods.palindromic.configuration import ProjectionPlacement, StateExtension, _validate_projection_formulation, _validate_projection_placement, _validate_state_extension
from methods.palindromic.core.operations import A, B, T
from methods.palindromic.core.records import ProjectedMapResult
from methods.palindromic.observations import EventBuilder, bind_event_builder
from methods.palindromic.projected_base import ProjectedPalindromicMethod

_CUBE_ROOT_TWO: float = float(np.cbrt(2.0))
_GAMMA: float = 1.0 / (2.0 - _CUBE_ROOT_TWO)
_DELTA: float = -_CUBE_ROOT_TWO / (2.0 - _CUBE_ROOT_TWO)
_ABBA4_COEFFICIENTS: np.ndarray = np.asarray((_GAMMA, _DELTA, _GAMMA))


@dataclass(slots=True)
class ABBA4Implicit(ProjectedPalindromicMethod):
	"""Fourth-order triple jump with one projection around all six stages."""

	solver: AlgebraicSolver = field(default_factory=lambda: NewtonSolver(jacobian_method='analytic'), kw_only=True)
	projection_formulation: ProjectionChoice = 'reduced_multiplier'
	state_extension: StateExtension = 'physical'
	projection_placement: ProjectionPlacement = 'around_complete_composition'
	order: ClassVar[Literal[2, 4]] = 4
	composition_name = 'ABBA4'
	recipe = (
		A(_GAMMA/2), B(_GAMMA/2), T(_GAMMA), B(_GAMMA/2), A(_GAMMA/2),
		A(_DELTA/2), B(_DELTA/2), T(_DELTA), B(_DELTA/2), A(_DELTA/2),
		A(_GAMMA/2), B(_GAMMA/2), T(_GAMMA), B(_GAMMA/2), A(_GAMMA/2),
	)

	def __post_init__(self) -> None:
		ProjectedPalindromicMethod.__post_init__(self)
		self.state_extension = _validate_state_extension(self.state_extension)
		self.projection_placement = _validate_projection_placement(self.projection_placement)

	def _projection_metadata(self, projection: ProjectionChoice) -> dict[str, DiagnosticValue]:
		return {**ProjectedPalindromicMethod._projection_metadata(self, projection),
			'state_extension': self.state_extension, 'composition_stage_count': 6,
			'implicit_substeps_per_step': 1, 'composition_coefficients': _ABBA4_COEFFICIENTS,
			'unprojected_abba_maps_per_step': 3,
			'unprojected_abba_maps_per_residual_evaluation': 3,
			'base_composition': 'unprojected_abba4_triple_jump'}

	def _publish_substeps(self) -> bool:
		return True

	def build_observation(self, info: StepInfo, step: StepResult[ProjectedMapResult]) -> IntegrationStep:
		"""Retain the three unprojected pairs and the single accepted multiplier."""
		build: EventBuilder = bind_event_builder(self.method_name, _validate_projection_formulation(self.projection_formulation),
			order=4, coefficients=(_GAMMA, _DELTA, _GAMMA), project=self.project)
		return build(info.time, info.duration, info.index, info.state_before, step.details)


__all__ = ['ABBA4Implicit']
