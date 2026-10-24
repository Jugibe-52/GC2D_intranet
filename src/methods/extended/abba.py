"""ABBA presets on the shared palindromic method and projection implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal
import numpy as np

from contracts.methods.observation import IntegrationStep
from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from methods.projection import ProjectionChoice
from request import SimulationRequest
from solvers.nonlinear import AlgebraicSolver, NewtonSolver
from methods.extended.configuration import (
	ProjectionPlacement, StateExtension,
	_validate_projection_formulation, _validate_projection_placement, _validate_state_extension,
)
from methods.extended.core.composition import ABBA2, ABBA4, ABBA6, Composition
from methods.extended.core.records import ProjectedMapResult
from methods.extended.observations import EventBuilder, bind_event_builder
from methods.base import FixedStepMethod
from methods.extended.palindromic import ProjectedPalindromicMethod, MidpointPalindromicMethod


@dataclass(slots=True)
class _ABBAImplicitMethod(ProjectedPalindromicMethod):
	"""Project one complete palindromic recipe; concrete methods select its order."""

	solver: AlgebraicSolver = field(default_factory=lambda: NewtonSolver(jacobian_method='analytic'), kw_only=True)
	projection_formulation: ProjectionChoice = 'reduced_multiplier'
	state_extension: StateExtension = 'physical'
	progress: bool = False
	order: ClassVar[Literal[2, 4, 6]]
	build_event: EventBuilder | None = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		FixedStepMethod.__post_init__(self)
		self._validate_solver_controls()
		_validate_projection_formulation(self.projection_formulation)
		self.state_extension = _validate_state_extension(self.state_extension)

	def initialize(self, request: SimulationRequest) -> None:
		"""Bind the complete recipe to one reduced or simultaneous projection."""
		self._prepare_projection(request, self.projection_formulation, probe_analytic=True)
		# Equal adjoint/direct weights describe unprojected pairs, not extra solves.
		coefficients: tuple[float, ...] = tuple(2 * c for c in self.recipe.coefficients[::2])
		self.build_event = (bind_event_builder(self.method_name,
			_validate_projection_formulation(self.projection_formulation), order=self.order,
			coefficients=coefficients, project=self.project) if self.observe else None)
		metadata: dict[str, DiagnosticValue] = {
			**self._projection_metadata(self.projection_formulation),
			'state_extension': self.state_extension,
			'composition_stage_count': len(self.recipe.coefficients),
		}
		if self.order != 2:
			metadata.update({
				'implicit_substeps_per_step': 1,
				'composition_coefficients': np.asarray(coefficients),
				'unprojected_abba_maps_per_step': len(coefficients),
				'unprojected_abba_maps_per_residual_evaluation': len(coefficients),
				'base_composition': 'unprojected_abba4_triple_jump' if self.order == 4 else 'unprojected_abba6_yoshida',
			})
		self.metadata = metadata

	def _publish_substeps(self) -> bool:
		return self.order != 2

	def build_observation(self, info: StepInfo, step: StepResult[ProjectedMapResult]) -> IntegrationStep:
		"""Expose physical snapshots from the accepted projection trace."""
		assert self.build_event is not None
		return self.build_event(info.time, info.duration, info.index, info.state_before, step.details)


@dataclass(slots=True)
class ABBA2Implicit(_ABBAImplicitMethod):
	"""Second-order ABBA pair with one symmetric spatial projection."""

	order: ClassVar[Literal[2, 4, 6]] = 2
	recipe: Composition = field(default=ABBA2, init=False)


@dataclass(slots=True)
class _ComposedABBAImplicit(_ABBAImplicitMethod):
	"""Shared placement selector for higher-order ABBA compositions."""

	projection_placement: ProjectionPlacement = 'around_complete_composition'

	def __post_init__(self) -> None:
		_ABBAImplicitMethod.__post_init__(self)
		self.projection_placement = _validate_projection_placement(self.projection_placement)


@dataclass(slots=True)
class ABBA4Implicit(_ComposedABBAImplicit):
	"""Fourth-order triple jump with one projection around all six stages."""

	order: ClassVar[Literal[2, 4, 6]] = 4
	recipe: Composition = field(default=ABBA4, init=False)


@dataclass(slots=True)
class ABBA6Implicit(_ComposedABBAImplicit):
	"""Sixth-order Yoshida recipe with one projection around fourteen stages."""

	order: ClassVar[Literal[2, 4, 6]] = 6
	recipe: Composition = field(default=ABBA6, init=False)


@dataclass(slots=True)
class ABBA2Midpoint(MidpointPalindromicMethod):
	"""Second-order ABBA palindrome followed by one arithmetic mean."""

	state_extension: StateExtension = 'physical'
	progress: bool = False
	recipe: Composition = field(default=ABBA2, init=False)


__all__ = ['ABBA2Implicit', 'ABBA4Implicit', 'ABBA6Implicit', 'ABBA2Midpoint']
