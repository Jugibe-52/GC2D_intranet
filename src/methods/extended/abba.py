"""ABBA presets on the shared palindromic method and projection implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal, TYPE_CHECKING
import numpy as np

from contracts.solvers.nonlinear import NewtonObserver
from contracts.methods.observation import IntegrationStep, StepObserver
from problem import InitialValueProblem
from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from contracts.dynamics.protocols import GuidingCenterJacobianSystem
from methods.projection import ProjectionChoice
from formulations.state import DoubleFormulation
from request import SimulationRequest
from solvers.nonlinear import AlgebraicSolver, NonlinearSolver
from methods.extended.configuration import (
	ProjectionPlacement, StateExtension, _state_dimension_diagnostics,
	_validate_projection_formulation, _validate_projection_placement, _validate_state_extension,
)
from methods.extended.core.composition import ABBA2, ABBA4, ABBA6, Composition
from methods.extended.core.jacobians import _checked_vector_field_jacobian
from contracts.methods.extended.core.records import ProjectedMap
from methods.extended.core.records import ProjectedMapResult
from methods.extended.observations import EventBuilder, bind_event_builder
from methods.base import FixedStepMethod
from methods.extended.palindromic import ProjectedPalindromicMethod, MidpointPalindromicMethod


if TYPE_CHECKING:
	from solvers.nonlinear import SolverOptions


@dataclass(slots=True)
class _ABBAImplicitMethod(ProjectedPalindromicMethod):
	"""Project one complete palindromic recipe; concrete methods select its order."""

	jax_mode = "device_resident"

	solver: AlgebraicSolver | None = field(default=None, kw_only=True)

	projection_formulation: ProjectionChoice = "reduced_multiplier"
	state_extension: StateExtension = "physical"
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	nonlinear_solver: NonlinearSolver = "newton"
	progress: bool = False
	step_observer: StepObserver | None = None
	track_energy: bool = False
	newton_observer: NewtonObserver | None = field(default=None, kw_only=True)

	order: ClassVar[Literal[2, 4, 6]]
	state_formulation: DoubleFormulation = field(init=False, repr=False, compare=False)
	build_event: EventBuilder | None = field(init=False, repr=False, compare=False)
	project: ProjectedMap = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Validate public ABBA controls using the common projected-method base."""
		FixedStepMethod.__post_init__(self)
		self._validate_solver_controls()
		_validate_projection_formulation(self.projection_formulation)
		self.state_extension = _validate_state_extension(self.state_extension)
		self.track_energy = bool(self.track_energy)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Bind the complete recipe to one reduced or simultaneous projection."""
		dynamics = problem.dynamics
		name = type(self).__name__
		if not isinstance(dynamics, GuidingCenterJacobianSystem) or dynamics.state_dimension != 2:
			raise TypeError(f"{name} requires planar GuidingCenterJacobianSystem dynamics.")
		options: SolverOptions = self._prepare_projection(problem, request, self.projection_formulation)
		if options.solver == "newton":
			_checked_vector_field_jacobian(dynamics, request.t_span[0], problem.initial_state)
		# Each ABBA pair contains equal adjoint/direct weights. Pair durations
		# describe unprojected stages, never independent nonlinear solves.
		coefficients: tuple[float, ...] = tuple(2 * c for c in self.recipe.coefficients[::2])
		self.build_event = None
		if self.step_observer is not None:
			self.build_event = bind_event_builder(
				dynamics, name, _validate_projection_formulation(self.projection_formulation), order=self.order,
				coefficients=coefficients, project=self.project,
			)
		metadata: dict[str, DiagnosticValue] = {
			"nonlinear_solves_per_step": 1,
			"nonlinear_solver": options.solver,
			"nonlinear_absolute_tolerance": options.absolute_tolerance,
			"nonlinear_relative_tolerance": options.relative_tolerance,
			"nonlinear_max_iterations": options.max_iterations,
			"projection_formulation": _validate_projection_formulation(self.projection_formulation),
			"projection_placement": "around_complete_composition",
			"state_extension": self.state_extension,
			"track_energy": self.track_energy,
			"composition_stage_count": len(self.recipe.coefficients),
		}
		metadata.update(_state_dimension_diagnostics(
			_validate_projection_formulation(self.projection_formulation),
			particle_count=problem.particle_count,
		))
		if self.order != 2:
			coefficient_array: np.ndarray = np.asarray(coefficients)
			metadata.update({
				"implicit_substeps_per_step": 1,
				"composition_coefficients": coefficient_array,
				"unprojected_abba_maps_per_step": len(coefficients),
				"unprojected_abba_maps_per_residual_evaluation": len(coefficients),
				"base_composition": (
					"unprojected_abba4_triple_jump" if self.order == 4
					else "unprojected_abba6_yoshida"
				),
			})
		self.metadata = {**metadata, **self._operation_metadata()}

	def _publish_substeps(self) -> bool:
		return self.order != 2

	def build_observation(self, info: StepInfo, step: StepResult[ProjectedMapResult]) -> IntegrationStep:
		"""Expose physical snapshots from the accepted projection trace."""
		assert self.build_event is not None
		before: np.ndarray = info.state_before
		return self.build_event(info.time, info.duration, info.index, before, step.details)


@dataclass(slots=True)
class ABBA2Implicit(_ABBAImplicitMethod):
	"""Second-order ABBA pair with one symmetric spatial projection."""

	order: ClassVar[Literal[2, 4, 6]] = 2
	recipe: Composition = field(default=ABBA2, init=False)


@dataclass(slots=True)
class _ComposedABBAImplicit(_ABBAImplicitMethod):
	"""Shared placement selector for higher-order ABBA compositions."""

	projection_placement: ProjectionPlacement = "around_complete_composition"

	def __post_init__(self) -> None:
		"""Validate the shared solver and the sole supported projection placement."""
		_ABBAImplicitMethod.__post_init__(self)
		self.projection_placement = _validate_projection_placement(self.projection_placement)


@dataclass(slots=True)
class ABBA4Implicit(_ComposedABBAImplicit):
	"""Fourth-order triple jump with one projection around all six stages."""

	order: ClassVar[Literal[2, 4, 6]] = 4
	recipe: Composition = field(default=ABBA4, init=False)


@dataclass(slots=True)
class ABBA6Implicit(_ComposedABBAImplicit):
	"""Sixth-order Yoshida recipe with one projection around fourteen stages.

	Seven signed, unprojected ABBA pairs keep both copies independent until
	the common outer projection. Negative coefficients reverse the stage clock.
	Optional physical momentum tracking is passive and uses the accepted trace.
	"""

	order: ClassVar[Literal[2, 4, 6]] = 6
	recipe: Composition = field(default=ABBA6, init=False)
@dataclass(slots=True)
class ABBA2Midpoint(MidpointPalindromicMethod):
	"""Second-order ABBA palindrome followed by one arithmetic mean.

	The common base owns execution, optional passive energy and observations.
	Arithmetic projection does not guarantee reversibility or symplecticity.
	"""

	state_extension: StateExtension = "physical"
	progress: bool = False
	step_observer: StepObserver | None = None
	track_energy: bool = False
	recipe: Composition = field(default=ABBA2, init=False)

__all__ = ["ABBA2Implicit", "ABBA4Implicit", "ABBA6Implicit", "ABBA2Midpoint"]
