"""Public ABBA configurations sharing one composition and outer projection."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import ClassVar, Literal

import numpy as np

from contracts.observation import IntegrationStep, StepObserver
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from contracts.result import DiagnosticValue
from contracts.step import StepInfo, StepResult as NumericalStep
from dynamics import DynamicalSystem, GuidingCenterJacobianSystem
from formulations.gc import GCDoubledMaps
from formulations.state import DoubledFormulation
from integration.core import IntegrationMethod
from methods._nonlinear import NonlinearSolver, SolverOptions, _validate_nonlinear_solver
from methods.extended.configuration import (
    ProjectionFormulation, ProjectionPlacement, StateExtension,
    _positive_finite, _positive_integer,
    _state_dimension_diagnostics, _validate_projection_formulation,
    _validate_projection_placement, _validate_state_extension,
)
from methods.extended.core.composition import ABBA2, ABBA4, ABBA6, Composition
from methods.extended.core.energy import momentum_increment
from methods.extended.core.jacobians import _checked_vector_field_jacobian
from methods.extended.core.midpoint import midpoint_step, MidpointResult
from methods.extended.core.projection import solve_projection
from methods.extended.core.records import ProjectedMap, StepResult, step_statistics
from methods.extended.observations import EventBuilder, bind_event_builder


@dataclass(slots=True)
class _ABBAImplicitMethod(IntegrationMethod[StepResult]):
    """Project one complete palindromic recipe; concrete methods select its order."""

    projection_formulation: ProjectionFormulation = "reduced_multiplier"
    state_extension: StateExtension = "physical"
    newton_absolute_tolerance: float = 1e-13
    newton_relative_tolerance: float = 1e-12
    newton_max_iterations: int = 12
    nonlinear_solver: NonlinearSolver = "newton"
    progress: bool = False
    step_observer: StepObserver | None = None
    track_energy: bool = False

    order: ClassVar[Literal[2, 4, 6]]
    recipe: ClassVar[Composition]
    state_formulation: DoubledFormulation = field(init=False, repr=False, compare=False)
    build_event: EventBuilder | None = field(init=False, repr=False, compare=False)
    project: ProjectedMap = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the common spatial projection and nonlinear solver controls."""
        self.projection_formulation = _validate_projection_formulation(self.projection_formulation)
        self.state_extension = _validate_state_extension(self.state_extension)
        self.track_energy = bool(self.track_energy)
        self.newton_absolute_tolerance = _positive_finite(self.newton_absolute_tolerance, 'newton_absolute_tolerance')
        self.newton_relative_tolerance = _positive_finite(self.newton_relative_tolerance, 'newton_relative_tolerance')
        self.newton_max_iterations = _positive_integer(self.newton_max_iterations, 'newton_max_iterations')
        self.nonlinear_solver = _validate_nonlinear_solver(self.nonlinear_solver)

    def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
        """Bind the complete recipe to one reduced or simultaneous projection."""
        dynamics = problem.dynamics
        name = type(self).__name__
        if not isinstance(dynamics, GuidingCenterJacobianSystem) or dynamics.state_dimension != 2:
            raise TypeError(f"{name} requires planar GuidingCenterJacobianSystem dynamics.")
        options = SolverOptions(self.nonlinear_solver, self.newton_absolute_tolerance,
                                self.newton_relative_tolerance, self.newton_max_iterations)
        self.state_formulation = DoubledFormulation(problem, request.t_span[0], self.track_energy)
        self.project = partial(
            solve_projection, GCDoubledMaps(problem, coupling_frequency=None),
            self.recipe, options, self.projection_formulation,
            retain_energy_points=self.track_energy,
        )
        if options.solver == "newton":
            _checked_vector_field_jacobian(dynamics, request.t_span[0], problem.initial_state)
        # Each ABBA pair contains equal adjoint/direct weights. Pair durations
        # describe unprojected stages, never independent nonlinear solves.
        coefficients = tuple(2 * c for c in self.recipe.coefficients[::2])
        self.build_event = None
        if self.step_observer is not None:
            self.build_event = bind_event_builder(
                dynamics, name, self.projection_formulation, order=self.order,
                coefficients=coefficients, project=self.project,
            )
        metadata: dict[str, DiagnosticValue] = {
            "nonlinear_solves_per_step": 1,
            "nonlinear_solver": options.solver,
            "nonlinear_absolute_tolerance": options.absolute_tolerance,
            "nonlinear_relative_tolerance": options.relative_tolerance,
            "nonlinear_max_iterations": options.max_iterations,
            "projection_formulation": self.projection_formulation,
            "projection_placement": "around_complete_composition",
            "state_extension": self.state_extension,
            "track_energy": self.track_energy,
            "composition_stage_count": len(self.recipe.coefficients),
        }
        metadata.update(_state_dimension_diagnostics(
            self.projection_formulation,
            particle_count=problem.particle_count,
        ))
        if self.order != 2:
            coefficient_array = np.asarray(coefficients)
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
        self.initial_state = self.state_formulation.initial_state
        self.metadata = metadata

    def advance(self, t: float, workspace: np.ndarray, h: float) -> NumericalStep[StepResult]:
        """Project the complete composition and accumulate its passive energy."""
        state_before = self.state_formulation.physical(workspace)
        projection = self.project(t, state_before, h)
        increment = momentum_increment(self.state_formulation, projection.energy_points) if self.track_energy else None
        after = self.state_formulation.finish(workspace, projection.state, t + h, increment)
        result = StepResult(after, (projection,))
        return NumericalStep(after, step_statistics(result, include_substeps=self.order != 2), result)

    def build_observation(self, info: StepInfo, step: NumericalStep[StepResult]) -> IntegrationStep:
        """Expose physical snapshots from the accepted projection trace."""
        assert self.build_event is not None
        before = self.state_formulation.physical(info.state_before)
        return self.build_event(info.time, info.duration, info.index, before, step.details)


@dataclass(slots=True)
class ABBA2Implicit(_ABBAImplicitMethod):
    """Second-order ABBA pair with one symmetric spatial projection."""

    order: ClassVar[Literal[2, 4, 6]] = 2
    recipe: ClassVar[Composition] = ABBA2


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
    recipe: ClassVar[Composition] = ABBA4


@dataclass(slots=True)
class ABBA6Implicit(_ComposedABBAImplicit):
    """Sixth-order Yoshida recipe with one projection around fourteen stages.

    Seven signed, unprojected ABBA pairs keep both copies independent until
    the common outer projection. Negative coefficients reverse the stage clock.
    Optional physical momentum tracking is passive and uses the accepted trace.
    """

    order: ClassVar[Literal[2, 4, 6]] = 6
    recipe: ClassVar[Composition] = ABBA6
@dataclass(slots=True)
class ABBA2Midpoint(IntegrationMethod[MidpointResult]):
	"""Second-order midpoint ABBA with optional physical energy tracking.

	The method duplicates only the physical state, applies
	the endpoint-time A-B-B-A shears, and averages the two final copies. Tracking
	the physical conjugate momentum is an auxiliary triangular update that does
	not feed back into this map. Midpoint has no residual-formulation or
	nonlinear-solver axis.
	"""

	state_extension: StateExtension = "physical"
	progress: bool = False
	step_observer: StepObserver | None = None
	track_energy: bool = False

	# Runtime resources are initialized once by new_run, never constructor inputs.
	state_formulation: DoubledFormulation = field(init=False, repr=False, compare=False)
	dynamics: DynamicalSystem = field(init=False, repr=False, compare=False)
	formulation: GCDoubledMaps = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Validate the state strategy and resolve inherent energy tracking."""
		self.state_extension = _validate_state_extension(self.state_extension)
		self.track_energy = bool(self.track_energy)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Initialize one spatial arithmetic-projection run with optional passive energy."""
		self.dynamics = problem.dynamics
		self.formulation = GCDoubledMaps(problem, coupling_frequency=None)
		self.state_formulation = DoubledFormulation(problem, request.t_span[0], self.track_energy)
		self.initial_state = self.state_formulation.initial_state
		metadata: dict[str, DiagnosticValue] = {
			"projection_kind": "arithmetic_mean", "state_extension": self.state_extension,
			"track_energy": self.track_energy, "nonlinear_unknown_dimension": 0,
			"vector_field_evaluations_per_step": 4,
		}
		metadata.update(_state_dimension_diagnostics(particle_count=problem.particle_count))
		self.metadata = metadata

	def advance(self, t: float, state: np.ndarray, h: float) -> NumericalStep[MidpointResult]:
		"""Project the spatial copies and independently accumulate their energy balance."""
		result = midpoint_step(self.formulation, ABBA2, t, self.state_formulation.physical(state), h)
		increment = None
		if self.track_energy:
			increment = momentum_increment(self.state_formulation, result.trace.energy_points)
		after = self.state_formulation.finish(state, result.state, t + h, increment)
		return NumericalStep(after, {"copy_separation_norms": result.copy_separation_norm}, result)

	def build_observation(self, info: StepInfo, step: NumericalStep[MidpointResult]) -> IntegrationStep:
		"""Observe only the physical map, independently of energy tracking."""
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return midpoint_step(self.formulation, ABBA2, info.time, candidate, info.duration).state
		return IntegrationStep(
			dynamics_name=type(self.dynamics).__name__, method_name=self.method_name,
			step_index=info.index, start_time=info.time, time=info.end_time,
			duration=info.duration, state_before=self.state_formulation.physical(info.state_before).copy(),
			state_after=step.details.state.copy(), map_state=map_state, dynamics=self.dynamics)

__all__ = ["ABBA2Implicit", "ABBA4Implicit", "ABBA6Implicit", "ABBA2Midpoint"]
