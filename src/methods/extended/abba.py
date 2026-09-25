"""Configuration and numerical operations shared by implicit ABBA methods."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from formulations.gc import GCDoubledMaps
from methods.extended.composition import ABBA2, ABBA4
from methods.extended.projection import solve_projection
from methods.extended.records import ProjectedMap
from methods.extended.energy import momentum_increment
from typing import ClassVar, Literal

import numpy as np

from dynamics import GuidingCenterJacobianSystem

from contracts.result import DiagnosticValue
from integration.core import IntegrationMethod, NEWTON_ALIASES
from contracts.step import StepInfo, StepResult as NumericalStep
from contracts.observation import IntegrationStep, StepObserver
from formulations.state import DoubledFormulation
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from methods._nonlinear import NonlinearSolver, SolverOptions, _validate_nonlinear_solver
from methods.extended.coefficients import _ABBA4_COEFFICIENTS, _ABBA6_COEFFICIENTS
from methods.extended.configuration import (
    ProjectionFormulation, StateExtension, _resolved_track_energy,
    _state_dimension_diagnostics, _validate_projection_formulation, _validate_state_extension,
)
from methods.extended.abba_maps import _checked_vector_field_jacobian
from methods.extended.abba_observations import EventBuilder, bind_event_builder
from methods.extended.records import ProjectedMapResult, StepResult, step_statistics


from methods.extended.configuration import _positive_finite, _positive_integer


@dataclass(slots=True)
class _ABBAImplicitMethod(IntegrationMethod[StepResult]):
	"""Own one ABBA run; concrete orders select the composition and projection placement."""

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
	state_formulation: DoubledFormulation = field(init=False, repr=False, compare=False)
	solver_options: SolverOptions = field(init=False, repr=False, compare=False)
	build_event: EventBuilder | None = field(init=False, repr=False, compare=False)
	coefficients: tuple[float, ...] = field(init=False, repr=False, compare=False)
	include_substep_metrics: bool = field(init=False, repr=False, compare=False)
	project: ProjectedMap = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Validate the nonlinear projection solver configuration."""
		self.projection_formulation = _validate_projection_formulation(self.projection_formulation)
		self.state_extension = _validate_state_extension(self.state_extension)
		self.track_energy = _resolved_track_energy(self.track_energy, self.state_extension)
		self.newton_absolute_tolerance = _positive_finite(self.newton_absolute_tolerance, 'newton_absolute_tolerance')
		self.newton_relative_tolerance = _positive_finite(self.newton_relative_tolerance, 'newton_relative_tolerance')
		self.newton_max_iterations = _positive_integer(self.newton_max_iterations, 'newton_max_iterations')
		self.nonlinear_solver = _validate_nonlinear_solver(self.nonlinear_solver)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Select this run's state representation, projection and numerical recipe."""
		projection_placement = getattr(self, "projection_placement", None)
		outer = self.order == 4
		if outer:
			if projection_placement not in (None, "around_complete_composition"):
				raise ValueError("ABBA4 requires one projection around the complete composition.")
			projection_placement = "around_complete_composition"
		else:
			if projection_placement not in (None, "after_each_abba_map"):
				raise ValueError("ABBA2 and ABBA6 project each of their base maps.")
			projection_placement = "after_each_abba_map"
		self.coefficients = (
			(1.0,) if self.order == 2 else tuple(float(c) for c in (
				_ABBA4_COEFFICIENTS if self.order == 4 else _ABBA6_COEFFICIENTS
			))
		)
		dynamics = problem.dynamics
		name = type(self).__name__
		options = SolverOptions(self.nonlinear_solver, self.newton_absolute_tolerance,
		                        self.newton_relative_tolerance, self.newton_max_iterations)
		z0 = problem.initial_state
		if not isinstance(dynamics, GuidingCenterJacobianSystem) or dynamics.state_dimension != 2:
			raise TypeError(f"{name} requires planar GuidingCenterJacobianSystem dynamics.")
		self.state_formulation = DoubledFormulation(problem, request.t_span[0], self.track_energy)
		self.solver_options = options
		self.project = partial(solve_projection, GCDoubledMaps(problem, coupling_frequency=None),
		    ABBA4 if outer else ABBA2, options, self.projection_formulation,
		    retain_energy_points=self.track_energy)
		if options.solver == "newton":
			_checked_vector_field_jacobian(dynamics, request.t_span[0], z0)
		self.build_event = None
		if self.step_observer is not None:
			self.build_event = bind_event_builder(
				dynamics, name, self.projection_formulation, order=self.order,
				outer=outer, coefficients=self.coefficients,
				solve_step=self.solve_step, project=self.project,
			)
		projection_count = 1 if outer else len(self.coefficients)
		metadata: dict[str, DiagnosticValue] = {
			"nonlinear_solves_per_step": projection_count,
			"nonlinear_solver": options.solver,
			"nonlinear_absolute_tolerance": options.absolute_tolerance,
			"nonlinear_relative_tolerance": options.relative_tolerance,
			"nonlinear_max_iterations": options.max_iterations,
			"projection_formulation": self.projection_formulation,
			"state_extension": self.state_extension,
			"track_energy": self.track_energy,
		}
		metadata.update(_state_dimension_diagnostics(
			self.state_extension, self.projection_formulation, particle_count=z0.size // 2,
		))
		self.include_substep_metrics = self.order != 2
		if self.include_substep_metrics:
			coefficient_array = np.asarray(self.coefficients)
			coefficient_array.setflags(write=False)
			metadata.update({
				"implicit_substeps_per_step": projection_count,
				"projection_placement": projection_placement,
				"composition_coefficients": coefficient_array,
			})
		if outer:
			metadata.update({
				"unprojected_abba_maps_per_step": len(self.coefficients),
				"unprojected_abba_maps_per_residual_evaluation": len(self.coefficients),
				"base_composition": "unprojected_abba4_triple_jump",
			})
		elif self.order != 2:
			metadata.update({
				"substep_projection_formulation": self.projection_formulation,
				"composition_policy": "project_each_abba_substep",
			})
		initial = self.state_formulation.initial_state
		initial.setflags(write=False)
		self.initial_state = initial
		self.metadata = metadata
		self.diagnostic_aliases = NEWTON_ALIASES

	def solve_step(self, t: float, state: np.ndarray, h: float) -> tuple[ProjectedMapResult, ...]:
		"""Apply the order's complete recipe using this run's projection equation."""
		if self.order != 6:
			return (self.project(t, state, h),)
		# ABBA6 composes complete projected pairs; flattening them would change it.
		projections = []
		for coefficient in self.coefficients:
			duration = coefficient * h
			result = self.project(t, state, duration)
			projections.append(result)
			state = result.state
			t += duration
		return tuple(projections)

	def advance(self, t: float, workspace: np.ndarray, h: float) -> NumericalStep[StepResult]:
		"""Apply the projections and finish the physical or extended state update."""
		state_before = self.state_formulation.physical(workspace)
		projections = (self.project(t, state_before, h),) if self.order != 6 else self.solve_step(t, state_before, h)
		increment = None
		if self.track_energy:
			increment = momentum_increment(self.state_formulation,
			    (point for projection in projections for point in projection.energy_points))
		next_workspace = self.state_formulation.finish(workspace, projections[-1].state, t + h, increment)
		result = StepResult(next_workspace, projections)
		return NumericalStep(next_workspace, step_statistics(
			result, include_substeps=self.include_substep_metrics,
		), result)

	def build_observation(self, info: StepInfo, step: NumericalStep[StepResult]) -> IntegrationStep:
		"""Expose the selected physical or extended observation domain."""
		assert self.build_event is not None
		state_before = self.state_formulation.physical(info.state_before)
		return self.build_event(info.time, info.duration, info.index, state_before, step.details)


__all__: list[str] = []
