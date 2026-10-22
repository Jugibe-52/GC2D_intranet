"""Common lifecycle and exterior projections for explicit operation palindromes."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import json
from typing import Generic, Literal, TypeVar

import numpy as np

from contracts.nonlinear import NewtonObserver
from contracts.observation import IntegrationStep, StepObserver
from contracts.problem import InitialValueProblem
from contracts.result import DiagnosticValue
from contracts.step import StepInfo, StepResult
from dynamics.protocols import DynamicalSystem, GuidingCenterJacobianSystem
from formulations.gc import GCDoubledMaps, HarmonicCoupling, _EnergyQuadraturePoint
from formulations.projection import ProjectionChoice, resolve_projection
from formulations.state import DoubledFormulation
from methods._validation import _positive_finite, _positive_integer
from methods.base import IntegrationMethod
from methods.extended.configuration import StateExtension, _state_dimension_diagnostics, _validate_state_extension
from methods.extended.core.composition import Composition
from methods.extended.core.energy import momentum_increment
from methods.extended.core.jacobians import _checked_vector_field_jacobian
from methods.extended.core.midpoint import MidpointResult, midpoint_step
from methods.extended.core.projection import solve_projection
from methods.extended.core.records import ProjectedMap, ProjectedMapResult, step_statistics
from request import SimulationRequest
from solvers.nonlinear import AlgebraicSolver, NonlinearSolver, SolverOptions, _validate_nonlinear_solver, validate_solver

Detail = TypeVar('Detail')


class PalindromicMethod(IntegrationMethod[Detail], Generic[Detail]):
	"""Shared run preparation and passive finishing; subclasses select a recipe.

	The recipe owns every time translation. Projection policies act outside the
	complete palindrome; physical momentum is accumulated only after acceptance.
	"""

	jax_mode = 'device_resident'
	recipe: Composition
	track_energy: bool
	state_formulation: DoubledFormulation
	formulation: GCDoubledMaps
	dynamics: DynamicalSystem

	def _configured_coupling(self) -> HarmonicCoupling | None:
		return None

	def _prepare_maps(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Bind fresh spatial maps and require C and its rate to be explicit together."""
		if not isinstance(self.recipe, Composition):
			raise TypeError('A palindromic method requires a Composition.')
		coupling: HarmonicCoupling | None = self._configured_coupling()
		if coupling is not None and not isinstance(coupling, HarmonicCoupling):
			raise TypeError('Coupling must be a HarmonicCoupling or None.')
		if self.recipe.has_coupling != (coupling is not None):
			raise ValueError('Explicit C operations and a HarmonicCoupling must be configured together.')
		self.dynamics = problem.dynamics
		self.state_formulation = DoubledFormulation(problem, request.t_span[0], self.track_energy)
		self.formulation = GCDoubledMaps(problem, coupling_frequency=None if coupling is None else coupling.frequency)
		self.initial_state = self.state_formulation.initial_state

	def _operation_metadata(self) -> dict[str, DiagnosticValue]:
		coupling: HarmonicCoupling | None = self._configured_coupling()
		return {
			'composition_name': self.recipe.name,
			'operation_sequence': ''.join(op.name for op in self.recipe.operations),
			# Configuration is scalar metadata; arrays are reserved by some block
			# consumers for histories aligned with accepted integration steps.
			'operation_coefficients': json.dumps([op.coefficient for op in self.recipe.operations], allow_nan=False),
			'operation_count': len(self.recipe.operations),
			'coupling_frequency': 0.0 if coupling is None else coupling.frequency,
		}

	def _finish(self, state: np.ndarray, physical: np.ndarray, time: float, step: float,
	            points: tuple[_EnergyQuadraturePoint, ...]) -> np.ndarray:
		increment: np.ndarray | None = momentum_increment(self.state_formulation, points) if self.track_energy else None
		return self.state_formulation.finish(state, physical, time + step, increment)


class ProjectedPalindromicMethod(PalindromicMethod[ProjectedMapResult]):
	"""One Hairer solve around any consistent operation palindrome."""

	solver: AlgebraicSolver | None
	newton_absolute_tolerance: float
	newton_relative_tolerance: float
	newton_max_iterations: int
	nonlinear_solver: NonlinearSolver
	newton_observer: NewtonObserver | None
	project: ProjectedMap

	def _validate_solver_controls(self) -> None:
		"""Normalize legacy controls, giving an explicit solver configuration priority."""
		if self.solver is not None:
			validate_solver(self.solver, newton_only=False)
			self.newton_absolute_tolerance = self.solver.absolute_tolerance
			self.newton_relative_tolerance = self.solver.relative_tolerance
			self.newton_max_iterations = self.solver.max_iterations
			self.nonlinear_solver = self.solver.name
		self.newton_absolute_tolerance = _positive_finite(self.newton_absolute_tolerance, 'newton_absolute_tolerance')
		self.newton_relative_tolerance = _positive_finite(self.newton_relative_tolerance, 'newton_relative_tolerance')
		self.newton_max_iterations = _positive_integer(self.newton_max_iterations, 'newton_max_iterations')
		self.nonlinear_solver = _validate_nonlinear_solver(self.nonlinear_solver)
		if self.newton_observer is not None and self.nonlinear_solver != 'newton':
			raise ValueError('A newton_observer requires nonlinear_solver="newton".')

	def _prepare_projection(self, problem: InitialValueProblem, request: SimulationRequest,
	                        projection: ProjectionChoice, *,
	                        jacobian: Literal['analytic', 'finite_difference'] = 'analytic',
	                        relative_step: float = float(np.cbrt(np.finfo(float).eps))) -> SolverOptions:
		"""Bind one exterior spatial projection without advancing the initial state."""
		self._prepare_maps(problem, request)
		options: SolverOptions = SolverOptions(self.nonlinear_solver, self.newton_absolute_tolerance,
		                        self.newton_relative_tolerance, self.newton_max_iterations)
		self.project = partial(solve_projection, self.formulation, self.recipe, options, projection,
		                       solver=self.solver, jacobian_method=jacobian,
		                       jacobian_relative_step=relative_step, retain_energy_points=self.track_energy)
		return options

	def _publish_substeps(self) -> bool:
		return False

	def advance(self, t: float, state: np.ndarray, h: float) -> StepResult[ProjectedMapResult]:
		"""Project the complete palindrome and finish passive state after convergence."""
		physical: np.ndarray = self.state_formulation.physical(state)
		result: ProjectedMapResult = self.project(t, physical, h, newton_observer=self.newton_observer)
		after: np.ndarray = self._finish(state, result.state, t, h, result.energy_points)
		return StepResult(after, step_statistics(result, include_substeps=self._publish_substeps()), result)

	def build_observation(self, info: StepInfo, step: StepResult[ProjectedMapResult]) -> IntegrationStep:
		"""Generic physical observation; named presets may supply richer adapters."""
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return self.project(info.time, candidate, info.duration).state
		return IntegrationStep(dynamics_name=type(self.dynamics).__name__, method_name=self.method_name,
			step_index=info.index, start_time=info.time, time=info.end_time, duration=info.duration,
			state_before=self.state_formulation.physical(info.state_before).copy(),
			state_after=step.details.state.copy(), map_state=map_state, dynamics=self.dynamics)


class MidpointPalindromicMethod(PalindromicMethod[MidpointResult]):
	"""Apply a complete palindrome, then average its spatial copies once."""

	state_extension: StateExtension

	def __post_init__(self) -> None:
		self.state_extension = _validate_state_extension(self.state_extension)
		self.track_energy = bool(self.track_energy)

	def _preset_metadata(self) -> dict[str, DiagnosticValue]:
		return {}

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Prepare fresh doubled maps and describe the arithmetic projection."""
		self._prepare_maps(problem, request)
		self.metadata = {
			'projection_kind': 'arithmetic_mean', 'state_extension': self.state_extension,
			'track_energy': self.track_energy, 'nonlinear_unknown_dimension': 0,
			'vector_field_evaluations_per_step': self.recipe.shear_count,
			**_state_dimension_diagnostics(particle_count=problem.particle_count),
			**self._operation_metadata(), **self._preset_metadata(),
		}

	def advance(self, t: float, state: np.ndarray, h: float) -> StepResult[MidpointResult]:
		"""Average the copies once, after all explicit operations have executed."""
		result: MidpointResult = midpoint_step(self.formulation, self.recipe, t, self.state_formulation.physical(state), h)
		after: np.ndarray = self._finish(state, result.state, t, h, result.trace.energy_points)
		return StepResult(after, {'copy_separation_norms': result.copy_separation_norm}, result)

	def build_observation(self, info: StepInfo, step: StepResult[MidpointResult]) -> IntegrationStep:
		"""Expose the accepted physical map to the shared observation interface."""
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return midpoint_step(self.formulation, self.recipe, info.time, candidate, info.duration).state
		return IntegrationStep(dynamics_name=type(self.dynamics).__name__, method_name=self.method_name,
			step_index=info.index, start_time=info.time, time=info.end_time, duration=info.duration,
			state_before=self.state_formulation.physical(info.state_before).copy(),
			state_after=step.details.state.copy(), map_state=map_state, dynamics=self.dynamics)


@dataclass(slots=True)
class PalindromicMidpoint(MidpointPalindromicMethod):
	"""Experimental recipe with final arithmetic projection; no order is inferred."""

	recipe: Composition
	state_extension: StateExtension = 'physical'
	progress: bool = False
	step_observer: StepObserver | None = None
	track_energy: bool = False
	coupling: HarmonicCoupling | None = field(default=None, kw_only=True)

	def _configured_coupling(self) -> HarmonicCoupling | None:
		return self.coupling


@dataclass(slots=True)
class PalindromicImplicit(ProjectedPalindromicMethod):
	"""Experimental recipe with one exterior spatial Hairer projection."""

	recipe: Composition
	projection_formulation: ProjectionChoice = 'reduced_multiplier'
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	newton_jacobian_method: Literal['analytic', 'finite_difference'] = 'analytic'
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	nonlinear_solver: NonlinearSolver = 'newton'
	track_energy: bool = False
	progress: bool = False
	step_observer: StepObserver | None = None
	solver: AlgebraicSolver | None = field(default=None, kw_only=True)
	newton_observer: NewtonObserver | None = field(default=None, kw_only=True)
	coupling: HarmonicCoupling | None = field(default=None, kw_only=True)

	def __post_init__(self) -> None:
		self._validate_solver_controls()
		resolve_projection(self.projection_formulation)
		if self.newton_jacobian_method not in ('analytic', 'finite_difference'):
			raise ValueError('Jacobian method must be analytic or finite_difference.')
		self.newton_jacobian_relative_step = _positive_finite(self.newton_jacobian_relative_step, 'newton_jacobian_relative_step')

	def _configured_coupling(self) -> HarmonicCoupling | None:
		return self.coupling

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Bind the selected equation and verify the requested derivative contract."""
		options: SolverOptions = self._prepare_projection(problem, request, self.projection_formulation,
			jacobian=self.newton_jacobian_method, relative_step=self.newton_jacobian_relative_step)
		if self.nonlinear_solver == 'newton' and self.newton_jacobian_method == 'analytic':
			if not isinstance(self.dynamics, GuidingCenterJacobianSystem):
				raise TypeError('Analytic palindromic projection requires GC particle Jacobians.')
			_checked_vector_field_jacobian(self.dynamics, request.t_span[0], problem.initial_state)
		self.metadata = {
			'nonlinear_solves_per_step': 1, 'nonlinear_solver': options.solver,
			'nonlinear_absolute_tolerance': options.absolute_tolerance,
			'nonlinear_relative_tolerance': options.relative_tolerance,
			'nonlinear_max_iterations': options.max_iterations,
			'newton_jacobian_method': self.newton_jacobian_method,
			'newton_jacobian_relative_step': self.newton_jacobian_relative_step,
			'projection_formulation': resolve_projection(self.projection_formulation).name,
			'projection_placement': 'around_complete_composition',
			**_state_dimension_diagnostics(resolve_projection(self.projection_formulation).name,
			                              particle_count=problem.particle_count),
			**self._operation_metadata(),
		}


__all__ = ['PalindromicMethod', 'PalindromicImplicit', 'PalindromicMidpoint']
