"""Common lifecycle and exterior projections for explicit operation palindromes."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import json
from typing import Generic, TypeVar

import numpy as np

from contracts.methods.observation import IntegrationStep
from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from methods.extended.maps import GCDoubledMaps, HarmonicCoupling
from methods.projection import ProjectionChoice, resolve_projection
from methods.base import FixedStepMethod
from methods.extended.configuration import StateExtension, _state_dimension_diagnostics, _validate_state_extension
from methods.extended.core.composition import Composition
from methods.extended.core.jacobians import _checked_vector_field_jacobian
from methods.extended.core.midpoint import MidpointResult, midpoint_step
from methods.extended.core.projection import solve_projection
from contracts.methods.extended.core.records import ProjectedMap
from methods.extended.core.records import ProjectedMapResult, step_statistics
from request import SimulationRequest
from solvers.derivatives import ResolvedJacobianMethod
from solvers.nonlinear import AlgebraicSolver, NewtonSolver, validate_solver

Detail = TypeVar('Detail')


class PalindromicMethod(FixedStepMethod[Detail], Generic[Detail]):
	"""Bind physical functions to a recipe and temporary doubled workspaces."""

	jax_mode = 'device_resident'
	copies = 2
	requires_planar = True
	recipe: Composition
	formulation: GCDoubledMaps

	def _configured_coupling(self) -> HarmonicCoupling | None:
		return None

	def _prepare_maps(self) -> None:
		"""Require explicit C operations and their independent rate together."""
		if not isinstance(self.recipe, Composition):
			raise TypeError('A palindromic method requires a Composition.')
		coupling: HarmonicCoupling | None = self._configured_coupling()
		if coupling is not None and not isinstance(coupling, HarmonicCoupling):
			raise TypeError('Coupling must be a HarmonicCoupling or None.')
		if self.recipe.has_coupling != (coupling is not None):
			raise ValueError('Explicit C operations and a HarmonicCoupling must be configured together.')
		self.formulation = GCDoubledMaps(self.f, self.physical_size, coordinates=self.layout,
			jacobian=self.jacobian, jacobian_structure=self.jacobian_structure,
			coupling_frequency=None if coupling is None else coupling.frequency)

	def _operation_metadata(self) -> dict[str, DiagnosticValue]:
		coupling: HarmonicCoupling | None = self._configured_coupling()
		return {
			'composition_name': self.recipe.name,
			'operation_sequence': ''.join(op.name for op in self.recipe.operations),
			'operation_coefficients': json.dumps([op.coefficient for op in self.recipe.operations], allow_nan=False),
			'operation_count': len(self.recipe.operations),
			'coupling_frequency': 0.0 if coupling is None else coupling.frequency,
		}


class ProjectedPalindromicMethod(PalindromicMethod[ProjectedMapResult]):
	"""Assemble one Hairer equation around a complete operation palindrome."""

	solver: AlgebraicSolver
	project: ProjectedMap
	resolved_jacobian_method: ResolvedJacobianMethod

	def _validate_solver_controls(self) -> None:
		validate_solver(self.solver)

	def _prepare_projection(self, request: SimulationRequest, projection: ProjectionChoice,
	                        *, probe_analytic: bool = False) -> None:
		"""Resolve capabilities once while leaving controls with the solver."""
		self._prepare_maps()
		requested = self.solver.jacobian_method
		resolved: ResolvedJacobianMethod = ('analytic' if self.jacobian is not None else 'finite_difference') if requested == 'auto' else requested
		if isinstance(self.solver, NewtonSolver) and resolved == 'analytic' and (probe_analytic or requested == 'auto'):
			try:
				_checked_vector_field_jacobian(self.formulation, request.t_span[0], self.initial_state)
			except (TypeError, ValueError, NotImplementedError):
				if requested != 'auto':
					raise
				resolved = 'finite_difference'
		self.resolved_jacobian_method = resolved
		self.project = partial(solve_projection, self.formulation, self.recipe, self.solver,
		                       projection, jacobian_method=resolved)

	def _projection_metadata(self, projection: ProjectionChoice) -> dict[str, DiagnosticValue]:
		return {
			'nonlinear_solves_per_step': 1, 'nonlinear_solver': self.solver.name,
			'nonlinear_absolute_tolerance': self.solver.absolute_tolerance,
			'nonlinear_relative_tolerance': self.solver.relative_tolerance,
			'nonlinear_max_iterations': self.solver.max_iterations,
			'newton_jacobian_method': self.resolved_jacobian_method,
			'newton_jacobian_relative_step': self.solver.jacobian_relative_step,
			'projection_formulation': resolve_projection(projection).name,
			'projection_placement': 'around_complete_composition',
			**_state_dimension_diagnostics(resolve_projection(projection).name, particle_count=self.particle_count),
			**self._operation_metadata(),
		}

	def _publish_substeps(self) -> bool:
		return False

	def advance(self, t: float, state: np.ndarray, h: float, *, observe: bool = True) -> StepResult[ProjectedMapResult]:
		"""Return the converged physical map and optional accepted-stage quadrature."""
		result: ProjectedMapResult = self.project(t, self._checked_state(state), h, observe=observe)
		return StepResult(result.state, step_statistics(result, include_substeps=self._publish_substeps()),
		                  result, result.trace.quadrature if self.capture_quadrature else None)

	def build_observation(self, info: StepInfo, step: StepResult[ProjectedMapResult]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return self.project(info.time, candidate, info.duration).state
		return IntegrationStep(dynamics_name='', method_name=self.method_name,
			step_index=info.index, start_time=info.time, time=info.end_time, duration=info.duration,
			state_before=info.state_before.copy(), state_after=step.details.state.copy(), map_state=map_state)


class MidpointPalindromicMethod(PalindromicMethod[MidpointResult]):
	"""Apply a complete palindrome, then average its spatial copies once."""

	state_extension: StateExtension

	def __post_init__(self) -> None:
		FixedStepMethod.__post_init__(self)
		self.state_extension = _validate_state_extension(self.state_extension)

	def _preset_metadata(self) -> dict[str, DiagnosticValue]:
		return {}

	def initialize(self, request: SimulationRequest) -> None:
		self._prepare_maps()
		self.metadata = {
			'projection_kind': 'arithmetic_mean', 'state_extension': self.state_extension,
			'nonlinear_unknown_dimension': 0, 'vector_field_evaluations_per_step': self.recipe.shear_count,
			**self._operation_metadata(), **self._preset_metadata(),
		}

	def advance(self, t: float, state: np.ndarray, h: float, *, observe: bool = True) -> StepResult[MidpointResult]:
		result: MidpointResult = midpoint_step(self.formulation, self.recipe, t, self._checked_state(state), h)
		return StepResult(result.state, {'copy_separation_norms': result.copy_separation_norm}, result,
		                  result.trace.quadrature if self.capture_quadrature else None)

	def build_observation(self, info: StepInfo, step: StepResult[MidpointResult]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return midpoint_step(self.formulation, self.recipe, info.time, candidate, info.duration).state
		return IntegrationStep(dynamics_name='', method_name=self.method_name,
			step_index=info.index, start_time=info.time, time=info.end_time, duration=info.duration,
			state_before=info.state_before.copy(), state_after=step.details.state.copy(), map_state=map_state)


@dataclass(slots=True)
class PalindromicMidpoint(MidpointPalindromicMethod):
	"""Experimental recipe with final arithmetic projection; no order is inferred."""

	recipe: Composition
	state_extension: StateExtension = 'physical'
	progress: bool = False
	coupling: HarmonicCoupling | None = field(default=None, kw_only=True)

	def _configured_coupling(self) -> HarmonicCoupling | None:
		return self.coupling


@dataclass(slots=True)
class PalindromicImplicit(ProjectedPalindromicMethod):
	"""Experimental recipe with one exterior spatial Hairer projection."""

	recipe: Composition
	projection_formulation: ProjectionChoice = 'reduced_multiplier'
	progress: bool = False
	solver: AlgebraicSolver = field(default_factory=lambda: NewtonSolver(jacobian_method='analytic'), kw_only=True)
	coupling: HarmonicCoupling | None = field(default=None, kw_only=True)

	def __post_init__(self) -> None:
		FixedStepMethod.__post_init__(self)
		self._validate_solver_controls()
		resolve_projection(self.projection_formulation)

	def _configured_coupling(self) -> HarmonicCoupling | None:
		return self.coupling

	def initialize(self, request: SimulationRequest) -> None:
		self._prepare_projection(request, self.projection_formulation, probe_analytic=True)
		self.metadata = self._projection_metadata(self.projection_formulation)


__all__ = ['PalindromicMethod', 'PalindromicImplicit', 'PalindromicMidpoint']
