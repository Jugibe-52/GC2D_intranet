"""Assemble one Hairer equation around a complete operation palindrome."""
from __future__ import annotations
from states.initial import InitialConditions

from functools import partial
import numpy as np
from contracts.arrays import BackendArray
from contracts.methods.observation import IntegrationStep
from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from methods.projection import ProjectionChoice, resolve_projection
from methods.base import FixedStepMethod
from methods.implicit import ImplicitIntegrationMethod
from methods.palindromic.configuration import _state_dimension_diagnostics
from methods.palindromic.core.projection import solve_projection
from contracts.methods.palindromic.core.records import ProjectedMap
from methods.palindromic.core.records import ProjectedMapResult, step_statistics
from solvers.derivatives import ResolvedJacobianMethod, resolve_jacobian_method
from solvers.nonlinear import AlgebraicSolver, BroydenSolver, validate_solver
from methods.palindromic.base import PalindromicMethod


class ProjectedPalindromicMethod(PalindromicMethod[ProjectedMapResult], ImplicitIntegrationMethod[ProjectedMapResult]):
	"""Assemble one Hairer equation around a complete operation palindrome."""

	solver: AlgebraicSolver
	projection_formulation: ProjectionChoice = 'reduced_multiplier'
	project: ProjectedMap
	resolved_jacobian_method: ResolvedJacobianMethod

	def __post_init__(self) -> None:
		FixedStepMethod.__post_init__(self)
		self._validate_solver_controls()
		resolve_projection(self.projection_formulation)

	def initialize(self, initial_conditions: InitialConditions) -> None:
		"""Bind one exterior solve and collect the concrete recipe's metadata."""
		self._prepare_projection(initial_conditions, self.projection_formulation)
		self.metadata = self._projection_metadata(self.projection_formulation)

	def _validate_solver_controls(self) -> None:
		validate_solver(self.solver)

	def _prepare_projection(self, initial_conditions: InitialConditions, projection: ProjectionChoice) -> None:
		"""Resolve capabilities once while leaving controls with the solver."""
		self._prepare_maps(initial_conditions, jacobian=self.jacobian)
		requested = self.solver.jacobian_method
		# Broyden uses the projection's initial matrix, not the field derivative.
		resolved: ResolvedJacobianMethod = (requested if isinstance(self.solver, BroydenSolver) and requested != 'auto'
			else resolve_jacobian_method(self.jacobian, requested))
		self.resolved_jacobian_method = resolved
		self.project = partial(solve_projection, self.formulation, self.recipe, self.solver,
		                       projection, jacobian_method=resolved, composition_name=self.composition_name)

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
			**_state_dimension_diagnostics(resolve_projection(projection).name, particle_count=self.formulation.particle_count),
			**self._operation_metadata(),
		}

	def _publish_substeps(self) -> bool:
		return False

	def advance(self, t: float, state: BackendArray, h: float, *, observe: bool = True) -> StepResult[ProjectedMapResult]:
		"""Return the converged physical map and its accepted-stage quadrature."""
		result: ProjectedMapResult = self.project(t, self._checked_state(state), h, observe=observe)
		return StepResult(result.state, step_statistics(result, include_substeps=self._publish_substeps()),
		                  result, result.trace.quadrature, result.stats.converged)

	def build_observation(self, info: StepInfo, step: StepResult[ProjectedMapResult]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return np.asarray(self.project(info.time, candidate, info.duration).state)
		return IntegrationStep(dynamics_name='', method_name=self.method_name,
			step_index=info.index, start_time=info.time, time=info.end_time, duration=info.duration,
			state_before=info.state_before.copy(), state_after=step.details.state.copy(), map_state=map_state)


__all__ = ['ProjectedPalindromicMethod']
