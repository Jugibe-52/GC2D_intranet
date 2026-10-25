"""Fourth-order HBVM(4,2) with a reduced Legendre-stage solve."""

from __future__ import annotations
from solvers.nonlinear import NewtonSolver, validate_solver
from solvers.derivatives import centered_difference

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING, TypeAlias

import numpy as np
from typing import cast
from contracts.arrays import BackendArray
from solvers.linear import solve_linear, _solve_particle_systems
from types import ModuleType
from contracts.arrays import array_namespace, checked_finite, map_arrays

from contracts.methods.functions import VectorField, JacobianStructure
from contracts.methods.quadrature import StageQuadrature
from methods.classical._jacobians import _checked_vector_field, _checked_analytic_jacobians, _resolved_jacobian_method

from methods.base import FixedStepMethod
from contracts.methods.step import StepInfo, StepResult
from contracts.methods.observation import IntegrationStep
from request import SimulationRequest


if TYPE_CHECKING:
	from solvers.nonlinear import _NonlinearResult
	from typing import Any


HBVMJacobianMethod: TypeAlias = Literal["auto", "analytic", "finite_difference"]

# Four-point Gauss--Legendre quadrature on [0, 1].  HBVM(4,2) uses the first
# two orthonormal shifted Legendre polynomials; the resulting Runge--Kutta
# matrix has rank two even though four quadrature stages sample the line integral.
_GAUSS_NODES: 'np.ndarray'
_GAUSS_WEIGHTS: 'np.ndarray'
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(4)
_HBVM42_NODES: 'np.ndarray' = np.asarray((_GAUSS_NODES + 1.0) / 2.0, dtype=float)
_HBVM42_WEIGHTS: 'np.ndarray' = np.asarray(_GAUSS_WEIGHTS / 2.0, dtype=float)
_HBVM42_LEGENDRE: 'np.ndarray' = np.column_stack(
	(
		np.ones(4, dtype=float),
		np.sqrt(3.0) * (2.0 * _HBVM42_NODES - 1.0),
	)
)
_HBVM42_INTEGRALS: 'np.ndarray' = np.column_stack(
	(
		_HBVM42_NODES,
		np.sqrt(3.0) * (_HBVM42_NODES**2 - _HBVM42_NODES),
	)
)
_HBVM42_RUNGE_KUTTA_MATRIX = (
	_HBVM42_INTEGRALS
	@ _HBVM42_LEGENDRE.T
	@ np.diag(_HBVM42_WEIGHTS)
)
for _coefficient_array in (
	_HBVM42_NODES,
	_HBVM42_WEIGHTS,
	_HBVM42_LEGENDRE,
	_HBVM42_INTEGRALS,
	_HBVM42_RUNGE_KUTTA_MATRIX,
):
	_coefficient_array.setflags(write=False)


@dataclass(frozen=True, slots=True)
class _ResidualPayload:
	"""Four stage states, fields, and projected Legendre coefficients."""

	stages: BackendArray
	fields: BackendArray
	projected_coefficients: BackendArray


@dataclass(frozen=True, slots=True)
class _HBVMStepResult:
	"""Accepted state and nonlinear work for one HBVM step."""

	state: BackendArray
	stage_states: BackendArray
	iterations: int
	residual_norm: float
	tolerance: float
	residual_evaluations: int
	jacobian_evaluations: int
	vector_field_evaluations: int
	converged: Any = True


def _analytic_stage_jacobians(jacobian: VectorField, structure: JacobianStructure,
                              times: BackendArray, stages: BackendArray) -> tuple[BackendArray, ...]:
	"""Retain dense or declared particle derivatives at all quadrature stages."""
	blocks: BackendArray = map_arrays(lambda time, stage:
		_checked_analytic_jacobians(jacobian, time, stage, structure), times, stages)
	return tuple(blocks)



def _finite_difference_stage_jacobians(
	f: VectorField, times: BackendArray, stages: BackendArray, *,
	relative_step: float,
) -> tuple[BackendArray, ...]:
	"""Differentiate each field while retaining HBVM's evaluation accounting."""
	return tuple(map_arrays(lambda time, stage: centered_difference(
		lambda value: _checked_vector_field(f, time, value), stage, relative_step=relative_step), times, stages))


def _reduced_residual_jacobian(
	stage_jacobians: tuple[BackendArray, ...],
	step: float,
) -> BackendArray:
	"""Assemble the two-by-two block Jacobian of the Legendre residual."""
	xp: ModuleType = array_namespace(stage_jacobians)
	state_size = stage_jacobians[0].shape[-1]
	identity: BackendArray = xp.eye(state_size)
	blocks: list[list[BackendArray]] = []
	for legendre_index in range(2):
		row: list[BackendArray] = []
		for integral_index in range(2):
			weighted: Any = sum(
				_HBVM42_WEIGHTS[stage_index]
				* _HBVM42_LEGENDRE[stage_index, legendre_index]
				* _HBVM42_INTEGRALS[stage_index, integral_index]
				* stage_jacobians[stage_index]
				for stage_index in range(4)
			)
			row.append(
				(identity if legendre_index == integral_index else 0.0)
				- step * weighted
			)
		blocks.append(row)
	return cast(BackendArray, xp.concatenate(
		[xp.concatenate(row, axis=-1) for row in blocks], axis=-2))


def _advance_hbvm42(
	f: VectorField, time: float, state: BackendArray, step: float, *,
	solver: NewtonSolver, jacobian: VectorField | None,
	jacobian_structure: JacobianStructure, observe: bool = True,
) -> _HBVMStepResult:
	"""Solve the two-coefficient HBVM(4,2) equation for one complete step."""
	xp: ModuleType = array_namespace(state)
	initial_state: BackendArray = xp.asarray(state, dtype=float)
	state_size = initial_state.size
	stage_times = time + step * _HBVM42_NODES

	# A first-order stage predictor supplies a substantially better Legendre
	# initial guess than (f(y_n), 0) while retaining a small fixed setup cost.
	initial_field: BackendArray = _checked_vector_field(f, time, initial_state)
	predicted_stages = initial_state + step * xp.outer(
		_HBVM42_NODES,
		initial_field,
	)
	predicted_fields: BackendArray = map_arrays(lambda time, stage: _checked_vector_field(f, time, stage),
		stage_times, predicted_stages)
	coefficients = _HBVM42_LEGENDRE.T @ (
		_HBVM42_WEIGHTS[:, None] * predicted_fields
	)
	tolerance = solver.tolerance(initial_state)

	def residual(
		candidate: BackendArray,
	) -> tuple[BackendArray, _ResidualPayload]:
		"""Evaluate the rank-two HBVM stage equation."""
		stages = initial_state + step * (_HBVM42_INTEGRALS @ candidate)
		fields: BackendArray = map_arrays(lambda time, stage: _checked_vector_field(f, time, stage),
			stage_times, stages)
		projected = _HBVM42_LEGENDRE.T @ (
			_HBVM42_WEIGHTS[:, None] * fields
		)
		return candidate - projected, _ResidualPayload(stages, fields, projected)

	def evaluate_flat(candidate: BackendArray) -> tuple[BackendArray, _ResidualPayload]:
		value: BackendArray
		data: _ResidualPayload
		value, data = residual(candidate.reshape(2, state_size))
		return value.reshape(-1), data

	# Preserve the established capability-check boundary after the first residual.
	cached: tuple[BackendArray, _ResidualPayload] = evaluate_flat(coefficients.reshape(-1))
	selected_jacobian_method = _resolved_jacobian_method(jacobian, solver.jacobian_method,
		initial_time=time, initial_state=initial_state, structure=jacobian_structure, probe=False)

	def correction(unknown: BackendArray, current_residual: BackendArray, payload: _ResidualPayload) -> BackendArray:
		"""Keep HBVM's stage Jacobian and field counters beside its equations."""
		if selected_jacobian_method == "analytic":
			assert jacobian is not None
			stage_jacobians: tuple[BackendArray, ...] = _analytic_stage_jacobians(
				jacobian, jacobian_structure,
				stage_times,
				payload.stages,
			)
		else:
			stage_jacobians = _finite_difference_stage_jacobians(
				f,
				stage_times,
				payload.stages,
				relative_step=solver.jacobian_relative_step,
				)
		residual_jacobian: BackendArray = _reduced_residual_jacobian(stage_jacobians, step)
		correction: BackendArray
		if residual_jacobian.ndim == 3:
			# Each independent particle owns both Legendre coefficients. Preserve
			# the global residual norm while solving these smaller linear systems.
			particles: int = residual_jacobian.shape[0]
			components: int = state_size // particles
			right_side: BackendArray = -current_residual.reshape(2, components, particles).transpose(2, 0, 1).reshape(particles, 2 * components)
			blocks: BackendArray = _solve_particle_systems(residual_jacobian, right_side,
				singular_message="The HBVM42 Newton Jacobian is singular.")
			correction = blocks.reshape(particles, 2, components).transpose(1, 2, 0).reshape(-1)
		else:
			correction = solve_linear(residual_jacobian, -current_residual,
				singular_message="The HBVM42 Newton Jacobian is singular.")
		correction = checked_finite(correction, "The HBVM42 Newton correction became non-finite.", RuntimeError)

		return cast(BackendArray, xp.asarray(correction))

	root: _NonlinearResult[_ResidualPayload] = solver.solve(evaluate_flat, coefficients.reshape(-1), correction,
		scale=initial_state, context=f"HBVM42 at t={time}",
		initial_evaluation=cached,
		norm=lambda r: xp.linalg.norm(r.reshape(2, state_size), ord=np.inf),
		update_returns_correction=True, time=time, duration=step, observe=observe)
	payload = root.payload
	accepted = initial_state + step * payload.projected_coefficients[0]
	return _HBVMStepResult(state=xp.asarray(accepted), stage_states=payload.stages,
		iterations=root.iterations, converged=root.converged,
		residual_norm=xp.linalg.norm(root.residual.reshape(2, state_size), ord=np.inf),
		tolerance=tolerance, residual_evaluations=root.residual_evaluations,
		jacobian_evaluations=root.iterations, vector_field_evaluations=(
			5 + 4 * root.residual_evaluations
			+ (8 * state_size * root.iterations if selected_jacobian_method == "finite_difference" else 0)))


@dataclass(slots=True)
class HBVM42(FixedStepMethod[_HBVMStepResult]):
	"""Fourth-order HBVM(4,2) with four quadrature nodes and two coefficients.

	For autonomous polynomial Hamiltonians of degree at most four, energy is
	preserved up to algebraic and round-off error. The method is not generally
	symplectic. Its default solver retains damped Newton corrections.
	"""

	solver: NewtonSolver = field(default_factory=lambda: NewtonSolver(backtracking=True), kw_only=True)
	progress: bool = False

	def __post_init__(self) -> None:
		FixedStepMethod.__post_init__(self)
		validate_solver(self.solver, newton_only=True)

	def initialize(self, request: SimulationRequest) -> None:
		"""Record the method identity without probing deferred derivatives."""
		self.metadata = {'method_order': 4, 'quadrature_stage_count': 4,
			'legendre_rank': 2, 'jacobian_method': self.solver.jacobian_method}

	def _solve_physical(self, time: float, state: BackendArray, step: float,
	                    *, observe: bool = True) -> _HBVMStepResult:
		return _advance_hbvm42(self.f, time, state, step, solver=self.solver,
			jacobian=self.jacobian, jacobian_structure=self.jacobian_structure, observe=observe)

	def advance(self, time: float, state: BackendArray, step: float, *, observe: bool = True) -> StepResult[_HBVMStepResult]:
		"""Return a coefficient solve, work counters and optional stage quadrature."""
		result: _HBVMStepResult = self._solve_physical(time, self._checked_state(state), step, observe=observe)
		quadrature: StageQuadrature | None = None
		if self.capture_quadrature:
			quadrature = StageQuadrature(tuple(time + step * _HBVM42_NODES),
				tuple(result.stage_states), tuple(step * _HBVM42_WEIGHTS))
		return StepResult(result.state, {
			'nonlinear_iterations': result.iterations,
			'nonlinear_residual_norms': result.residual_norm,
			'nonlinear_tolerances': result.tolerance,
			'residual_evaluations': result.residual_evaluations,
			'residual_evaluations_per_step': result.residual_evaluations,
			'jacobian_evaluations_per_step': result.jacobian_evaluations,
			'vector_field_evaluations_per_step': result.vector_field_evaluations,
		}, result, quadrature, result.converged)

	def build_observation(self, info: StepInfo, result: StepResult[_HBVMStepResult]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return np.asarray(self._solve_physical(info.time, candidate, info.duration, observe=False).state)
		return IntegrationStep(
			dynamics_name='', method_name=type(self).__name__,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=info.state_before.copy(),
			state_after=result.state.copy(), map_state=map_state,
		)


__all__ = ['HBVM42', 'HBVMJacobianMethod']
