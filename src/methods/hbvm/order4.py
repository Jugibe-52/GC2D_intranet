"""Fourth-order HBVM(4,2) with a reduced Legendre-stage solve."""

from __future__ import annotations
from solvers.nonlinear import NewtonSolver, validate_solver
from solvers.derivatives import centered_difference, packed_particle_jacobian

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING, TypeAlias

import numpy as np

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

	stages: np.ndarray
	fields: np.ndarray
	projected_coefficients: np.ndarray


@dataclass(frozen=True, slots=True)
class _HBVMStepResult:
	"""Accepted state and nonlinear work for one HBVM step."""

	state: np.ndarray
	stage_states: np.ndarray
	iterations: int
	residual_norm: float
	tolerance: float
	residual_evaluations: int
	jacobian_evaluations: int
	vector_field_evaluations: int


@dataclass(slots=True)
class _EvaluationCounter:
	"""Mutable vector-field counter local to one nonlinear solve."""

	vector_field_evaluations: int = 0


def _evaluate_vector_field(f: VectorField, time: float, state: np.ndarray,
                           counter: _EvaluationCounter) -> np.ndarray:
	"""Count one validated field evaluation, including derivative probes."""
	value: np.ndarray = _checked_vector_field(f, time, state)
	counter.vector_field_evaluations += 1
	return value


def _analytic_stage_jacobians(jacobian: VectorField, structure: JacobianStructure,
                              times: np.ndarray, stages: np.ndarray) -> tuple[np.ndarray, ...]:
	"""Return dense field derivatives for the four quadrature stages."""
	blocks: tuple[np.ndarray, ...] = tuple(_checked_analytic_jacobians(jacobian,
		float(time), stage, structure) for time, stage in zip(times, stages, strict=True))
	return tuple(packed_particle_jacobian(block) for block in blocks) if structure == 'particle' else blocks


def _finite_difference_stage_jacobians(
	f: VectorField, times: np.ndarray, stages: np.ndarray, *,
	relative_step: float, counter: _EvaluationCounter,
) -> tuple[np.ndarray, ...]:
	"""Differentiate each field while retaining HBVM's evaluation accounting."""
	return tuple(centered_difference(lambda value: _evaluate_vector_field(f, float(time), value, counter),
		stage, relative_step=relative_step) for time, stage in zip(times, stages, strict=True))


def _reduced_residual_jacobian(
	stage_jacobians: tuple[np.ndarray, ...],
	step: float,
) -> np.ndarray:
	"""Assemble the two-by-two block Jacobian of the Legendre residual."""
	state_size = stage_jacobians[0].shape[0]
	identity: np.ndarray = np.eye(state_size)
	blocks: list[list[np.ndarray]] = []
	for legendre_index in range(2):
		row: list[np.ndarray] = []
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
	return np.block(blocks)


def _advance_hbvm42(
	f: VectorField, time: float, state: np.ndarray, step: float, *,
	solver: NewtonSolver, jacobian: VectorField | None,
	jacobian_structure: JacobianStructure, observe: bool = True,
) -> _HBVMStepResult:
	"""Solve the two-coefficient HBVM(4,2) equation for one complete step."""
	initial_state: np.ndarray = np.asarray(state, dtype=float)
	state_size = initial_state.size
	counter: _EvaluationCounter = _EvaluationCounter()
	stage_times = time + step * _HBVM42_NODES

	# A first-order stage predictor supplies a substantially better Legendre
	# initial guess than (f(y_n), 0) while retaining a small fixed setup cost.
	initial_field: np.ndarray = _evaluate_vector_field(f, time, initial_state, counter)
	predicted_stages = initial_state + step * np.outer(
		_HBVM42_NODES,
		initial_field,
	)
	predicted_fields: np.ndarray = np.stack(
		[
			_evaluate_vector_field(f, float(stage_time), stage, counter)
			for stage_time, stage in zip(
				stage_times,
				predicted_stages,
				strict=True,
			)
		],
		axis=0,
	)
	coefficients = _HBVM42_LEGENDRE.T @ (
		_HBVM42_WEIGHTS[:, None] * predicted_fields
	)
	tolerance = solver.tolerance(initial_state)
	residual_evaluations = 0
	jacobian_evaluations = 0

	def residual(
		candidate: np.ndarray,
	) -> tuple[np.ndarray, _ResidualPayload]:
		"""Evaluate the rank-two HBVM stage equation."""
		nonlocal residual_evaluations
		stages = initial_state + step * (_HBVM42_INTEGRALS @ candidate)
		fields: np.ndarray = np.stack(
			[
				_evaluate_vector_field(f, float(stage_time), stage, counter)
				for stage_time, stage in zip(stage_times, stages, strict=True)
			],
			axis=0,
		)
		projected = _HBVM42_LEGENDRE.T @ (
			_HBVM42_WEIGHTS[:, None] * fields
		)
		residual_evaluations += 1
		return candidate - projected, _ResidualPayload(stages, fields, projected)

	def evaluate_flat(candidate: np.ndarray) -> tuple[np.ndarray, _ResidualPayload]:
		value: np.ndarray
		data: _ResidualPayload
		value, data = residual(candidate.reshape(2, state_size))
		return value.reshape(-1), data

	# Preserve the established capability-check boundary after the first residual.
	cached: tuple[np.ndarray, _ResidualPayload] = evaluate_flat(coefficients.reshape(-1))
	selected_jacobian_method = _resolved_jacobian_method(jacobian, solver.jacobian_method,
		initial_time=time, initial_state=initial_state, structure=jacobian_structure, probe=False)

	def correction(unknown: np.ndarray, current_residual: np.ndarray, payload: _ResidualPayload) -> np.ndarray:
		"""Keep HBVM's stage Jacobian and field counters beside its equations."""
		nonlocal jacobian_evaluations
		if selected_jacobian_method == "analytic":
			assert jacobian is not None
			stage_jacobians: tuple[np.ndarray, ...] = _analytic_stage_jacobians(
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
				counter=counter,
			)
		jacobian_evaluations += 1
		residual_jacobian: np.ndarray = _reduced_residual_jacobian(stage_jacobians, step)
		try:
			correction: np.ndarray = np.linalg.solve(
				residual_jacobian,
				-current_residual,
			)
		except np.linalg.LinAlgError as exc:
			raise RuntimeError("The HBVM42 Newton Jacobian is singular.") from exc
		if not np.all(np.isfinite(correction)):
			raise RuntimeError("The HBVM42 Newton correction became non-finite.")

		return np.asarray(correction)

	root: _NonlinearResult[_ResidualPayload] = solver.solve(evaluate_flat, coefficients.reshape(-1), correction,
		scale=initial_state, context=f"HBVM42 at t={time:.8g}",
		initial_evaluation=cached,
		norm=lambda r: float(np.linalg.norm(r.reshape(2, state_size), ord=np.inf)),
		update_returns_correction=True, time=time, duration=step, observe=observe)
	payload = root.payload
	accepted = initial_state + step * payload.projected_coefficients[0]
	return _HBVMStepResult(state=np.asarray(accepted), stage_states=payload.stages,
		iterations=root.iterations,
		residual_norm=float(np.linalg.norm(root.residual.reshape(2, state_size), ord=np.inf)),
		tolerance=tolerance, residual_evaluations=root.residual_evaluations,
		jacobian_evaluations=jacobian_evaluations, vector_field_evaluations=counter.vector_field_evaluations)


@dataclass(slots=True)
class HBVM42(FixedStepMethod[_HBVMStepResult]):
	"""Fourth-order HBVM(4,2) with four quadrature nodes and two coefficients.

	For autonomous polynomial Hamiltonians of degree at most four, energy is
	preserved up to algebraic and round-off error. The method is not generally
	symplectic. Its default solver retains damped Newton corrections.
	"""

	jax_mode = 'device_resident'
	solver: NewtonSolver = field(default_factory=lambda: NewtonSolver(backtracking=True), kw_only=True)
	progress: bool = False

	def __post_init__(self) -> None:
		FixedStepMethod.__post_init__(self)
		validate_solver(self.solver, newton_only=True)

	def initialize(self, request: SimulationRequest) -> None:
		"""Record the method identity without probing deferred derivatives."""
		self.metadata = {'method_order': 4, 'quadrature_stage_count': 4,
			'legendre_rank': 2, 'jacobian_method': self.solver.jacobian_method}

	def _solve_physical(self, time: float, state: np.ndarray, step: float,
	                    *, observe: bool = True) -> _HBVMStepResult:
		return _advance_hbvm42(self.f, time, state, step, solver=self.solver,
			jacobian=self.jacobian, jacobian_structure=self.jacobian_structure, observe=observe)

	def advance(self, time: float, state: np.ndarray, step: float, *, observe: bool = True) -> StepResult[_HBVMStepResult]:
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
		}, result, quadrature)

	def build_observation(self, info: StepInfo, result: StepResult[_HBVMStepResult]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return self._solve_physical(info.time, candidate, info.duration, observe=False).state
		return IntegrationStep(
			dynamics_name='', method_name=type(self).__name__,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=info.state_before.copy(),
			state_after=result.state.copy(), map_state=map_state,
		)


__all__ = ['HBVM42', 'HBVMJacobianMethod']
