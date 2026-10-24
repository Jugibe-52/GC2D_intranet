"""Five-stage, fourth-order singly diagonally implicit Runge--Kutta method."""

from __future__ import annotations
from solvers.nonlinear import NewtonSolver, validate_solver

from dataclasses import dataclass, field

import numpy as np

from contracts.methods.functions import VectorField, JacobianStructure
from contracts.methods.quadrature import StageQuadrature

from methods.base import FixedStepMethod
from contracts.methods.step import StepInfo, StepResult
from contracts.states.history import DiagnosticValue
from contracts.methods.observation import IntegrationStep
from request import SimulationRequest
from solvers.linear import _solve_particle_systems
from ._jacobians import (
	JacobianMethod as SDIRKJacobianMethod,
	ResolvedJacobianMethod as ResolvedSDIRKJacobianMethod,
	_checked_vector_field,
	_dense_finite_difference_jacobian,
	_checked_analytic_jacobians,
	_resolved_jacobian_method,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from solvers.nonlinear import _NonlinearResult
	from typing import Any


SDIRK_JACOBIAN_METHODS: tuple[SDIRKJacobianMethod, ...] = (
	"auto",
	"analytic",
	"finite_difference",
)

# Skvortsov's S54b scheme has a common diagonal gamma=1/4.  Its fifth row is
# also the output formula, so the accepted state equals the last stage up to
# the nonlinear solve tolerance.
SDIRK4_TABLEAU_A: 'np.ndarray' = np.asarray(
	(
		(1.0 / 4.0, 0.0, 0.0, 0.0, 0.0),
		(-1.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0),
		(1.0 / 8.0, 1.0 / 8.0, 1.0 / 4.0, 0.0, 0.0),
		(-3.0 / 2.0, 3.0 / 4.0, 3.0 / 2.0, 1.0 / 4.0, 0.0),
		(0.0, 1.0 / 6.0, 2.0 / 3.0, -1.0 / 12.0, 1.0 / 4.0),
	),
	dtype=float,
)
SDIRK4_TABLEAU_B: 'np.ndarray' = np.asarray(
	(0.0, 1.0 / 6.0, 2.0 / 3.0, -1.0 / 12.0, 1.0 / 4.0),
	dtype=float,
)
SDIRK4_TABLEAU_C: 'np.ndarray' = np.sum(SDIRK4_TABLEAU_A, axis=1)
for _coefficient_array in (
	SDIRK4_TABLEAU_A,
	SDIRK4_TABLEAU_B,
	SDIRK4_TABLEAU_C,
):
	_coefficient_array.setflags(write=False)

_SDIRK_DIAGONAL: 'float' = float(SDIRK4_TABLEAU_A[0, 0])
_STAGE_COUNT: 'int' = int(SDIRK4_TABLEAU_B.size)


@dataclass(frozen=True, slots=True)
class _StageResult:
	"""Accepted stage state, field, and Newton work."""

	state: np.ndarray
	field: np.ndarray
	iterations: int
	residual_evaluations: int
	residual_norm: float


@dataclass(frozen=True, slots=True)
class _SDIRKStepResult:
	"""Accepted physical state and five sequential implicit stages."""

	state: np.ndarray
	stage_states: tuple[np.ndarray, ...]
	stage_fields: tuple[np.ndarray, ...]
	stage_iterations: np.ndarray
	stage_residual_evaluations: np.ndarray
	stage_residual_norms: np.ndarray
	tolerance: float


def _analytic_newton_correction(step: float, residual: np.ndarray, jacobians: np.ndarray) -> np.ndarray:
	"""Solve independent diagonal-stage systems in component-major coordinates."""
	particle_count = jacobians.shape[0]
	components = jacobians.shape[1]
	matrix = np.broadcast_to(np.eye(components), jacobians.shape) - step * _SDIRK_DIAGONAL * jacobians
	particle_residuals: np.ndarray = residual.reshape(components, particle_count).T
	corrections: np.ndarray = _solve_particle_systems(matrix, -particle_residuals,
		singular_message='A per-particle SDIRK Newton matrix is singular.')
	return corrections.T.reshape(-1)

def _solve_stage(
	f: VectorField, stage_time: float, right_side: np.ndarray,
	initial_guess: np.ndarray, step: float, *, solver: NewtonSolver,
	jacobian: VectorField | None, jacobian_structure: JacobianStructure,
	jacobian_method: ResolvedSDIRKJacobianMethod,
	stage_index: int, step_time: float, scale: np.ndarray, observe: bool,
) -> _StageResult:
	"""Solve one diagonal stage, with algebraic policy supplied by the solver."""
	def evaluate(stage_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		value: np.ndarray = _checked_vector_field(f, stage_time, stage_state)
		return stage_state - right_side - step * _SDIRK_DIAGONAL * value, value

	def update(stage_state: np.ndarray, residual: np.ndarray, value: np.ndarray) -> np.ndarray:
		if jacobian_method == 'analytic':
			assert jacobian is not None
			derivative: np.ndarray = _checked_analytic_jacobians(jacobian,
				stage_time, stage_state, jacobian_structure)
		else:
			derivative = _dense_finite_difference_jacobian(f, stage_time, stage_state,
				relative_step=solver.jacobian_relative_step)
		if jacobian_method == 'analytic' and jacobian_structure == 'particle':
			correction: np.ndarray = _analytic_newton_correction(step, residual, derivative)
		else:
			matrix = np.eye(stage_state.size) - step * _SDIRK_DIAGONAL * derivative
			try:
				correction = np.linalg.solve(matrix, -residual)
			except np.linalg.LinAlgError as exc:
				raise RuntimeError('The SDIRK Newton matrix is singular.') from exc
		if not np.all(np.isfinite(correction)):
			raise RuntimeError('The SDIRK Newton correction became non-finite.')
		return np.asarray(stage_state + correction)

	root: _NonlinearResult[np.ndarray] = solver.solve(evaluate, initial_guess, update,
		scale=scale, context=f'SDIRK4 at stage time t={stage_time:.16g} with h={step:.16g}',
		time=step_time, duration=step, stage_index=stage_index, observe=observe)
	return _StageResult(state=root.unknown, field=root.payload, iterations=root.iterations,
		residual_evaluations=root.residual_evaluations,
		residual_norm=float(np.linalg.norm(root.residual, ord=np.inf)))

def _solve_sdirk_step(
	f: VectorField, time: float, state: np.ndarray, step: float, *,
	solver: NewtonSolver, jacobian: VectorField | None,
	jacobian_structure: JacobianStructure,
	jacobian_method: ResolvedSDIRKJacobianMethod, observe: bool = True,
) -> _SDIRKStepResult:
	"""Advance one S54b step through five sequential implicit solves."""
	value: np.ndarray = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The SDIRK physical state must be a finite vector.")
	initial_field: np.ndarray = _checked_vector_field(f, time, value)
	tolerance = solver.tolerance(value)
	stage_states: list[np.ndarray] = []
	stage_fields: list[np.ndarray] = []
	stage_iterations: list[int] = []
	stage_residual_evaluations: list[int] = []
	stage_residual_norms: list[float] = []
	for stage_index in range(_STAGE_COUNT):
		right_side = value + step * sum(
			SDIRK4_TABLEAU_A[stage_index, previous_index]
			* stage_fields[previous_index]
			for previous_index in range(stage_index)
		)
		initial_guess = value + (
			step * SDIRK4_TABLEAU_C[stage_index] * initial_field
		)
		result: _StageResult = _solve_stage(
			f,
			time + step * SDIRK4_TABLEAU_C[stage_index],
			right_side,
			initial_guess,
			step,
			solver=solver, jacobian=jacobian, jacobian_structure=jacobian_structure,
			jacobian_method=jacobian_method, stage_index=stage_index, step_time=time,
			scale=value, observe=observe,
		)
		stage_states.append(result.state)
		stage_fields.append(result.field)
		stage_iterations.append(result.iterations)
		stage_residual_evaluations.append(result.residual_evaluations)
		stage_residual_norms.append(result.residual_norm)
	state_after = value + step * sum(
		SDIRK4_TABLEAU_B[index] * stage_fields[index]
		for index in range(_STAGE_COUNT)
	)
	if not np.all(np.isfinite(state_after)):
		raise RuntimeError("The converged SDIRK state is non-finite.")
	return _SDIRKStepResult(
		state=np.asarray(state_after),
		stage_states=tuple(stage_states),
		stage_fields=tuple(stage_fields),
		stage_iterations=np.asarray(stage_iterations, dtype=int),
		stage_residual_evaluations=np.asarray(
			stage_residual_evaluations,
			dtype=int,
		),
		stage_residual_norms=np.asarray(stage_residual_norms, dtype=float),
		tolerance=tolerance,
	)


@dataclass(slots=True)
class SDIRK4(FixedStepMethod[_SDIRKStepResult]):
	"""Skvortsov S54b: five-stage, fourth-order non-geometric SDIRK."""

	jax_mode = 'device_resident'
	solver: NewtonSolver = field(default_factory=lambda: NewtonSolver(1e-14, 1e-13, 20), kw_only=True)
	progress: bool = False
	resolved_jacobian_method: ResolvedSDIRKJacobianMethod = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		FixedStepMethod.__post_init__(self)
		validate_solver(self.solver, newton_only=True)

	def initialize(self, request: SimulationRequest) -> None:
		"""Resolve supplied derivative functions and describe the S54b tableau."""
		self.resolved_jacobian_method = _resolved_jacobian_method(self.jacobian,
			self.solver.jacobian_method, initial_time=request.t_span[0],
			initial_state=self.initial_state, structure=self.jacobian_structure)
		self.metadata = {
			'stage_count': _STAGE_COUNT, 'designed_order': 4, 'tableau_name': 'Skvortsov S54b',
			'diagonal_coefficient': _SDIRK_DIAGONAL, 'stiffly_accurate': True,
			'symmetric': False, 'symplectic': False, 'nonlinear_solver': 'newton',
			'nonlinear_solves_per_step': _STAGE_COUNT,
			'nonlinear_absolute_tolerance': self.solver.absolute_tolerance,
			'nonlinear_relative_tolerance': self.solver.relative_tolerance,
			'nonlinear_max_iterations': self.solver.max_iterations,
			'newton_jacobian_method': self.resolved_jacobian_method,
			'requested_newton_jacobian_method': self.solver.jacobian_method,
			'newton_jacobian_relative_step': self.solver.jacobian_relative_step,
		}

	def _solve_physical(self, time: float, physical: np.ndarray, step: float,
	                    *, observe: bool = True) -> _SDIRKStepResult:
		return _solve_sdirk_step(self.f, time, physical, step, solver=self.solver,
			jacobian=self.jacobian, jacobian_structure=self.jacobian_structure,
			jacobian_method=self.resolved_jacobian_method, observe=observe)

	def advance(self, time: float, value: np.ndarray, step: float, *, observe: bool = True) -> StepResult[_SDIRKStepResult]:
		"""Solve the physical stages and optionally expose their signed weights."""
		physical: np.ndarray = np.asarray(self._checked_state(value), dtype=float)
		result: _SDIRKStepResult = self._solve_physical(time, physical, step, observe=observe)
		statistics: dict[str, np.ndarray | float | int] = {
			'nonlinear_iterations': int(np.sum(result.stage_iterations)),
			'residual_evaluations': int(np.sum(result.stage_residual_evaluations)),
			'nonlinear_residual_norms': float(np.max(result.stage_residual_norms)),
			'nonlinear_tolerances': result.tolerance,
			'stage_nonlinear_iterations': result.stage_iterations,
			'stage_residual_evaluations': result.stage_residual_evaluations,
			'stage_nonlinear_residual_norms': result.stage_residual_norms,
		}
		quadrature: StageQuadrature | None = None
		if self.capture_quadrature:
			quadrature = StageQuadrature(tuple(time + step * SDIRK4_TABLEAU_C),
				result.stage_states, tuple(step * SDIRK4_TABLEAU_B))
		return StepResult(result.state, statistics, result, quadrature)

	def build_observation(self, info: StepInfo, step: StepResult[_SDIRKStepResult]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return self._solve_physical(info.time, candidate, info.duration, observe=False).state
		return IntegrationStep(
			dynamics_name='', method_name=type(self).__name__,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=info.state_before.copy(),
			state_after=step.state.copy(), map_state=map_state,
		)


__all__ = ['SDIRK4_TABLEAU_A', 'SDIRK4_TABLEAU_B', 'SDIRK4_TABLEAU_C',
	'SDIRK_JACOBIAN_METHODS', 'SDIRK4', 'SDIRKJacobianMethod']
