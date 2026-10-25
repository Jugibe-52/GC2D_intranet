"""Implicit two-stage, fourth-order Gauss--Legendre collocation method."""

from __future__ import annotations
from solvers.nonlinear import NewtonSolver, validate_solver

from dataclasses import dataclass, field

import numpy as np
from typing import cast
from contracts.arrays import BackendArray
from solvers.linear import solve_linear
from types import ModuleType
from contracts.arrays import array_namespace, checked_finite, map_arrays

from contracts.methods.functions import VectorField, JacobianStructure
from contracts.methods.quadrature import StageQuadrature

from methods.base import FixedStepMethod
from contracts.methods.step import StepInfo, StepResult
from contracts.states.history import DiagnosticValue
from contracts.methods.classical.gauss_legendre import GaussLegendre4IntegrationStep
from request import SimulationRequest
from solvers.linear import _solve_particle_systems
from ._jacobians import (
	JacobianMethod as GaussJacobianMethod,
	ResolvedJacobianMethod as ResolvedGaussJacobianMethod,
	_checked_vector_field,
	_dense_finite_difference_jacobian,
	_checked_analytic_jacobians,
	_resolved_jacobian_method,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from solvers.nonlinear import _NonlinearResult
	from typing import Any


GAUSS_JACOBIAN_METHODS: tuple[GaussJacobianMethod, ...] = (
	"auto",
	"analytic",
	"finite_difference",
)

_ROOT_THREE_OVER_SIX: 'float' = float(np.sqrt(3.0) / 6.0)
_GAUSS_NODES: 'np.ndarray' = np.asarray(
	(0.5 - _ROOT_THREE_OVER_SIX, 0.5 + _ROOT_THREE_OVER_SIX),
	dtype=float,
)
_GAUSS_MATRIX: 'np.ndarray' = np.asarray(
	(
		(0.25, 0.25 - _ROOT_THREE_OVER_SIX),
		(0.25 + _ROOT_THREE_OVER_SIX, 0.25),
	),
	dtype=float,
)


@dataclass(frozen=True, slots=True)
class _StageEvaluation:
	"""Fields and residuals at two candidate collocation stage states."""

	residuals: tuple[BackendArray, BackendArray]
	fields: tuple[BackendArray, BackendArray]


@dataclass(frozen=True, slots=True)
class _GaussStepResult:
	"""Accepted physical state, stages, and Newton work for one complete step."""

	state: BackendArray
	stage_states: tuple[BackendArray, BackendArray]
	iterations: int
	residual_evaluations: int
	residual_norm: float
	converged: Any = True


def _stage_evaluation(
	f: VectorField, time: float, state: BackendArray, step: float,
	stage_states: tuple[BackendArray, BackendArray],
) -> _StageEvaluation:
	"""Evaluate both fields and coupled collocation residuals."""
	xp: ModuleType = array_namespace(state)
	stage_times = time + step * _GAUSS_NODES
	fields: tuple[BackendArray, ...] = tuple(map_arrays(
		lambda time, state: _checked_vector_field(f, time, state), stage_times, xp.stack(stage_states)))
	residuals: tuple[Any, ...] = tuple(
		stage_states[index] - state - step * sum(
			_GAUSS_MATRIX[index, column] * fields[column] for column in range(2))
		for index in range(2))
	return _StageEvaluation((residuals[0], residuals[1]), (fields[0], fields[1]))


def _stage_jacobians(
	f: VectorField, time: float, step: float,
	stage_states: tuple[BackendArray, BackendArray], *,
	jacobian: VectorField | None, jacobian_structure: JacobianStructure,
	jacobian_method: ResolvedGaussJacobianMethod, jacobian_relative_step: float,
) -> tuple[BackendArray, BackendArray]:
	"""Evaluate stage derivatives only after a residual fails convergence."""
	xp: ModuleType = array_namespace(stage_states)
	stage_times = time + step * _GAUSS_NODES
	if jacobian_method == 'analytic':
		assert jacobian is not None
		jacobians: tuple[BackendArray, ...] = tuple(map_arrays(
			lambda t, z: _checked_analytic_jacobians(jacobian, t, z, jacobian_structure),
			stage_times, xp.stack(stage_states)))
	else:
		jacobians = tuple(map_arrays(lambda t, z: _dense_finite_difference_jacobian(f, t, z,
			relative_step=jacobian_relative_step), stage_times, xp.stack(stage_states)))
	return jacobians[0], jacobians[1]

def _particle_vectors(first: BackendArray, second: BackendArray, components: int) -> BackendArray:
	"""Gather component-major vectors into stage-major independent particles."""
	xp: ModuleType = array_namespace(first)
	return cast(BackendArray, xp.concatenate((first.reshape(components, -1).T, second.reshape(components, -1).T), axis=1))

def _component_major_stage_corrections(corrections: BackendArray) -> tuple[BackendArray, BackendArray]:
	"""Repack the two stage corrections into component-major coordinates."""
	components = corrections.shape[1] // 2
	return corrections[:, :components].T.reshape(-1), corrections[:, components:].T.reshape(-1)

def _analytic_newton_correction(
	step: float,
	evaluation: _StageEvaluation,
	jacobians: tuple[BackendArray, BackendArray],
) -> tuple[BackendArray, BackendArray]:
	"""Solve independent two-stage Gauss Newton systems for every particle."""
	xp: ModuleType = array_namespace(jacobians)
	blocks_1: BackendArray
	blocks_2: BackendArray
	blocks_1, blocks_2 = jacobians
	particle_count = blocks_1.shape[0]
	components = blocks_1.shape[1]
	identity: BackendArray = xp.broadcast_to(xp.eye(components), (particle_count, components, components))
	top: BackendArray = xp.concatenate(
		(
			identity - step * _GAUSS_MATRIX[0, 0] * blocks_1,
			-step * _GAUSS_MATRIX[0, 1] * blocks_2,
		),
		axis=-1,
	)
	bottom: BackendArray = xp.concatenate(
		(
			-step * _GAUSS_MATRIX[1, 0] * blocks_1,
			identity - step * _GAUSS_MATRIX[1, 1] * blocks_2,
		),
		axis=-1,
	)
	matrix: BackendArray = xp.concatenate((top, bottom), axis=-2)
	residual: BackendArray = _particle_vectors(*evaluation.residuals, components)
	corrections: BackendArray = _solve_particle_systems(
		matrix, -residual,
		singular_message="The per-particle Gauss Newton matrix is singular.",
	)
	return _component_major_stage_corrections(corrections)


def _dense_newton_correction(
	step: float,
	evaluation: _StageEvaluation,
	jacobians: tuple[BackendArray, BackendArray],
) -> tuple[BackendArray, BackendArray]:
	"""Solve the generic dense coupled-stage Newton equation."""
	xp: ModuleType = array_namespace(jacobians)
	identity: BackendArray = xp.eye(evaluation.residuals[0].size)
	first_jacobian: BackendArray
	second_jacobian: BackendArray
	first_jacobian, second_jacobian = jacobians
	matrix: BackendArray = xp.block(
		[
			[
				identity - step * _GAUSS_MATRIX[0, 0] * first_jacobian,
				-step * _GAUSS_MATRIX[0, 1] * second_jacobian,
			],
			[
				-step * _GAUSS_MATRIX[1, 0] * first_jacobian,
				identity - step * _GAUSS_MATRIX[1, 1] * second_jacobian,
			],
		]
	)
	residual: BackendArray = xp.concatenate(evaluation.residuals)
	correction: BackendArray = solve_linear(matrix, -residual, singular_message="The Gauss Newton matrix is singular.")
	dimension = identity.shape[0]
	return correction[:dimension], correction[dimension:]


def _solve_gauss_step(
	f: VectorField, time: float, state: BackendArray, step: float, *,
	solver: NewtonSolver, jacobian: VectorField | None,
	jacobian_structure: JacobianStructure,
	jacobian_method: ResolvedGaussJacobianMethod, observe: bool = True,
) -> _GaussStepResult:
	"""Solve the two coupled collocation stages with full Newton corrections."""
	xp: ModuleType = array_namespace(state)
	value: BackendArray = xp.asarray(state, dtype=float)
	value = checked_finite(value, "The Gauss physical state must be a finite vector.", ValueError)
	if value.ndim != 1 or value.size == 0:
		raise ValueError("The Gauss physical state must be a finite vector.")
	initial_field: BackendArray = _checked_vector_field(f, time, value)
	stage_states = (
		value + step * _GAUSS_NODES[0] * initial_field,
		value + step * _GAUSS_NODES[1] * initial_field,
	)
	size = value.size

	def evaluate(unknown: BackendArray) -> tuple[BackendArray, _StageEvaluation]:
		"""Pack both stage residuals without changing their infinity norm."""
		evaluation: _StageEvaluation = _stage_evaluation(
			f, time, value, step, (unknown[:size], unknown[size:]),
		)
		return xp.concatenate(evaluation.residuals), evaluation

	def update(unknown: BackendArray, residual: BackendArray, evaluation: _StageEvaluation) -> BackendArray:
		"""Keep the collocation-specific block algebra beside its stage equations."""
		jacobians: tuple[BackendArray, BackendArray] = _stage_jacobians(
			f,
			time,
			step,
			(unknown[:size], unknown[size:]),
			jacobian_method=jacobian_method,
			jacobian_relative_step=solver.jacobian_relative_step,
			jacobian=jacobian, jacobian_structure=jacobian_structure,
		)
		if jacobian_method == "analytic" and jacobian_structure == "particle":
			corrections: tuple[BackendArray, BackendArray] = _analytic_newton_correction(
				step,
				evaluation,
				jacobians,
			)
		else:
			corrections = _dense_newton_correction(step, evaluation, jacobians)
		corrections = (checked_finite(corrections[0], "The Gauss Newton correction became non-finite.", RuntimeError),
			checked_finite(corrections[1], "The Gauss Newton correction became non-finite.", RuntimeError))
		return cast(BackendArray, xp.asarray(unknown + xp.concatenate(corrections)))

	root: _NonlinearResult[_StageEvaluation] = solver.solve(
		evaluate, xp.concatenate(stage_states), update, scale=value,
		context=f"GaussLegendre4 at t={time} with h={step}",
		time=time, duration=step, observe=observe,
	)
	state_after = value + step * 0.5 * (root.payload.fields[0] + root.payload.fields[1])
	state_after = checked_finite(state_after, "The converged Gauss state is non-finite.", RuntimeError)
	return _GaussStepResult(
		state=xp.asarray(state_after),
		stage_states=(root.unknown[:size].copy(), root.unknown[size:].copy()),
		iterations=root.iterations, residual_evaluations=root.residual_evaluations,
		residual_norm=xp.linalg.norm(root.residual, ord=np.inf), converged=root.converged,
	)


@dataclass(slots=True)
class GaussLegendre4(FixedStepMethod[_GaussStepResult]):
	"""Two-stage, fourth-order symmetric Gauss--Legendre Runge--Kutta method."""

	solver: NewtonSolver = field(default_factory=lambda: NewtonSolver(1e-14, 1e-13, 20), kw_only=True)
	progress: bool = False
	resolved_jacobian_method: ResolvedGaussJacobianMethod = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		FixedStepMethod.__post_init__(self)
		validate_solver(self.solver, newton_only=True)

	def initialize(self, request: SimulationRequest) -> None:
		"""Resolve the supplied derivative without retaining physical resources."""
		self.resolved_jacobian_method = _resolved_jacobian_method(self.jacobian,
			self.solver.jacobian_method, initial_time=request.t_span[0],
			initial_state=self.initial_state, structure=self.jacobian_structure)
		self.metadata = {
			'stage_count': 2, 'designed_order': 4, 'nonlinear_solver': 'newton',
			'nonlinear_solves_per_step': 1,
			'nonlinear_absolute_tolerance': self.solver.absolute_tolerance,
			'nonlinear_relative_tolerance': self.solver.relative_tolerance,
			'nonlinear_max_iterations': self.solver.max_iterations,
			'newton_jacobian_method': self.resolved_jacobian_method,
			'requested_newton_jacobian_method': self.solver.jacobian_method,
			'newton_jacobian_relative_step': self.solver.jacobian_relative_step,
		}

	def _solve_physical(self, time: float, physical: BackendArray, step: float,
	                    *, observe: bool = True) -> _GaussStepResult:
		return _solve_gauss_step(self.f, time, physical, step, solver=self.solver,
			jacobian=self.jacobian, jacobian_structure=self.jacobian_structure,
			jacobian_method=self.resolved_jacobian_method, observe=observe)

	def advance(self, time: float, value: BackendArray, step: float, *, observe: bool = True) -> StepResult[_GaussStepResult]:
		"""Solve the physical stages and optionally expose accepted quadrature."""
		xp: ModuleType = array_namespace(value)
		physical: BackendArray = xp.asarray(self._checked_state(value), dtype=float)
		result: _GaussStepResult = self._solve_physical(time, physical, step, observe=observe)
		statistics = {
			'nonlinear_iterations': result.iterations,
			'residual_evaluations': result.residual_evaluations,
			'nonlinear_residual_norms': result.residual_norm,
			'nonlinear_tolerances': self.solver.tolerance(physical),
		}
		quadrature: StageQuadrature | None = None
		if self.capture_quadrature:
			quadrature = StageQuadrature(tuple(time + step * _GAUSS_NODES),
				result.stage_states, (step * .5, step * .5))
		return StepResult(result.state, statistics, result, quadrature, result.converged)

	def build_observation(self, info: StepInfo, step: StepResult[_GaussStepResult]) -> GaussLegendre4IntegrationStep:
		"""Build independent snapshots and a replay without iteration callbacks."""
		result = step.details
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return np.asarray(self._solve_physical(info.time, candidate, info.duration, observe=False).state)
		return GaussLegendre4IntegrationStep(
			dynamics_name='', method_name=type(self).__name__,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=info.state_before.copy(),
			state_after=result.state.copy(), map_state=map_state,
			first_stage_time=info.time + info.duration * _GAUSS_NODES[0],
			second_stage_time=info.time + info.duration * _GAUSS_NODES[1],
			newton_iterations=result.iterations, residual_evaluations=result.residual_evaluations,
			newton_residual_norm=result.residual_norm,
			newton_tolerance=float(step.statistics['nonlinear_tolerances']),
			first_stage_state=np.asarray(result.stage_states[0].copy()), second_stage_state=np.asarray(result.stage_states[1].copy()),
		)


__all__ = ['GAUSS_JACOBIAN_METHODS', 'GaussJacobianMethod', 'GaussLegendre4']
