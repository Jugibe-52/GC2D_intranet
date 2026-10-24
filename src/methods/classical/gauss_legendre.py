"""Implicit two-stage, fourth-order Gauss--Legendre collocation method."""

from __future__ import annotations
from solvers.nonlinear import NewtonSolver, validate_solver

from dataclasses import dataclass, field

import numpy as np

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

	residuals: tuple[np.ndarray, np.ndarray]
	fields: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True, slots=True)
class _GaussStepResult:
	"""Accepted physical state, stages, and Newton work for one complete step."""

	state: np.ndarray
	stage_states: tuple[np.ndarray, np.ndarray]
	iterations: int
	residual_evaluations: int
	residual_norm: float


def _stage_evaluation(
	f: VectorField,
	time: float,
	state: np.ndarray,
	step: float,
	stage_states: tuple[np.ndarray, np.ndarray],
) -> _StageEvaluation:
	"""Evaluate both fields and coupled collocation residuals."""
	stage_times = time + step * _GAUSS_NODES
	fields: tuple[np.ndarray, ...] = tuple(
		_checked_vector_field(f, float(stage_time), stage_state)
		for stage_time, stage_state in zip(
			stage_times,
			stage_states,
			strict=True,
		)
	)
	residuals: tuple[Any, ...] = tuple(
		stage_states[index]
		- state
		- step
		* sum(
			_GAUSS_MATRIX[index, column] * fields[column]
			for column in range(2)
		)
		for index in range(2)
	)
	return _StageEvaluation(
		residuals=(residuals[0], residuals[1]),
		fields=(fields[0], fields[1]),
	)


def _stage_jacobians(
	f: VectorField, time: float, step: float,
	stage_states: tuple[np.ndarray, np.ndarray], *,
	jacobian: VectorField | None, jacobian_structure: JacobianStructure,
	jacobian_method: ResolvedGaussJacobianMethod, jacobian_relative_step: float,
) -> tuple[np.ndarray, np.ndarray]:
	"""Evaluate stage derivatives only after a residual fails convergence."""
	stage_times = time + step * _GAUSS_NODES
	if jacobian_method == 'analytic':
		assert jacobian is not None
		jacobians: tuple[np.ndarray, ...] = tuple(
			_checked_analytic_jacobians(jacobian, float(t), z, jacobian_structure)
			for t, z in zip(stage_times, stage_states, strict=True)
		)
	else:
		jacobians = tuple(_dense_finite_difference_jacobian(f, float(t), z,
			relative_step=jacobian_relative_step)
			for t, z in zip(stage_times, stage_states, strict=True))
	return jacobians[0], jacobians[1]

def _particle_vectors(first: np.ndarray, second: np.ndarray, components: int) -> np.ndarray:
	"""Gather component-major vectors into stage-major independent particles."""
	return np.concatenate((first.reshape(components, -1).T, second.reshape(components, -1).T), axis=1)

def _component_major_stage_corrections(corrections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Repack the two stage corrections into component-major coordinates."""
	components = corrections.shape[1] // 2
	return corrections[:, :components].T.reshape(-1), corrections[:, components:].T.reshape(-1)

def _analytic_newton_correction(
	step: float,
	evaluation: _StageEvaluation,
	jacobians: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
	"""Solve independent two-stage Gauss Newton systems for every particle."""
	blocks_1: np.ndarray
	blocks_2: np.ndarray
	blocks_1, blocks_2 = jacobians
	particle_count = blocks_1.shape[0]
	components = blocks_1.shape[1]
	identity: np.ndarray = np.broadcast_to(np.eye(components), (particle_count, components, components))
	top: np.ndarray = np.concatenate(
		(
			identity - step * _GAUSS_MATRIX[0, 0] * blocks_1,
			-step * _GAUSS_MATRIX[0, 1] * blocks_2,
		),
		axis=-1,
	)
	bottom: np.ndarray = np.concatenate(
		(
			-step * _GAUSS_MATRIX[1, 0] * blocks_1,
			identity - step * _GAUSS_MATRIX[1, 1] * blocks_2,
		),
		axis=-1,
	)
	matrix: np.ndarray = np.concatenate((top, bottom), axis=-2)
	residual: np.ndarray = _particle_vectors(*evaluation.residuals, components)
	corrections: np.ndarray = _solve_particle_systems(
		matrix, -residual,
		singular_message="The per-particle Gauss Newton matrix is singular.",
	)
	return _component_major_stage_corrections(corrections)


def _dense_newton_correction(
	step: float,
	evaluation: _StageEvaluation,
	jacobians: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
	"""Solve the generic dense coupled-stage Newton equation."""
	identity: np.ndarray = np.eye(evaluation.residuals[0].size)
	first_jacobian: np.ndarray
	second_jacobian: np.ndarray
	first_jacobian, second_jacobian = jacobians
	matrix: np.ndarray = np.block(
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
	residual: np.ndarray = np.concatenate(evaluation.residuals)
	try:
		correction: np.ndarray = np.linalg.solve(matrix, -residual)
	except np.linalg.LinAlgError as exc:
		raise RuntimeError("The Gauss Newton matrix is singular.") from exc
	dimension = identity.shape[0]
	return correction[:dimension], correction[dimension:]


def _solve_gauss_step(
	f: VectorField, time: float, state: np.ndarray, step: float, *,
	solver: NewtonSolver, jacobian: VectorField | None,
	jacobian_structure: JacobianStructure,
	jacobian_method: ResolvedGaussJacobianMethod, observe: bool = True,
) -> _GaussStepResult:
	"""Solve the two coupled collocation stages with full Newton corrections."""
	value: np.ndarray = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The Gauss physical state must be a finite vector.")
	initial_field: np.ndarray = _checked_vector_field(f, time, value)
	stage_states = (
		value + step * _GAUSS_NODES[0] * initial_field,
		value + step * _GAUSS_NODES[1] * initial_field,
	)
	size = value.size

	def evaluate(unknown: np.ndarray) -> tuple[np.ndarray, _StageEvaluation]:
		"""Pack both stage residuals without changing their infinity norm."""
		evaluation: _StageEvaluation = _stage_evaluation(
			f, time, value, step, (unknown[:size], unknown[size:]),
		)
		return np.concatenate(evaluation.residuals), evaluation

	def update(unknown: np.ndarray, residual: np.ndarray, evaluation: _StageEvaluation) -> np.ndarray:
		"""Keep the collocation-specific block algebra beside its stage equations."""
		jacobians: tuple[np.ndarray, np.ndarray] = _stage_jacobians(
			f,
			time,
			step,
			(unknown[:size], unknown[size:]),
			jacobian_method=jacobian_method,
			jacobian_relative_step=solver.jacobian_relative_step,
			jacobian=jacobian, jacobian_structure=jacobian_structure,
		)
		if jacobian_method == "analytic" and jacobian_structure == "particle":
			corrections: tuple[np.ndarray, np.ndarray] = _analytic_newton_correction(
				step,
				evaluation,
				jacobians,
			)
		else:
			corrections = _dense_newton_correction(step, evaluation, jacobians)
		if not all(np.all(np.isfinite(value)) for value in corrections):
			raise RuntimeError("The Gauss Newton correction became non-finite.")
		return np.asarray(unknown + np.concatenate(corrections))

	root: _NonlinearResult[_StageEvaluation] = solver.solve(
		evaluate, np.concatenate(stage_states), update, scale=value,
		context=f"GaussLegendre4 at t={time:.16g} with h={step:.16g}",
		time=time, duration=step, observe=observe,
	)
	state_after = value + step * 0.5 * (root.payload.fields[0] + root.payload.fields[1])
	if not np.all(np.isfinite(state_after)):
		raise RuntimeError("The converged Gauss state is non-finite.")
	return _GaussStepResult(
		state=np.asarray(state_after),
		stage_states=(root.unknown[:size].copy(), root.unknown[size:].copy()),
		iterations=root.iterations, residual_evaluations=root.residual_evaluations,
		residual_norm=float(np.linalg.norm(root.residual, ord=np.inf)),
	)


@dataclass(slots=True)
class GaussLegendre4(FixedStepMethod[_GaussStepResult]):
	"""Two-stage, fourth-order symmetric Gauss--Legendre Runge--Kutta method."""

	jax_mode = 'device_resident'
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

	def _solve_physical(self, time: float, physical: np.ndarray, step: float,
	                    *, observe: bool = True) -> _GaussStepResult:
		return _solve_gauss_step(self.f, time, physical, step, solver=self.solver,
			jacobian=self.jacobian, jacobian_structure=self.jacobian_structure,
			jacobian_method=self.resolved_jacobian_method, observe=observe)

	def advance(self, time: float, value: np.ndarray, step: float, *, observe: bool = True) -> StepResult[_GaussStepResult]:
		"""Solve the physical stages and optionally expose accepted quadrature."""
		physical: np.ndarray = np.asarray(self._checked_state(value), dtype=float)
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
		return StepResult(result.state, statistics, result, quadrature)

	def build_observation(self, info: StepInfo, step: StepResult[_GaussStepResult]) -> GaussLegendre4IntegrationStep:
		"""Build independent snapshots and a replay without iteration callbacks."""
		result = step.details
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return self._solve_physical(info.time, candidate, info.duration, observe=False).state
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
			first_stage_state=result.stage_states[0].copy(), second_stage_state=result.stage_states[1].copy(),
		)


__all__ = ['GAUSS_JACOBIAN_METHODS', 'GaussJacobianMethod', 'GaussLegendre4']
