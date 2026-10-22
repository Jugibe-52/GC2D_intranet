"""Fourth-order HBVM(4,2) with a reduced Legendre-stage solve."""

from __future__ import annotations
from solvers.nonlinear import NewtonSolver, AlgebraicSolver, validate_solver

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING, TypeAlias

import numpy as np

from dynamics import GuidingCenterJacobianSystem

from formulations.state import PhysicalFormulation
from methods.base import IntegrationMethod
from contracts.step import StepInfo, StepResult
from contracts.observation import IntegrationStep, StepObserver
from problem import InitialValueProblem
from request import SimulationRequest
from methods._validation import _positive_finite, _positive_integer


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


def _validated_jacobian_method(value: str) -> HBVMJacobianMethod:
	"""Return one supported HBVM Jacobian strategy."""
	if value not in ("auto", "analytic", "finite_difference"):
		raise ValueError(
			"`jacobian_method` must be 'auto', 'analytic', or 'finite_difference'."
		)
	return value  # type: ignore[return-value]


def _resolved_jacobian_method(
	dynamics: object,
	requested: HBVMJacobianMethod,
) -> HBVMJacobianMethod:
	"""Resolve HBVM's capability-based strategy without probing field Jacobians."""
	selected: HBVMJacobianMethod = requested
	if selected == "auto":
		selected = (
			"analytic"
			if isinstance(dynamics, GuidingCenterJacobianSystem)
			else "finite_difference"
		)
	if selected == "analytic" and not isinstance(dynamics, GuidingCenterJacobianSystem):
		raise TypeError(
			"Analytic HBVM Jacobians require GuidingCenterJacobianSystem dynamics."
		)
	return selected


def _evaluate_vector_field(
	dynamics: object,
	time: float,
	state: np.ndarray,
	counter: _EvaluationCounter,
) -> np.ndarray:
	"""Evaluate one shape-preserving finite vector field and update its counter."""
	evaluate: Any | None = getattr(dynamics, "vector_field", None)
	if not callable(evaluate):
		raise TypeError("HBVM42 requires dynamics with a callable vector field.")
	value: np.ndarray = np.asarray(evaluate(time, state), dtype=float)
	counter.vector_field_evaluations += 1
	if value.shape != state.shape or not np.all(np.isfinite(value)):
		raise ValueError(
			"The vector field must return a finite array matching the state shape."
		)
	return value


def _complete_particle_jacobian(
	particle_blocks: np.ndarray,
	state_size: int,
) -> np.ndarray:
	"""Assemble component-major particle blocks into one dense Jacobian."""
	if state_size % 2:
		raise ValueError("Analytic HBVM Jacobians require an even physical state size.")
	particle_count = state_size // 2
	blocks: np.ndarray = np.asarray(particle_blocks, dtype=float)
	if blocks.shape != (particle_count, 2, 2) or not np.all(np.isfinite(blocks)):
		raise ValueError(
			"Particle vector-field Jacobians must have shape (particle_count, 2, 2)."
		)
	result: np.ndarray = np.zeros((state_size, state_size), dtype=float)
	particles: np.ndarray = np.arange(particle_count)
	x_indices = particles
	y_indices = particle_count + particles
	result[x_indices, x_indices] = blocks[:, 0, 0]
	result[x_indices, y_indices] = blocks[:, 0, 1]
	result[y_indices, x_indices] = blocks[:, 1, 0]
	result[y_indices, y_indices] = blocks[:, 1, 1]
	return result


def _analytic_stage_jacobians(
	dynamics: GuidingCenterJacobianSystem,
	times: np.ndarray,
	stages: np.ndarray,
) -> tuple[np.ndarray, ...]:
	"""Return exact dense field Jacobians for the four quadrature stages."""
	return tuple(
		_complete_particle_jacobian(
			dynamics.particle_vector_field_jacobians(float(time), stage),
			stage.size,
		)
		for time, stage in zip(times, stages, strict=True)
	)


def _finite_difference_stage_jacobians(
	dynamics: object,
	times: np.ndarray,
	stages: np.ndarray,
	*,
	relative_step: float,
	counter: _EvaluationCounter,
) -> tuple[np.ndarray, ...]:
	"""Differentiate each stage field by centered component perturbations."""
	jacobians: list[np.ndarray] = []
	for time, stage in zip(times, stages, strict=True):
		state_size = stage.size
		jacobian: np.ndarray = np.empty((state_size, state_size), dtype=float)
		for column in range(state_size):
			increment = relative_step * max(1.0, abs(float(stage[column])))
			plus: Any = stage.copy()
			minus: Any = stage.copy()
			plus[column] += increment
			minus[column] -= increment
			jacobian[:, column] = (
				_evaluate_vector_field(dynamics, float(time), plus, counter)
				- _evaluate_vector_field(dynamics, float(time), minus, counter)
			) / (2.0 * increment)
		jacobians.append(jacobian)
	return tuple(jacobians)


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
	dynamics: object,
	time: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	jacobian_method: HBVMJacobianMethod,
	jacobian_relative_step: float,
	solver: NewtonSolver | None = None,
) -> _HBVMStepResult:
	"""Solve the two-coefficient HBVM(4,2) equation for one complete step."""
	initial_state: np.ndarray = np.asarray(state, dtype=float)
	state_size = initial_state.size
	counter: _EvaluationCounter = _EvaluationCounter()
	stage_times = time + step * _HBVM42_NODES

	# A first-order stage predictor supplies a substantially better Legendre
	# initial guess than (f(y_n), 0) while retaining a small fixed setup cost.
	initial_field: np.ndarray = _evaluate_vector_field(dynamics, time, initial_state, counter)
	predicted_stages = initial_state + step * np.outer(
		_HBVM42_NODES,
		initial_field,
	)
	predicted_fields: np.ndarray = np.stack(
		[
			_evaluate_vector_field(dynamics, float(stage_time), stage, counter)
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
	tolerance = absolute_tolerance + relative_tolerance * max(
		1.0,
		float(np.linalg.norm(initial_state, ord=np.inf)),
	)
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
				_evaluate_vector_field(dynamics, float(stage_time), stage, counter)
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
	selected_jacobian_method: Literal['auto'] | Literal['analytic'] | Literal['finite_difference'] = _resolved_jacobian_method(dynamics, jacobian_method)

	def correction(unknown: np.ndarray, current_residual: np.ndarray, payload: _ResidualPayload) -> np.ndarray:
		"""Keep HBVM's stage Jacobian and field counters beside its equations."""
		nonlocal jacobian_evaluations
		if selected_jacobian_method == "analytic":
			assert isinstance(dynamics, GuidingCenterJacobianSystem)
			stage_jacobians: tuple[np.ndarray, ...] = _analytic_stage_jacobians(
				dynamics,
				stage_times,
				payload.stages,
			)
		else:
			stage_jacobians = _finite_difference_stage_jacobians(
				dynamics,
				stage_times,
				payload.stages,
				relative_step=jacobian_relative_step,
				counter=counter,
			)
		jacobian_evaluations += 1
		jacobian: np.ndarray = _reduced_residual_jacobian(stage_jacobians, step)
		try:
			correction: np.ndarray = np.linalg.solve(
				jacobian,
				-current_residual,
			)
		except np.linalg.LinAlgError as exc:
			raise RuntimeError("The HBVM42 Newton Jacobian is singular.") from exc
		if not np.all(np.isfinite(correction)):
			raise RuntimeError("The HBVM42 Newton correction became non-finite.")

		return np.asarray(correction)

	selected_solver = solver or NewtonSolver(absolute_tolerance, relative_tolerance, max_iterations)
	root: _NonlinearResult[_ResidualPayload] = selected_solver.solve(evaluate_flat, coefficients.reshape(-1), correction,
		scale=initial_state, context=f"HBVM42 at t={time:.8g}",
		initial_evaluation=cached,
		norm=lambda r: float(np.linalg.norm(r.reshape(2, state_size), ord=np.inf)),
		backtracking=True, update_returns_correction=True)
	payload = root.payload
	accepted = initial_state + step * payload.projected_coefficients[0]
	return _HBVMStepResult(state=np.asarray(accepted), stage_states=payload.stages,
		iterations=root.iterations,
		residual_norm=float(np.linalg.norm(root.residual.reshape(2, state_size), ord=np.inf)),
		tolerance=tolerance, residual_evaluations=root.residual_evaluations,
		jacobian_evaluations=jacobian_evaluations, vector_field_evaluations=counter.vector_field_evaluations)


@dataclass(slots=True)
class HBVM42(IntegrationMethod[_HBVMStepResult]):
	"""Fourth-order energy-preserving HBVM(4,2).

	The four Gauss--Legendre nodes approximate the Hamiltonian line integral,
	while two shifted-Legendre coefficients form the nonlinear unknown. The
	method has classical order four. For autonomous polynomial Hamiltonians of
	degree at most four it preserves energy up to nonlinear-solver and round-off
	error. Unlike the Gauss method HBVM(2,2), HBVM(4,2) is not generally
	symplectic.
	"""

	jax_mode = "device_resident"

	solver: AlgebraicSolver | None = field(default=None, kw_only=True)

	absolute_tolerance: float = 1e-13
	relative_tolerance: float = 1e-12
	max_iterations: int = 12
	jacobian_method: HBVMJacobianMethod = "auto"
	jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	track_energy: bool = False
	progress: bool = False
	step_observer: StepObserver | None = None

	def __post_init__(self) -> None:
		"""Validate nonlinear controls and normalize scalar fields."""
		if self.solver is not None:
			validate_solver(self.solver, newton_only=True)
			self.absolute_tolerance = self.solver.absolute_tolerance
			self.relative_tolerance = self.solver.relative_tolerance
			self.max_iterations = self.solver.max_iterations

		self.absolute_tolerance = _positive_finite(
			self.absolute_tolerance, "absolute_tolerance", allow_boolean=True,
		)
		self.relative_tolerance = _positive_finite(
			self.relative_tolerance, "relative_tolerance", allow_boolean=True,
		)
		self.max_iterations = _positive_integer(self.max_iterations, "max_iterations")
		self.jacobian_method = _validated_jacobian_method(self.jacobian_method)
		self.jacobian_relative_step = _positive_finite(
			self.jacobian_relative_step, "jacobian_relative_step", allow_boolean=True,
		)

	state_formulation: PhysicalFormulation = field(init=False, repr=False, compare=False)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Bind the HBVM coefficient solve and optional physical observation."""
		self.state_formulation = PhysicalFormulation(problem, request.t_span[0], self.track_energy)
		self.initial_state = self.state_formulation.initial_state
		self.metadata = {'method_order': 4, 'quadrature_stage_count': 4, 'legendre_rank': 2,
							  'jacobian_method': self.jacobian_method}

	def advance(self, time: float, state: np.ndarray, step: float) -> StepResult[_HBVMStepResult]:
		"""Return one accepted coefficient solve and its local work counters."""
		assert self.solver is None or isinstance(self.solver, NewtonSolver)
		result: _HBVMStepResult = _advance_hbvm42(
			self.problem.dynamics,
			time,
			self.state_formulation.physical(state),
			step,
			absolute_tolerance=self.absolute_tolerance,
			relative_tolerance=self.relative_tolerance,
			max_iterations=self.max_iterations,
			jacobian_method=self.jacobian_method,
			jacobian_relative_step=self.jacobian_relative_step, solver=self.solver,
		)
		increment = None
		if self.track_energy:
			increment = step * sum(weight * self.state_formulation.momentum_rate(time + step * node, stage)
			    for node, weight, stage in zip(_HBVM42_NODES, _HBVM42_WEIGHTS, result.stage_states))
		after: np.ndarray = self.state_formulation.finish(state, result.state, time + step, increment)
		return StepResult(after, {
			'nonlinear_iterations': result.iterations,
			'nonlinear_residual_norms': result.residual_norm,
			'nonlinear_tolerances': result.tolerance,
			'residual_evaluations': result.residual_evaluations,
			'residual_evaluations_per_step': result.residual_evaluations,
			'jacobian_evaluations_per_step': result.jacobian_evaluations,
			'vector_field_evaluations_per_step': result.vector_field_evaluations,
		}, result)

	def build_observation(self, info: StepInfo, result: StepResult[_HBVMStepResult]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			internal: np.ndarray = self.state_formulation.pack(candidate, info.time, self.state_formulation.momentum(info.state_before))
			return self.state_formulation.physical(self.advance(info.time, internal, info.duration).state)
		return IntegrationStep(
			dynamics_name=type(self.problem.dynamics).__name__, method_name=type(self).__name__,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=self.state_formulation.physical(info.state_before).copy(),
			state_after=self.state_formulation.physical(result.state).copy(), map_state=map_state, dynamics=self.problem.dynamics,
		)


__all__ = ["HBVM42", "HBVMJacobianMethod"]
