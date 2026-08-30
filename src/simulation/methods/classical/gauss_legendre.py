"""Implicit two-stage, fourth-order Gauss--Legendre collocation method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from dynamics import (
	DynamicalSystem,
	ExtendedHamiltonianSystem,
	GuidingCenterJacobianSystem,
)

from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
from ...formulations.base import generalized_energy_error
from ...observation import GaussLegendre4IntegrationStep, StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest


GaussJacobianMethod: TypeAlias = Literal[
	"auto",
	"analytic",
	"finite_difference",
]
GAUSS_JACOBIAN_METHODS: tuple[GaussJacobianMethod, ...] = (
	"auto",
	"analytic",
	"finite_difference",
)
ResolvedGaussJacobianMethod: TypeAlias = Literal["analytic", "finite_difference"]

_ROOT_THREE_OVER_SIX = float(np.sqrt(3.0) / 6.0)
_GAUSS_NODES = np.asarray(
	(0.5 - _ROOT_THREE_OVER_SIX, 0.5 + _ROOT_THREE_OVER_SIX),
	dtype=float,
)
_GAUSS_MATRIX = np.asarray(
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


def _checked_vector_field(
	dynamics: DynamicalSystem,
	time: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate one finite vector field without allowing a shape change."""
	result = np.asarray(dynamics.vector_field(time, state), dtype=float)
	if result.shape != state.shape or not np.all(np.isfinite(result)):
		raise ValueError("The vector field must be finite and preserve state shape.")
	return result


def _dense_finite_difference_jacobian(
	dynamics: DynamicalSystem,
	time: float,
	state: np.ndarray,
	*,
	relative_step: float,
) -> np.ndarray:
	"""Differentiate a vector field with scale-aware centered differences."""
	dimension = state.size
	jacobian = np.empty((dimension, dimension), dtype=float)
	for column in range(dimension):
		increment = relative_step * max(1.0, abs(float(state[column])))
		perturbation = np.zeros_like(state)
		perturbation[column] = increment
		forward = _checked_vector_field(dynamics, time, state + perturbation)
		backward = _checked_vector_field(dynamics, time, state - perturbation)
		jacobian[:, column] = (forward - backward) / (2.0 * increment)
	if not np.all(np.isfinite(jacobian)):
		raise ValueError("The finite-difference vector-field Jacobian is non-finite.")
	return jacobian


def _checked_analytic_jacobians(
	dynamics: GuidingCenterJacobianSystem,
	time: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Return one exact finite ``2 x 2`` block per independent GC particle."""
	if state.size % 2:
		raise ValueError("Analytic GC Jacobians require an even physical state size.")
	particle_count = state.size // 2
	result = np.asarray(
		dynamics.particle_vector_field_jacobians(time, state),
		dtype=float,
	)
	expected_shape = (particle_count, 2, 2)
	if result.shape != expected_shape or not np.all(np.isfinite(result)):
		raise ValueError(
			"The analytic GC Jacobian must be finite and have shape "
			f"{expected_shape}."
		)
	return result


def _stage_evaluation(
	dynamics: DynamicalSystem,
	time: float,
	state: np.ndarray,
	step: float,
	stage_states: tuple[np.ndarray, np.ndarray],
) -> _StageEvaluation:
	"""Evaluate both fields and coupled collocation residuals."""
	stage_times = time + step * _GAUSS_NODES
	fields = tuple(
		_checked_vector_field(dynamics, float(stage_time), stage_state)
		for stage_time, stage_state in zip(
			stage_times,
			stage_states,
			strict=True,
		)
	)
	residuals = tuple(
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
	dynamics: DynamicalSystem,
	time: float,
	step: float,
	stage_states: tuple[np.ndarray, np.ndarray],
	*,
	jacobian_method: ResolvedGaussJacobianMethod,
	jacobian_relative_step: float,
) -> tuple[np.ndarray, np.ndarray]:
	"""Evaluate Newton Jacobians only after a residual fails convergence."""
	stage_times = time + step * _GAUSS_NODES
	if jacobian_method == "analytic":
		if not isinstance(dynamics, GuidingCenterJacobianSystem):
			raise TypeError(
				"Analytic Gauss Jacobians require GuidingCenterJacobianSystem."
			)
		jacobians = tuple(
			_checked_analytic_jacobians(
				dynamics,
				float(stage_time),
				stage_state,
			)
			for stage_time, stage_state in zip(
				stage_times,
				stage_states,
				strict=True,
			)
		)
	else:
		jacobians = tuple(
			_dense_finite_difference_jacobian(
				dynamics,
				float(stage_time),
				stage_state,
				relative_step=jacobian_relative_step,
			)
			for stage_time, stage_state in zip(
				stage_times,
				stage_states,
				strict=True,
			)
		)
	return jacobians[0], jacobians[1]


def _particle_vectors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
	"""Gather two component-major planar vectors into stage-major particles."""
	particle_count = first.size // 2
	return np.stack(
		(
			first[:particle_count],
			first[particle_count:],
			second[:particle_count],
			second[particle_count:],
		),
		axis=-1,
	)


def _component_major_stage_corrections(
	corrections: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Repack batched ``[stage1 x,y, stage2 x,y]`` corrections."""
	return (
		np.concatenate((corrections[:, 0], corrections[:, 1])),
		np.concatenate((corrections[:, 2], corrections[:, 3])),
	)


def _analytic_newton_correction(
	step: float,
	evaluation: _StageEvaluation,
	jacobians: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
	"""Solve independent ``4 x 4`` Gauss Newton systems for every particle."""
	blocks_1, blocks_2 = jacobians
	particle_count = blocks_1.shape[0]
	identity = np.broadcast_to(np.eye(2), (particle_count, 2, 2))
	top = np.concatenate(
		(
			identity - step * _GAUSS_MATRIX[0, 0] * blocks_1,
			-step * _GAUSS_MATRIX[0, 1] * blocks_2,
		),
		axis=-1,
	)
	bottom = np.concatenate(
		(
			-step * _GAUSS_MATRIX[1, 0] * blocks_1,
			identity - step * _GAUSS_MATRIX[1, 1] * blocks_2,
		),
		axis=-1,
	)
	matrix = np.concatenate((top, bottom), axis=-2)
	residual = _particle_vectors(*evaluation.residuals)
	try:
		corrections = np.linalg.solve(matrix, -residual)
	except np.linalg.LinAlgError as exc:
		raise RuntimeError(
			"The per-particle Gauss Newton matrix is singular."
		) from exc
	return _component_major_stage_corrections(corrections)


def _dense_newton_correction(
	step: float,
	evaluation: _StageEvaluation,
	jacobians: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
	"""Solve the generic dense coupled-stage Newton equation."""
	identity = np.eye(evaluation.residuals[0].size)
	first_jacobian, second_jacobian = jacobians
	matrix = np.block(
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
	residual = np.concatenate(evaluation.residuals)
	try:
		correction = np.linalg.solve(matrix, -residual)
	except np.linalg.LinAlgError as exc:
		raise RuntimeError("The Gauss Newton matrix is singular.") from exc
	dimension = identity.shape[0]
	return correction[:dimension], correction[dimension:]


def _solve_gauss_step(
	dynamics: DynamicalSystem,
	time: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	jacobian_method: ResolvedGaussJacobianMethod,
	jacobian_relative_step: float,
) -> _GaussStepResult:
	"""Solve the two coupled collocation stages with full Newton corrections."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The Gauss physical state must be a finite vector.")
	initial_field = _checked_vector_field(dynamics, time, value)
	stage_states = (
		value + step * _GAUSS_NODES[0] * initial_field,
		value + step * _GAUSS_NODES[1] * initial_field,
	)
	tolerance = absolute_tolerance + relative_tolerance * max(
		1.0,
		float(np.linalg.norm(value, ord=np.inf)),
	)
	evaluation = _stage_evaluation(
		dynamics,
		time,
		value,
		step,
		stage_states,
	)
	residual_evaluations = 1

	for iteration in range(max_iterations + 1):
		residual_norm = max(
			float(np.linalg.norm(residual, ord=np.inf))
			for residual in evaluation.residuals
		)
		if residual_norm <= tolerance:
			state_after = value + step * 0.5 * (
				evaluation.fields[0] + evaluation.fields[1]
			)
			if not np.all(np.isfinite(state_after)):
				raise RuntimeError("The converged Gauss state is non-finite.")
			return _GaussStepResult(
				state=np.asarray(state_after),
				stage_states=(stage_states[0].copy(), stage_states[1].copy()),
				iterations=iteration,
				residual_evaluations=residual_evaluations,
				residual_norm=residual_norm,
			)
		if iteration == max_iterations:
			break
		jacobians = _stage_jacobians(
			dynamics,
			time,
			step,
			stage_states,
			jacobian_method=jacobian_method,
			jacobian_relative_step=jacobian_relative_step,
		)
		if jacobian_method == "analytic":
			assert isinstance(dynamics, GuidingCenterJacobianSystem)
			corrections = _analytic_newton_correction(
				step,
				evaluation,
				jacobians,
			)
		else:
			corrections = _dense_newton_correction(step, evaluation, jacobians)
		if not all(np.all(np.isfinite(value)) for value in corrections):
			raise RuntimeError("The Gauss Newton correction became non-finite.")
		stage_states = (
			stage_states[0] + corrections[0],
			stage_states[1] + corrections[1],
		)
		evaluation = _stage_evaluation(
			dynamics,
			time,
			value,
			step,
			stage_states,
		)
		residual_evaluations += 1

	raise RuntimeError(
		"GaussLegendre4 Newton iteration did not converge at "
		f"t={time:.16g} with h={step:.16g}: residual norm "
		f"{residual_norm:.3e} exceeds {tolerance:.3e} after "
		f"{max_iterations} corrections."
	)


def _resolved_jacobian_method(
	dynamics: DynamicalSystem,
	requested: GaussJacobianMethod,
	*,
	initial_time: float,
	initial_state: np.ndarray,
) -> ResolvedGaussJacobianMethod:
	"""Resolve automatic differentiation from an exercised GC capability."""
	if requested == "auto":
		if (
			not isinstance(dynamics, GuidingCenterJacobianSystem)
			or dynamics.state_dimension != 2
		):
			return "finite_difference"
		try:
			_checked_analytic_jacobians(dynamics, initial_time, initial_state)
		except (TypeError, ValueError, NotImplementedError):
			return "finite_difference"
		return "analytic"
	if requested == "analytic" and (
		not isinstance(dynamics, GuidingCenterJacobianSystem)
		or dynamics.state_dimension != 2
	):
		raise TypeError(
			"`newton_jacobian_method='analytic'` requires "
			"planar GuidingCenterJacobianSystem dynamics."
		)
	return requested


@dataclass(frozen=True, slots=True)
class GaussLegendre4:
	"""Two-stage, fourth-order symmetric Gauss--Legendre Runge--Kutta method."""

	track_energy: bool = False
	newton_absolute_tolerance: float = 1e-14
	newton_relative_tolerance: float = 1e-13
	newton_max_iterations: int = 20
	newton_jacobian_method: GaussJacobianMethod = "auto"
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	progress: bool = False
	step_observer: StepObserver | None = None

	def __post_init__(self) -> None:
		"""Validate nonlinear and differentiation controls before integration."""
		for name in (
			"newton_absolute_tolerance",
			"newton_relative_tolerance",
			"newton_jacobian_relative_step",
		):
			value = float(getattr(self, name))
			if not np.isfinite(value) or value <= 0.0:
				raise ValueError(f"`{name}` must be positive and finite.")
			object.__setattr__(self, name, value)
		if (
			isinstance(self.newton_max_iterations, (bool, np.bool_))
			or not isinstance(self.newton_max_iterations, (int, np.integer))
			or self.newton_max_iterations < 1
		):
			raise ValueError("`newton_max_iterations` must be a positive integer.")
		object.__setattr__(self, "newton_max_iterations", int(self.newton_max_iterations))
		if self.newton_jacobian_method not in GAUSS_JACOBIAN_METHODS:
			raise ValueError(
				"`newton_jacobian_method` must be 'auto', 'analytic', or "
				"'finite_difference'."
			)

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate one physical ODE and optionally its conjugate momentum."""
		dynamics = problem.dynamics
		if not isinstance(dynamics, DynamicalSystem):
			raise TypeError("GaussLegendre4 requires DynamicalSystem.")
		if self.track_energy and not isinstance(
			dynamics,
			ExtendedHamiltonianSystem,
		):
			raise TypeError("Energy tracking requires ExtendedHamiltonianSystem.")
		physical_initial = problem.initial_state
		jacobian_method = _resolved_jacobian_method(
			dynamics,
			self.newton_jacobian_method,
			initial_time=request.t_span[0],
			initial_state=physical_initial,
		)
		physical_size = physical_initial.size
		particle_count = problem.particle_count
		initial_state = (
			physical_initial
			if not self.track_energy
			else np.concatenate((physical_initial, np.zeros(particle_count)))
		)
		iteration_counts: list[int] = []
		residual_evaluation_counts: list[int] = []
		residual_norms: list[float] = []
		tolerances: list[float] = []

		def solve_physical(
			time: float,
			physical: np.ndarray,
			step: float,
		) -> _GaussStepResult:
			return _solve_gauss_step(
				dynamics,
				time,
				physical,
				step,
				absolute_tolerance=self.newton_absolute_tolerance,
				relative_tolerance=self.newton_relative_tolerance,
				max_iterations=self.newton_max_iterations,
				jacobian_method=jacobian_method,
				jacobian_relative_step=self.newton_jacobian_relative_step,
			)

		def advance(
			time: float,
			value: np.ndarray,
			step: float,
			step_index: int,
			observe: bool,
		) -> np.ndarray:
			physical_before = np.asarray(value[:physical_size], dtype=float)
			result = solve_physical(time, physical_before, step)
			if observe:
				tolerance = self.newton_absolute_tolerance + (
					self.newton_relative_tolerance
					* max(1.0, float(np.linalg.norm(physical_before, ord=np.inf)))
				)
				iteration_counts.append(result.iterations)
				residual_evaluation_counts.append(result.residual_evaluations)
				residual_norms.append(result.residual_norm)
				tolerances.append(tolerance)
				if self.step_observer is not None:
					def map_state(candidate: np.ndarray) -> np.ndarray:
						"""Apply this fixed-time Gauss map to a physical candidate."""
						return solve_physical(time, candidate, step).state

					self.step_observer(
						GaussLegendre4IntegrationStep(
							dynamics_name=type(dynamics).__name__,
							method_name=type(self).__name__,
							step_index=step_index,
							start_time=time,
							time=time + step,
							duration=step,
							state_before=physical_before.copy(),
							state_after=result.state.copy(),
							map_state=map_state,
							dynamics=dynamics,
							first_stage_time=time + step * _GAUSS_NODES[0],
							second_stage_time=time + step * _GAUSS_NODES[1],
							newton_iterations=result.iterations,
							residual_evaluations=result.residual_evaluations,
							newton_residual_norm=result.residual_norm,
							newton_tolerance=tolerance,
							first_stage_state=result.stage_states[0].copy(),
							second_stage_state=result.stage_states[1].copy(),
						)
					)
			if not self.track_energy:
				return result.state
			assert isinstance(dynamics, ExtendedHamiltonianSystem)
			momentum_before = np.asarray(value[physical_size:], dtype=float)
			momentum_derivatives = tuple(
				np.asarray(
					dynamics.extended_momentum_derivative(
						time + step * _GAUSS_NODES[index],
						result.stage_states[index],
					),
					dtype=float,
				)
				for index in range(2)
			)
			if any(
				derivative.shape != momentum_before.shape
				or not np.all(np.isfinite(derivative))
				for derivative in momentum_derivatives
			):
				raise ValueError(
					"The extended-momentum derivative must be finite and have "
					"one value per particle."
				)
			momentum_after = momentum_before + step * 0.5 * (
				momentum_derivatives[0] + momentum_derivatives[1]
			)
			return np.concatenate((result.state, momentum_after))

		history, step_count = integrate_fixed_grid(
			initial_state,
			request,
			advance,
			progress=bool(self.progress),
			label=type(self).__name__,
		)
		states = np.asarray(history[:physical_size])
		diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
			"step_count": step_count,
			"stage_count": 2,
			"designed_order": 4,
			"nonlinear_solver": "newton",
			"nonlinear_solves_per_step": 1,
			"nonlinear_iterations": np.asarray(iteration_counts, dtype=int),
			"residual_evaluations": np.asarray(
				residual_evaluation_counts,
				dtype=int,
			),
			"nonlinear_residual_norms": np.asarray(residual_norms, dtype=float),
			"nonlinear_tolerances": np.asarray(tolerances, dtype=float),
			"nonlinear_absolute_tolerance": self.newton_absolute_tolerance,
			"nonlinear_relative_tolerance": self.newton_relative_tolerance,
			"nonlinear_max_iterations": self.newton_max_iterations,
			"newton_iterations": np.asarray(iteration_counts, dtype=int),
			"newton_residual_norms": np.asarray(residual_norms, dtype=float),
			"newton_absolute_tolerance": self.newton_absolute_tolerance,
			"newton_relative_tolerance": self.newton_relative_tolerance,
			"newton_max_iterations": self.newton_max_iterations,
			"newton_jacobian_method": jacobian_method,
			"requested_newton_jacobian_method": self.newton_jacobian_method,
			"newton_jacobian_relative_step": self.newton_jacobian_relative_step,
		}
		if self.track_energy:
			momentum = np.asarray(history[physical_size:])
			diagnostics["extended_momentum"] = momentum
			diagnostics["energy_error"] = generalized_energy_error(
				request.output_times,
				states,
				momentum,
				dynamics,
			)
		return IntegrationData(
			t=request.output_times,
			states=states,
			diagnostics=diagnostics,
		)


__all__ = [
	"GAUSS_JACOBIAN_METHODS",
	"GaussJacobianMethod",
	"GaussLegendre4",
]
