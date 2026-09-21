"""Fourth-order HBVM(4,2) with a reduced Legendre-stage solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from dynamics import GuidingCenterJacobianSystem, HamiltonianSystem

from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
from ...problem import InitialValueProblem
from ...request import SimulationRequest


HBVMJacobianMethod: TypeAlias = Literal["auto", "analytic", "finite_difference"]

# Four-point Gauss--Legendre quadrature on [0, 1].  HBVM(4,2) uses the first
# two orthonormal shifted Legendre polynomials; the resulting Runge--Kutta
# matrix has rank two even though four quadrature stages sample the line integral.
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(4)
_HBVM42_NODES = np.asarray((_GAUSS_NODES + 1.0) / 2.0, dtype=float)
_HBVM42_WEIGHTS = np.asarray(_GAUSS_WEIGHTS / 2.0, dtype=float)
_HBVM42_LEGENDRE = np.column_stack(
	(
		np.ones(4, dtype=float),
		np.sqrt(3.0) * (2.0 * _HBVM42_NODES - 1.0),
	)
)
_HBVM42_INTEGRALS = np.column_stack(
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


def _evaluate_vector_field(
	dynamics: object,
	time: float,
	state: np.ndarray,
	counter: _EvaluationCounter,
) -> np.ndarray:
	"""Evaluate one shape-preserving finite vector field and update its counter."""
	evaluate = getattr(dynamics, "vector_field", None)
	if not callable(evaluate):
		raise TypeError("HBVM42 requires dynamics with a callable vector field.")
	value = np.asarray(evaluate(time, state), dtype=float)
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
	blocks = np.asarray(particle_blocks, dtype=float)
	if blocks.shape != (particle_count, 2, 2) or not np.all(np.isfinite(blocks)):
		raise ValueError(
			"Particle vector-field Jacobians must have shape (particle_count, 2, 2)."
		)
	result = np.zeros((state_size, state_size), dtype=float)
	particles = np.arange(particle_count)
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
		jacobian = np.empty((state_size, state_size), dtype=float)
		for column in range(state_size):
			increment = relative_step * max(1.0, abs(float(stage[column])))
			plus = stage.copy()
			minus = stage.copy()
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
	identity = np.eye(state_size)
	blocks: list[list[np.ndarray]] = []
	for legendre_index in range(2):
		row: list[np.ndarray] = []
		for integral_index in range(2):
			weighted = sum(
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
) -> _HBVMStepResult:
	"""Solve the two-coefficient HBVM(4,2) equation for one complete step."""
	initial_state = np.asarray(state, dtype=float)
	state_size = initial_state.size
	counter = _EvaluationCounter()
	stage_times = time + step * _HBVM42_NODES

	# A first-order stage predictor supplies a substantially better Legendre
	# initial guess than (f(y_n), 0) while retaining a small fixed setup cost.
	initial_field = _evaluate_vector_field(dynamics, time, initial_state, counter)
	predicted_stages = initial_state + step * np.outer(
		_HBVM42_NODES,
		initial_field,
	)
	predicted_fields = np.stack(
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
		fields = np.stack(
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

	current_residual, payload = residual(coefficients)
	residual_norm = float(np.linalg.norm(current_residual, ord=np.inf))
	selected_jacobian_method: HBVMJacobianMethod = jacobian_method
	if selected_jacobian_method == "auto":
		selected_jacobian_method = (
			"analytic"
			if isinstance(dynamics, GuidingCenterJacobianSystem)
			else "finite_difference"
		)
	if selected_jacobian_method == "analytic" and not isinstance(
		dynamics,
		GuidingCenterJacobianSystem,
	):
		raise TypeError(
			"Analytic HBVM Jacobians require GuidingCenterJacobianSystem dynamics."
		)

	for iteration in range(max_iterations + 1):
		if residual_norm <= tolerance:
			# The quadrature update is the Runge--Kutta definition. At convergence
			# it equals y_n + h*gamma_0 up to the accepted nonlinear residual.
			accepted = initial_state + step * payload.projected_coefficients[0]
			return _HBVMStepResult(
				state=np.asarray(accepted),
				iterations=iteration,
				residual_norm=residual_norm,
				tolerance=tolerance,
				residual_evaluations=residual_evaluations,
				jacobian_evaluations=jacobian_evaluations,
				vector_field_evaluations=counter.vector_field_evaluations,
			)
		if iteration == max_iterations:
			break

		if selected_jacobian_method == "analytic":
			assert isinstance(dynamics, GuidingCenterJacobianSystem)
			stage_jacobians = _analytic_stage_jacobians(
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
		jacobian = _reduced_residual_jacobian(stage_jacobians, step)
		try:
			correction = np.linalg.solve(
				jacobian,
				-current_residual.reshape(-1),
			).reshape(2, state_size)
		except np.linalg.LinAlgError as exc:
			raise RuntimeError("The HBVM42 Newton Jacobian is singular.") from exc
		if not np.all(np.isfinite(correction)):
			raise RuntimeError("The HBVM42 Newton correction became non-finite.")

		# Backtracking is only activated when a full Newton step increases the
		# residual. It makes coarse-step notebook sweeps robust without changing
		# the local quadratic convergence near the root.
		damping = 1.0
		while True:
			candidate = coefficients + damping * correction
			next_residual, next_payload = residual(candidate)
			next_norm = float(np.linalg.norm(next_residual, ord=np.inf))
			if next_norm < residual_norm or damping <= 1.0 / 128.0:
				break
			damping /= 2.0
		coefficients = candidate
		current_residual = next_residual
		payload = next_payload
		residual_norm = next_norm

	raise RuntimeError(
		f"HBVM42 Newton solve did not converge at t={time:.8g}: residual norm "
		f"{residual_norm:.3e} exceeds {tolerance:.3e} after "
		f"{max_iterations} iterations."
	)


@dataclass(frozen=True, slots=True)
class HBVM42:
	"""Fourth-order energy-preserving HBVM(4,2).

	The four Gauss--Legendre nodes approximate the Hamiltonian line integral,
	while two shifted-Legendre coefficients form the nonlinear unknown. The
	method has classical order four. For autonomous polynomial Hamiltonians of
	degree at most four it preserves energy up to nonlinear-solver and round-off
	error. Unlike the Gauss method HBVM(2,2), HBVM(4,2) is not generally
	symplectic.
	"""

	absolute_tolerance: float = 1e-13
	relative_tolerance: float = 1e-12
	max_iterations: int = 12
	jacobian_method: HBVMJacobianMethod = "auto"
	jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	track_energy: bool = False
	progress: bool = False

	def __post_init__(self) -> None:
		"""Validate nonlinear controls and normalize scalar fields."""
		for name in ("absolute_tolerance", "relative_tolerance"):
			value = float(getattr(self, name))
			if not np.isfinite(value) or value <= 0.0:
				raise ValueError(f"`{name}` must be positive and finite.")
			object.__setattr__(self, name, value)
		if (
			isinstance(self.max_iterations, (bool, np.bool_))
			or not isinstance(self.max_iterations, (int, np.integer))
			or self.max_iterations < 1
		):
			raise ValueError("`max_iterations` must be a positive integer.")
		object.__setattr__(self, "max_iterations", int(self.max_iterations))
		object.__setattr__(
			self,
			"jacobian_method",
			_validated_jacobian_method(self.jacobian_method),
		)
		relative_step = float(self.jacobian_relative_step)
		if not np.isfinite(relative_step) or relative_step <= 0.0:
			raise ValueError("`jacobian_relative_step` must be positive and finite.")
		object.__setattr__(self, "jacobian_relative_step", relative_step)

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate a physical problem on the shared fixed-step grid."""
		if self.track_energy and not isinstance(problem.dynamics, HamiltonianSystem):
			raise TypeError("HBVM42 energy tracking requires HamiltonianSystem dynamics.")
		iterations: list[int] = []
		residual_norms: list[float] = []
		tolerances: list[float] = []
		residual_evaluations: list[int] = []
		jacobian_evaluations: list[int] = []
		field_evaluations: list[int] = []

		def advance(
			time: float,
			state: np.ndarray,
			step: float,
			step_index: int,
			observe: bool,
		) -> np.ndarray:
			"""Advance once and retain work only for main-grid steps."""
			del step_index
			result = _advance_hbvm42(
				problem.dynamics,
				time,
				state,
				step,
				absolute_tolerance=self.absolute_tolerance,
				relative_tolerance=self.relative_tolerance,
				max_iterations=self.max_iterations,
				jacobian_method=self.jacobian_method,
				jacobian_relative_step=self.jacobian_relative_step,
			)
			if observe:
				iterations.append(result.iterations)
				residual_norms.append(result.residual_norm)
				tolerances.append(result.tolerance)
				residual_evaluations.append(result.residual_evaluations)
				jacobian_evaluations.append(result.jacobian_evaluations)
				field_evaluations.append(result.vector_field_evaluations)
			return result.state

		states, step_count = integrate_fixed_grid(
			problem.initial_state,
			request,
			advance,
			progress=bool(self.progress),
			label=type(self).__name__,
		)
		diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
			"step_count": step_count,
			"method_order": 4,
			"quadrature_stage_count": 4,
			"legendre_rank": 2,
			"nonlinear_iterations": np.asarray(iterations, dtype=int),
			"nonlinear_residual_norms": np.asarray(residual_norms, dtype=float),
			"nonlinear_tolerances": np.asarray(tolerances, dtype=float),
			"residual_evaluations_per_step": np.asarray(
				residual_evaluations,
				dtype=int,
			),
			"jacobian_evaluations_per_step": np.asarray(
				jacobian_evaluations,
				dtype=int,
			),
			"vector_field_evaluations_per_step": np.asarray(
				field_evaluations,
				dtype=int,
			),
			"jacobian_method": self.jacobian_method,
		}
		if self.track_energy:
			assert isinstance(problem.dynamics, HamiltonianSystem)
			energies = np.asarray(
				problem.dynamics.hamiltonian(request.output_times, states),
				dtype=float,
			)
			if energies.ndim == 1:
				energies = energies[np.newaxis, :]
			if energies.shape[-1] != request.output_times.size:
				raise ValueError(
					"Hamiltonian values must retain the saved-time dimension."
				)
			energy_drift = energies - energies[:, :1]
			diagnostics["hamiltonian"] = energies
			diagnostics["energy_drift"] = energy_drift
			diagnostics["energy_error"] = float(np.max(np.abs(energy_drift)))
		return IntegrationData(
			t=request.output_times,
			states=np.asarray(states),
			diagnostics=diagnostics,
		)


__all__ = ["HBVM42", "HBVMJacobianMethod"]
