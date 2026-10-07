"""Five-stage, fourth-order singly diagonally implicit Runge--Kutta method."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics import (
	DynamicalSystem,
	HamiltonianSystem,
	GuidingCenterJacobianSystem,
)

from methods._compiled import CompiledFixedMethod
from contracts.step import StepInfo, StepResult
from contracts.result import DiagnosticValue
from formulations.state import PhysicalFormulation
from contracts.observation import IntegrationStep, StepObserver
from contracts.nonlinear import NewtonObserver
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from methods._validation import _positive_finite, _positive_integer
from methods._linear import _solve_particle_systems
from methods._nonlinear import _bind_newton_observer, _solve_newton
from ._jacobians import (
	JacobianMethod as SDIRKJacobianMethod,
	ResolvedJacobianMethod as ResolvedSDIRKJacobianMethod,
	_checked_vector_field,
	_dense_finite_difference_jacobian,
	_checked_analytic_jacobians,
	_resolved_jacobian_method,
)


SDIRK_JACOBIAN_METHODS: tuple[SDIRKJacobianMethod, ...] = (
	"auto",
	"analytic",
	"finite_difference",
)

# Skvortsov's S54b scheme has a common diagonal gamma=1/4.  Its fifth row is
# also the output formula, so the accepted state equals the last stage up to
# the nonlinear solve tolerance.
SDIRK4_TABLEAU_A = np.asarray(
	(
		(1.0 / 4.0, 0.0, 0.0, 0.0, 0.0),
		(-1.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0),
		(1.0 / 8.0, 1.0 / 8.0, 1.0 / 4.0, 0.0, 0.0),
		(-3.0 / 2.0, 3.0 / 4.0, 3.0 / 2.0, 1.0 / 4.0, 0.0),
		(0.0, 1.0 / 6.0, 2.0 / 3.0, -1.0 / 12.0, 1.0 / 4.0),
	),
	dtype=float,
)
SDIRK4_TABLEAU_B = np.asarray(
	(0.0, 1.0 / 6.0, 2.0 / 3.0, -1.0 / 12.0, 1.0 / 4.0),
	dtype=float,
)
SDIRK4_TABLEAU_C = np.sum(SDIRK4_TABLEAU_A, axis=1)
for _coefficient_array in (
	SDIRK4_TABLEAU_A,
	SDIRK4_TABLEAU_B,
	SDIRK4_TABLEAU_C,
):
	_coefficient_array.setflags(write=False)

_SDIRK_DIAGONAL = float(SDIRK4_TABLEAU_A[0, 0])
_STAGE_COUNT = int(SDIRK4_TABLEAU_B.size)


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


def _analytic_newton_correction(
	step: float,
	residual: np.ndarray,
	jacobians: np.ndarray,
) -> np.ndarray:
	"""Solve independent ``2 x 2`` SDIRK Newton systems for every particle."""
	particle_count = jacobians.shape[0]
	matrix = (
		np.broadcast_to(np.eye(2), (particle_count, 2, 2))
		- step * _SDIRK_DIAGONAL * jacobians
	)
	particle_residuals = np.stack(
		(residual[:particle_count], residual[particle_count:]),
		axis=-1,
	)
	particle_corrections = _solve_particle_systems(
		matrix, -particle_residuals,
		singular_message="A per-particle SDIRK Newton matrix is singular.",
	)
	return np.concatenate(
		(particle_corrections[:, 0], particle_corrections[:, 1])
	)


def _solve_stage(
	dynamics: DynamicalSystem,
	stage_time: float,
	right_side: np.ndarray,
	initial_guess: np.ndarray,
	step: float,
	*,
	tolerance: float,
	max_iterations: int,
	jacobian_method: ResolvedSDIRKJacobianMethod,
	jacobian_relative_step: float,
	newton_observer: NewtonObserver | None = None,
	stage_index: int | None = None,
	step_time: float | None = None,
) -> _StageResult:
	"""Solve one diagonal stage with full Newton corrections."""
	def evaluate(stage_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Retain the field paired with this exact diagonal-stage residual."""
		field = _checked_vector_field(dynamics, stage_time, stage_state)
		return stage_state - right_side - step * _SDIRK_DIAGONAL * field, field

	def update(stage_state: np.ndarray, residual: np.ndarray, field: np.ndarray) -> np.ndarray:
		"""Compute the S54b-specific Newton correction in physical coordinates."""
		if jacobian_method == "analytic":
			if not isinstance(dynamics, GuidingCenterJacobianSystem):
				raise TypeError(
					"Analytic SDIRK Jacobians require GuidingCenterJacobianSystem."
				)
			correction = _analytic_newton_correction(
				step,
				residual,
				_checked_analytic_jacobians(dynamics, stage_time, stage_state),
			)
		else:
			jacobian = _dense_finite_difference_jacobian(
				dynamics,
				stage_time,
				stage_state,
				relative_step=jacobian_relative_step,
			)
			matrix = np.eye(stage_state.size) - (
				step * _SDIRK_DIAGONAL * jacobian
			)
			try:
				correction = np.linalg.solve(matrix, -residual)
			except np.linalg.LinAlgError as exc:
				raise RuntimeError("The SDIRK Newton matrix is singular.") from exc
		if not np.all(np.isfinite(correction)):
			raise RuntimeError("The SDIRK Newton correction became non-finite.")
		return np.asarray(stage_state + correction)

	root = _solve_newton(
		evaluate, initial_guess, update,
		tolerance=tolerance, max_iterations=max_iterations,
		context=f"SDIRK4 at stage time t={stage_time:.16g} with h={step:.16g}",
		iteration_observer=_bind_newton_observer(
			newton_observer, time=stage_time if step_time is None else step_time,
			duration=step, tolerance=tolerance, stage_index=stage_index,
		),
	)
	return _StageResult(
		state=root.unknown, field=root.payload, iterations=root.iterations,
		residual_evaluations=root.residual_evaluations,
		residual_norm=float(np.linalg.norm(root.residual, ord=np.inf)),
	)


def _solve_sdirk_step(
	dynamics: DynamicalSystem,
	time: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	jacobian_method: ResolvedSDIRKJacobianMethod,
	jacobian_relative_step: float,
	newton_observer: NewtonObserver | None = None,
) -> _SDIRKStepResult:
	"""Advance one S54b step through five sequential implicit solves."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The SDIRK physical state must be a finite vector.")
	initial_field = _checked_vector_field(dynamics, time, value)
	tolerance = absolute_tolerance + relative_tolerance * max(
		1.0,
		float(np.linalg.norm(value, ord=np.inf)),
	)
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
		result = _solve_stage(
			dynamics,
			time + step * SDIRK4_TABLEAU_C[stage_index],
			right_side,
			initial_guess,
			step,
			tolerance=tolerance,
			max_iterations=max_iterations,
			jacobian_method=jacobian_method,
			jacobian_relative_step=jacobian_relative_step,
			newton_observer=newton_observer, stage_index=stage_index, step_time=time,
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
class SDIRK4(CompiledFixedMethod[_SDIRKStepResult]):
	"""Skvortsov S54b: five-stage, fourth-order non-geometric SDIRK."""

	track_energy: bool = False
	newton_absolute_tolerance: float = 1e-14
	newton_relative_tolerance: float = 1e-13
	newton_max_iterations: int = 20
	newton_jacobian_method: SDIRKJacobianMethod = "auto"
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	progress: bool = False
	step_observer: StepObserver | None = None
	newton_observer: NewtonObserver | None = field(default=None, kw_only=True)

	# Resources owned by one run; excluded from constructor options.
	state_formulation: PhysicalFormulation = field(init=False, repr=False, compare=False)
	dynamics: DynamicalSystem = field(init=False, repr=False, compare=False)
	physical_size: int = field(init=False, repr=False, compare=False)
	resolved_jacobian_method: ResolvedSDIRKJacobianMethod = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Validate nonlinear and differentiation controls before integration."""
		self.newton_absolute_tolerance = _positive_finite(
			self.newton_absolute_tolerance, "newton_absolute_tolerance", allow_boolean=True,
		)
		self.newton_relative_tolerance = _positive_finite(
			self.newton_relative_tolerance, "newton_relative_tolerance", allow_boolean=True,
		)
		self.newton_jacobian_relative_step = _positive_finite(
			self.newton_jacobian_relative_step, "newton_jacobian_relative_step", allow_boolean=True,
		)
		self.newton_max_iterations = _positive_integer(self.newton_max_iterations, "newton_max_iterations")
		if self.newton_jacobian_method not in SDIRK_JACOBIAN_METHODS:
			raise ValueError(
				"`newton_jacobian_method` must be 'auto', 'analytic', or "
				"'finite_difference'."
			)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Bind the nonlinear map, physical observation and energy exporter."""
		self.dynamics = problem.dynamics
		self.state_formulation = PhysicalFormulation(problem, request.t_span[0], self.track_energy)
		physical_initial = problem.initial_state
		self.resolved_jacobian_method = _resolved_jacobian_method(
			self.dynamics,
			self.newton_jacobian_method,
			initial_time=request.t_span[0],
			initial_state=physical_initial,
		)
		self.physical_size = physical_initial.size
		initial_state = self.state_formulation.initial_state
		metadata: dict[str, DiagnosticValue] = {
			'stage_count': _STAGE_COUNT,
			'designed_order': 4,
			'tableau_name': 'Skvortsov S54b',
			'diagonal_coefficient': _SDIRK_DIAGONAL,
			'stiffly_accurate': True,
			'symmetric': False,
			'symplectic': False,
			'nonlinear_solver': 'newton',
			'nonlinear_solves_per_step': _STAGE_COUNT,
			'nonlinear_absolute_tolerance': self.newton_absolute_tolerance,
			'nonlinear_relative_tolerance': self.newton_relative_tolerance,
			'nonlinear_max_iterations': self.newton_max_iterations,
			'newton_jacobian_method': self.resolved_jacobian_method,
			'requested_newton_jacobian_method': self.newton_jacobian_method,
			'newton_jacobian_relative_step': self.newton_jacobian_relative_step,
		}
		self.initial_state = initial_state
		self.metadata = metadata

	def _solve_physical(
		self,
		time: float,
		physical: np.ndarray,
		step: float,
		*, newton_observer: NewtonObserver | None = None,
	) -> _SDIRKStepResult:
		return _solve_sdirk_step(
			self.dynamics,
			time,
			physical,
			step,
			absolute_tolerance=self.newton_absolute_tolerance,
			relative_tolerance=self.newton_relative_tolerance,
			max_iterations=self.newton_max_iterations,
			jacobian_method=self.resolved_jacobian_method,
			jacobian_relative_step=self.newton_jacobian_relative_step,
			newton_observer=newton_observer,
		)

	def advance(self, time: float, value: np.ndarray, step: float) -> StepResult[_SDIRKStepResult]:
		"""Solve one complete physical step and finish its auxiliary state."""
		physical_before = np.asarray(value[:self.physical_size], dtype=float)
		result = self._solve_physical(time, physical_before, step, newton_observer=self.newton_observer)
		statistics: dict[str, np.ndarray | float | int] = {
			'nonlinear_iterations': int(np.sum(result.stage_iterations)),
			'residual_evaluations': int(np.sum(result.stage_residual_evaluations)),
			'nonlinear_residual_norms': float(np.max(result.stage_residual_norms)),
			'nonlinear_tolerances': result.tolerance,
			'stage_nonlinear_iterations': result.stage_iterations,
			'stage_residual_evaluations': result.stage_residual_evaluations,
			'stage_nonlinear_residual_norms': result.stage_residual_norms,
		}
		if not self.track_energy:
			return StepResult(result.state, statistics, result)
		momentum_derivatives = tuple(
			self.state_formulation.momentum_rate(
				time + step * SDIRK4_TABLEAU_C[index], result.stage_states[index],
			)
			for index in range(_STAGE_COUNT)
		)
		increment = step * sum(
			SDIRK4_TABLEAU_B[index] * momentum_derivatives[index]
			for index in range(_STAGE_COUNT)
		)
		after = self.state_formulation.finish(value, result.state, time + step, increment)
		return StepResult(after, statistics, result)

	def build_observation(self, info: StepInfo, step: StepResult[_SDIRKStepResult]) -> IntegrationStep:
		"""Build independent physical snapshots from accepted solve details."""
		result = step.details
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return self._solve_physical(info.time, candidate, info.duration).state
		return IntegrationStep(
			dynamics_name=type(self.dynamics).__name__, method_name=type(self).__name__,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=info.state_before[:self.physical_size].copy(),
			state_after=result.state.copy(), map_state=map_state, dynamics=self.dynamics,
		)


__all__ = [
	"SDIRK4_TABLEAU_A",
	"SDIRK4_TABLEAU_B",
	"SDIRK4_TABLEAU_C",
	"SDIRK_JACOBIAN_METHODS",
	"SDIRK4",
	"SDIRKJacobianMethod",
]
