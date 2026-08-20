"""Taylor-like updates driven by exact implicit-ABBA map tangents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias

import numpy as np

from dynamics import GuidingCenterJacobianSystem

from .._fixed import integrate_fixed_grid
from .._result import IntegrationData
from ..problem import InitialValueProblem
from ..request import SimulationRequest
from ._nonlinear import NonlinearSolver, _validate_nonlinear_solver
from ._projected_abba import (
	_checked_vector_field,
	_checked_vector_field_jacobian,
	_positive_finite,
	_positive_integer,
	_projected_step_particle_jacobians,
	_solve_projected_step,
)
from .abba4_implicit_1 import _solve_abba4_step


TangentTaylorBase: TypeAlias = Literal["implicit_abba_1", "abba4_implicit_1"]


@dataclass(frozen=True, slots=True)
class _BaseTangentStep:
	"""One base-map solve and its complete physical tangent blocks."""

	state: np.ndarray
	jacobians: np.ndarray
	iterations: tuple[int, ...]
	residual_evaluations: tuple[int, ...]
	residual_norms: tuple[float, ...]
	tolerances: tuple[float, ...]
	multiplier_norms: tuple[float, ...]


def _solver_tolerance(
	state: np.ndarray,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
) -> float:
	"""Return the effective nonlinear tolerance for one signed base substep."""
	state_scale = max(1.0, float(np.linalg.norm(state, ord=np.inf)))
	return absolute_tolerance + relative_tolerance * state_scale


def _solve_base_step(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	base: TangentTaylorBase,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver,
) -> _BaseTangentStep:
	"""Solve the selected base map and compose its exact physical tangent."""
	if base == "implicit_abba_1":
		projected_result = _solve_projected_step(
			dynamics,
			t,
			state,
			step,
			absolute_tolerance=absolute_tolerance,
			relative_tolerance=relative_tolerance,
			max_iterations=max_iterations,
			nonlinear_solver=nonlinear_solver,
		)
		jacobians = _projected_step_particle_jacobians(
			dynamics,
			t,
			state,
			step,
			projected_result,
		)
		return _BaseTangentStep(
			state=np.asarray(projected_result.state, dtype=float),
			jacobians=jacobians,
			iterations=(projected_result.iterations,),
			residual_evaluations=(projected_result.residual_evaluations,),
			residual_norms=(projected_result.residual_norm,),
			tolerances=(
				_solver_tolerance(
					state,
					absolute_tolerance=absolute_tolerance,
					relative_tolerance=relative_tolerance,
				),
			),
			multiplier_norms=(
				float(np.linalg.norm(projected_result.multiplier, ord=np.inf)),
			),
		)

	abba4_result = _solve_abba4_step(
		dynamics,
		t,
		state,
		step,
		absolute_tolerance=absolute_tolerance,
		relative_tolerance=relative_tolerance,
		max_iterations=max_iterations,
		nonlinear_solver=nonlinear_solver,
	)
	particle_count = state.size // 2
	jacobians = np.broadcast_to(
		np.eye(2),
		(particle_count, 2, 2),
	).copy()
	iterations: list[int] = []
	residual_evaluations: list[int] = []
	residual_norms: list[float] = []
	tolerances: list[float] = []
	multiplier_norms: list[float] = []
	for accepted in abba4_result.substeps:
		factor = _projected_step_particle_jacobians(
			dynamics,
			accepted.start_time,
			accepted.state_before,
			accepted.duration,
			accepted.result,
		)
		# The first executed factor acts first and therefore remains rightmost.
		jacobians = factor @ jacobians
		iterations.append(accepted.result.iterations)
		residual_evaluations.append(accepted.result.residual_evaluations)
		residual_norms.append(accepted.result.residual_norm)
		tolerances.append(
			_solver_tolerance(
				accepted.state_before,
				absolute_tolerance=absolute_tolerance,
				relative_tolerance=relative_tolerance,
			)
		)
		multiplier_norms.append(
			float(np.linalg.norm(accepted.result.multiplier, ord=np.inf))
		)
	return _BaseTangentStep(
		state=np.asarray(abba4_result.state, dtype=float),
		jacobians=jacobians,
		iterations=tuple(iterations),
		residual_evaluations=tuple(residual_evaluations),
		residual_norms=tuple(residual_norms),
		tolerances=tuple(tolerances),
		multiplier_norms=tuple(multiplier_norms),
	)


def _apply_planar_particle_jacobians(
	jacobians: np.ndarray,
	vector: np.ndarray,
) -> np.ndarray:
	"""Apply particle-major ``2 x 2`` blocks to a component-major vector."""
	value = np.asarray(vector, dtype=float)
	particle_count = value.size // 2
	blocks = value.reshape(2, particle_count).T
	result = np.einsum("nij,nj->ni", jacobians, blocks)
	packed = result.T.reshape(-1)
	if packed.shape != value.shape or not np.all(np.isfinite(packed)):
		raise ValueError("The tangent action changed shape or became non-finite.")
	return np.asarray(packed, dtype=float)


def _integrate_tangent_taylor(
	method: _TangentTaylorABBA,
	problem: InitialValueProblem,
	request: SimulationRequest,
) -> IntegrationData:
	"""Advance ``z+h*f+h**2/2*D(Psi_base)*f`` on a fixed grid."""
	dynamics = problem.dynamics
	method_name = type(method).__name__
	if not isinstance(dynamics, GuidingCenterJacobianSystem):
		raise TypeError(f"{method_name} requires GuidingCenterJacobianSystem.")
	if dynamics.state_dimension != 2:
		raise TypeError(f"{method_name} requires planar two-component dynamics.")
	# The update always differentiates the accepted base root, including when the
	# root itself is found by Broyden, so exact field Jacobians are mandatory.
	_checked_vector_field_jacobian(
		dynamics,
		request.t_span[0],
		problem.initial_state,
	)

	iteration_rows: list[tuple[int, ...]] = []
	residual_evaluation_rows: list[tuple[int, ...]] = []
	residual_norm_rows: list[tuple[float, ...]] = []
	tolerance_rows: list[tuple[float, ...]] = []
	multiplier_norm_rows: list[tuple[float, ...]] = []
	base_displacement_norms: list[float] = []
	tangent_action_norms: list[float] = []
	proposed_increment_norms: list[float] = []

	def advance(
		t: float,
		state: np.ndarray,
		step: float,
		_step_index: int,
		observe: bool,
	) -> np.ndarray:
		state_before = np.asarray(state, dtype=float)
		base_step = _solve_base_step(
			dynamics,
			t,
			state_before,
			step,
			base=method._base,
			absolute_tolerance=method.newton_absolute_tolerance,
			relative_tolerance=method.newton_relative_tolerance,
			max_iterations=method.newton_max_iterations,
			nonlinear_solver=method.nonlinear_solver,
		)
		velocity = _checked_vector_field(dynamics, t, state_before)
		tangent_action = _apply_planar_particle_jacobians(
			base_step.jacobians,
			velocity,
		)
		increment = step * velocity + 0.5 * step**2 * tangent_action
		state_after = state_before + increment
		if not np.all(np.isfinite(state_after)):
			raise ValueError(f"{method_name} produced a non-finite physical state.")
		if observe:
			iteration_rows.append(base_step.iterations)
			residual_evaluation_rows.append(base_step.residual_evaluations)
			residual_norm_rows.append(base_step.residual_norms)
			tolerance_rows.append(base_step.tolerances)
			multiplier_norm_rows.append(base_step.multiplier_norms)
			base_displacement_norms.append(
				float(np.linalg.norm(base_step.state - state_before))
			)
			tangent_action_norms.append(float(np.linalg.norm(tangent_action)))
			proposed_increment_norms.append(float(np.linalg.norm(increment)))
		return np.asarray(state_after, dtype=float)

	history, step_count = integrate_fixed_grid(
		problem.initial_state,
		request,
		advance,
		progress=method.progress,
		label=method_name,
	)
	iterations = np.asarray(iteration_rows, dtype=int)
	residual_evaluations = np.asarray(residual_evaluation_rows, dtype=int)
	residual_norms = np.asarray(residual_norm_rows, dtype=float)
	tolerances = np.asarray(tolerance_rows, dtype=float)
	multiplier_norms = np.asarray(multiplier_norm_rows, dtype=float)
	base_method_name = (
		"ImplicitABBA1"
		if method._base == "implicit_abba_1"
		else "ABBA4Implicit1"
	)
	base_formulation = (
		"implicit_1_reduced_equation_11"
		if method._base == "implicit_abba_1"
		else "abba4_implicit_1_triple_jump"
	)
	return IntegrationData(
		t=request.output_times,
		states=np.asarray(history),
		diagnostics={
			"step_count": step_count,
			"base_method_name": base_method_name,
			"base_projection_solver_formulation": base_formulation,
			"update_formula": "z+h*f+h^2/2*Dpsi_base*f",
			"tangent_source": "exact_implicit_function",
			"base_substeps_per_step": iterations.shape[1],
			"nonlinear_solver": method.nonlinear_solver,
			"nonlinear_iterations": np.sum(iterations, axis=1),
			"residual_evaluations": np.sum(residual_evaluations, axis=1),
			"nonlinear_residual_norms": np.max(residual_norms, axis=1),
			"nonlinear_tolerances": np.max(tolerances, axis=1),
			"projection_multiplier_norms": np.max(multiplier_norms, axis=1),
			"base_substep_nonlinear_iterations": iterations,
			"base_substep_residual_evaluations": residual_evaluations,
			"base_substep_nonlinear_residual_norms": residual_norms,
			"base_substep_nonlinear_tolerances": tolerances,
			"base_substep_projection_multiplier_norms": multiplier_norms,
			"base_map_displacement_norms": np.asarray(
				base_displacement_norms,
				dtype=float,
			),
			"tangent_action_norms": np.asarray(tangent_action_norms, dtype=float),
			"proposed_increment_norms": np.asarray(
				proposed_increment_norms,
				dtype=float,
			),
			"nonlinear_absolute_tolerance": method.newton_absolute_tolerance,
			"nonlinear_relative_tolerance": method.newton_relative_tolerance,
			"nonlinear_max_iterations": method.newton_max_iterations,
		},
	)


@dataclass(frozen=True, slots=True)
class _TangentTaylorABBA:
	"""Shared nonlinear-solver configuration for both tangent-Taylor methods."""

	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	nonlinear_solver: NonlinearSolver = "newton"
	progress: bool = False

	_base: ClassVar[TangentTaylorBase] = "implicit_abba_1"

	def __post_init__(self) -> None:
		"""Validate every parameter used by the internal base-map solves."""
		object.__setattr__(
			self,
			"newton_absolute_tolerance",
			_positive_finite(
				self.newton_absolute_tolerance,
				"newton_absolute_tolerance",
			),
		)
		object.__setattr__(
			self,
			"newton_relative_tolerance",
			_positive_finite(
				self.newton_relative_tolerance,
				"newton_relative_tolerance",
			),
		)
		object.__setattr__(
			self,
			"newton_max_iterations",
			_positive_integer(self.newton_max_iterations, "newton_max_iterations"),
		)
		object.__setattr__(
			self,
			"nonlinear_solver",
			_validate_nonlinear_solver(self.nonlinear_solver),
		)
		object.__setattr__(self, "progress", bool(self.progress))

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate the physical state with the tangent-Taylor update."""
		return _integrate_tangent_taylor(self, problem, request)


@dataclass(frozen=True, slots=True)
class ImplicitABBA1TangentTaylor(_TangentTaylorABBA):
	"""Use the exact `ImplicitABBA1` tangent in the proposed update formula."""

	_base: ClassVar[TangentTaylorBase] = "implicit_abba_1"


@dataclass(frozen=True, slots=True)
class ABBA4Implicit1TangentTaylor(_TangentTaylorABBA):
	"""Use the exact three-factor `ABBA4Implicit1` tangent in the update."""

	_base: ClassVar[TangentTaylorBase] = "abba4_implicit_1"


__all__ = [
	"ABBA4Implicit1TangentTaylor",
	"ImplicitABBA1TangentTaylor",
]
