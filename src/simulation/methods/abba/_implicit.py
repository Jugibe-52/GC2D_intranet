"""Shared configuration and integration contract for implicit ABBA variants."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from dynamics import GuidingCenterJacobianSystem

from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
from ...observation import ImplicitABBAIntegrationStep, StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from .._nonlinear import NonlinearSolver, _validate_nonlinear_solver
from ._projection import (
	_ProjectedStep,
	_checked_vector_field_jacobian,
	_solve_projected_step,
)


def _positive_finite(value: float, name: str) -> float:
	"""Normalize a strictly positive finite solver parameter."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be positive and finite.")
	result = float(value)
	if not np.isfinite(result) or result <= 0:
		raise ValueError(f"`{name}` must be positive and finite.")
	return result


def _positive_integer(value: int, name: str) -> int:
	"""Normalize a strictly positive integer solver parameter."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 1
	):
		raise ValueError(f"`{name}` must be a positive integer.")
	return int(value)


def _integrate_projected_abba(
	problem: InitialValueProblem,
	request: SimulationRequest,
	*,
	method_name: str,
	step_solver: Callable[..., _ProjectedStep],
	solver_formulation: str,
	newton_absolute_tolerance: float,
	newton_relative_tolerance: float,
	newton_max_iterations: int,
	nonlinear_solver: NonlinearSolver,
	progress: bool,
	step_observer: StepObserver | None,
) -> IntegrationData:
	"""Coordinate one projected ABBA run and collect main-step diagnostics."""
	dynamics = problem.dynamics
	if not isinstance(dynamics, GuidingCenterJacobianSystem):
		raise TypeError(f"{method_name} requires GuidingCenterJacobianSystem.")
	if dynamics.state_dimension != 2:
		raise TypeError(f"{method_name} requires planar two-component dynamics.")
	if nonlinear_solver == "newton":
		# Exact Newton requires spatial Hessians; Broyden evaluates only the
		# A-B-B-A projection residual and therefore does not call this capability.
		_checked_vector_field_jacobian(
			dynamics,
			request.t_span[0],
			problem.initial_state,
		)

	iteration_counts: list[int] = []
	residual_evaluation_counts: list[int] = []
	residual_norms: list[float] = []
	tolerance_values: list[float] = []
	multiplier_norms: list[float] = []

	def advance(
		t: float,
		state: np.ndarray,
		step: float,
		step_index: int,
		observe: bool,
	) -> np.ndarray:
		def apply_step(candidate: np.ndarray) -> np.ndarray:
			"""Apply this fixed-time projected ABBA map to one candidate."""
			return step_solver(
				dynamics,
				t,
				candidate,
				step,
				absolute_tolerance=newton_absolute_tolerance,
				relative_tolerance=newton_relative_tolerance,
				max_iterations=newton_max_iterations,
				nonlinear_solver=nonlinear_solver,
			).state

		state_before = np.asarray(state, dtype=float)
		result = step_solver(
			dynamics,
			t,
			state_before,
			step,
			absolute_tolerance=newton_absolute_tolerance,
			relative_tolerance=newton_relative_tolerance,
			max_iterations=newton_max_iterations,
			nonlinear_solver=nonlinear_solver,
		)
		if observe:
			state_scale = max(1.0, float(np.linalg.norm(state_before, ord=np.inf)))
			newton_tolerance = (
				newton_absolute_tolerance
				+ newton_relative_tolerance * state_scale
			)
			multiplier_norm = float(
				np.linalg.norm(result.multiplier, ord=np.inf)
			)
			iteration_counts.append(result.iterations)
			residual_evaluation_counts.append(result.residual_evaluations)
			residual_norms.append(result.residual_norm)
			tolerance_values.append(newton_tolerance)
			multiplier_norms.append(multiplier_norm)
			if step_observer is not None:
				step_observer(
					ImplicitABBAIntegrationStep(
						dynamics_name=type(dynamics).__name__,
						method_name=method_name,
						step_index=step_index,
						time=t + step,
						duration=step,
						state_before=state_before.copy(),
						state_after=result.state.copy(),
						map_state=apply_step,
						formulation_name=solver_formulation,
						start_time=t,
						nonlinear_solver=nonlinear_solver,
						newton_iterations=result.iterations,
						residual_evaluations=result.residual_evaluations,
						newton_residual_norm=result.residual_norm,
						newton_tolerance=newton_tolerance,
						projection_multiplier_norm=multiplier_norm,
						dynamics=dynamics,
						multiplier=result.multiplier.copy(),
						u_initial=result.stages.u_initial.copy(),
						v_initial=result.stages.v_initial.copy(),
						u_first=result.stages.u_first.copy(),
						v_final=result.stages.v_final.copy(),
						u_final=result.stages.u_final.copy(),
					)
				)
		return result.state

	history, step_count = integrate_fixed_grid(
		problem.initial_state,
		request,
		advance,
		progress=bool(progress),
		label=method_name,
	)
	diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
		"step_count": step_count,
		"nonlinear_solver": nonlinear_solver,
		"nonlinear_iterations": np.asarray(iteration_counts, dtype=int),
		"residual_evaluations": np.asarray(
			residual_evaluation_counts,
			dtype=int,
		),
		"nonlinear_residual_norms": np.asarray(residual_norms, dtype=float),
		"nonlinear_tolerances": np.asarray(tolerance_values, dtype=float),
		"nonlinear_absolute_tolerance": newton_absolute_tolerance,
		"nonlinear_relative_tolerance": newton_relative_tolerance,
		"nonlinear_max_iterations": newton_max_iterations,
		"newton_iterations": np.asarray(iteration_counts, dtype=int),
		"newton_residual_norms": np.asarray(residual_norms, dtype=float),
		"projection_multiplier_norms": np.asarray(
			multiplier_norms,
			dtype=float,
		),
		"newton_absolute_tolerance": newton_absolute_tolerance,
		"newton_relative_tolerance": newton_relative_tolerance,
		"newton_max_iterations": newton_max_iterations,
		"projection_solver_formulation": solver_formulation,
	}
	return IntegrationData(
		t=request.output_times,
		states=np.asarray(history),
		diagnostics=diagnostics,
	)


@dataclass(frozen=True, slots=True)
class _ImplicitABBA:
	"""Configure a symmetric projected ABBA nonlinear formulation."""

	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	nonlinear_solver: NonlinearSolver = "newton"
	progress: bool = False
	step_observer: StepObserver | None = None

	_step_solver: ClassVar[Callable[..., _ProjectedStep]] = _solve_projected_step
	_solver_formulation: ClassVar[str] = "reduced_multiplier"

	def __post_init__(self) -> None:
		"""Validate the nonlinear projection solver configuration."""
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
			_positive_integer(
				self.newton_max_iterations,
				"newton_max_iterations",
			),
		)
		object.__setattr__(
			self,
			"nonlinear_solver",
			_validate_nonlinear_solver(self.nonlinear_solver),
		)

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate one GC problem with the selected implicit formulation."""
		return _integrate_projected_abba(
			problem,
			request,
			method_name=type(self).__name__,
			step_solver=type(self)._step_solver,
			solver_formulation=type(self)._solver_formulation,
			newton_absolute_tolerance=self.newton_absolute_tolerance,
			newton_relative_tolerance=self.newton_relative_tolerance,
			newton_max_iterations=self.newton_max_iterations,
			nonlinear_solver=self.nonlinear_solver,
			progress=self.progress,
			step_observer=self.step_observer,
		)


__all__: list[str] = []
