"""Shared configuration and integration contract for implicit ABBA variants."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dynamics import ExtendedHamiltonianSystem, GuidingCenterJacobianSystem

from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
from ...observation import ABBA2ImplicitIntegrationStep, StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from .._nonlinear import NonlinearSolver, _validate_nonlinear_solver
from ._configuration import (
	ABBA_PROJECTION_FORMULATIONS,
	ProjectionFormulation,
	StateExtension,
	_resolved_track_energy,
	_state_dimension_diagnostics,
	_validate_projection_formulation,
	_validate_state_extension,
)
from ._energy import (
	_conjugate_momentum_increment_from_stages,
	_energy_tracking_diagnostics,
	_energy_tracking_initial_state,
	_pack_energy_tracking_state,
	_unpack_energy_tracking_state,
	_validate_energy_tracking,
)
from ._projection_common import (
	_ProjectedStep,
	_checked_vector_field_jacobian,
)
from ._projection_reduced import (
	_solve_reduced_multiplier_step,
)
from ._projection_simultaneous import (
	_solve_simultaneous_state_multiplier_step,
)


def _step_solver_for(
	formulation: ProjectionFormulation,
) -> Callable[..., _ProjectedStep]:
	"""Select the nonlinear equation used to obtain the same physical map."""
	if formulation == "reduced_multiplier":
		return _solve_reduced_multiplier_step
	return _solve_simultaneous_state_multiplier_step


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
	projection_formulation: ProjectionFormulation,
	newton_absolute_tolerance: float,
	newton_relative_tolerance: float,
	newton_max_iterations: int,
	nonlinear_solver: NonlinearSolver,
	progress: bool,
	step_observer: StepObserver | None,
	track_energy: bool = False,
) -> IntegrationData:
	"""Coordinate one projected physical ABBA run with optional energy tracking."""
	dynamics = problem.dynamics
	if not isinstance(dynamics, GuidingCenterJacobianSystem):
		raise TypeError(f"{method_name} requires GuidingCenterJacobianSystem.")
	if dynamics.state_dimension != 2:
		raise TypeError(f"{method_name} requires planar two-component dynamics.")
	_validate_energy_tracking(
		dynamics,
		enabled=track_energy,
		method_name=method_name,
	)
	physical_size = problem.initial_state.size
	particle_count = physical_size // dynamics.state_dimension
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
		state_before, momentum_before = _unpack_energy_tracking_state(
			state,
			physical_size=physical_size,
			particle_count=particle_count,
			enabled=track_energy,
		)

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
					ABBA2ImplicitIntegrationStep(
						dynamics_name=type(dynamics).__name__,
						method_name=method_name,
						step_index=step_index,
						time=t + step,
						duration=step,
						state_before=state_before.copy(),
						state_after=result.state.copy(),
						map_state=apply_step,
						formulation_name=projection_formulation,
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
		momentum_after = momentum_before
		if momentum_before is not None:
			assert isinstance(dynamics, ExtendedHamiltonianSystem)
			momentum_after = (
				momentum_before
				+ _conjugate_momentum_increment_from_stages(
					dynamics,
					t,
					step,
					result.stages,
					particle_count=particle_count,
				)
			)
		return _pack_energy_tracking_state(result.state, momentum_after)

	initial_state = _energy_tracking_initial_state(
		problem.initial_state,
		particle_count=particle_count,
		enabled=track_energy,
	)
	history, step_count = integrate_fixed_grid(
		initial_state,
		request,
		advance,
		progress=bool(progress),
		label=method_name,
	)
	states = np.asarray(history[:physical_size])
	momentum = np.asarray(history[physical_size:]) if track_energy else None
	diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
		"step_count": step_count,
		"nonlinear_solves_per_step": 1,
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
		"projection_formulation": projection_formulation,
		"state_extension": "physical",
		"track_energy": track_energy,
	}
	diagnostics.update(
		_state_dimension_diagnostics(
			"physical",
			projection_formulation,
			particle_count=particle_count,
		)
	)
	diagnostics.update(
		_energy_tracking_diagnostics(
			request.output_times,
			states,
			momentum,
			dynamics,
		)
	)
	return IntegrationData(
		t=request.output_times,
		states=states,
		diagnostics=diagnostics,
	)


@dataclass(frozen=True, slots=True)
class _ABBAImplicitConfig:
	"""Validate configuration shared by projected implicit ABBA methods."""

	projection_formulation: ProjectionFormulation = "reduced_multiplier"
	state_extension: StateExtension = "physical"
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	nonlinear_solver: NonlinearSolver = "newton"
	progress: bool = False
	step_observer: StepObserver | None = None
	track_energy: bool = False

	def __post_init__(self) -> None:
		"""Validate the nonlinear projection solver configuration."""
		object.__setattr__(
			self,
			"projection_formulation",
			_validate_projection_formulation(self.projection_formulation),
		)
		object.__setattr__(
			self,
			"state_extension",
			_validate_state_extension(self.state_extension),
		)
		object.__setattr__(
			self,
			"track_energy",
			_resolved_track_energy(self.track_energy, self.state_extension),
		)
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

__all__: list[str] = []
