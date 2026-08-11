"""Hairer symmetric projection around one complete BM4 composition step."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .._fixed import integrate_fixed_grid
from .._result import IntegrationData
from ..formulations import GCExtendedFormulation
from ..formulations.base import PreparedDirectAdjointFormulation
from ..observation import IntegrationStep, StepObserver
from ..problem import InitialValueProblem
from ..request import SimulationRequest
from .bm4 import _advance_composition


@dataclass(frozen=True, slots=True)
class _ProjectedBM4Step:
	"""Converged physical state and nonlinear diagnostics for one BM4 step."""

	state: np.ndarray
	multiplier: np.ndarray
	iterations: int
	residual_norm: float


def _positive_finite(value: float, name: str) -> float:
	"""Normalize a strictly positive finite parameter."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be positive and finite.")
	result = float(value)
	if not np.isfinite(result) or result <= 0:
		raise ValueError(f"`{name}` must be positive and finite.")
	return result


def _nonnegative_finite(value: float, name: str) -> float:
	"""Normalize a non-negative finite parameter."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be non-negative and finite.")
	result = float(value)
	if not np.isfinite(result) or result < 0:
		raise ValueError(f"`{name}` must be non-negative and finite.")
	return result


def _positive_integer(value: int, name: str) -> int:
	"""Normalize a strictly positive integer parameter."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 1
	):
		raise ValueError(f"`{name}` must be a positive integer.")
	return int(value)


def _bm4_map(
	prepared: PreparedDirectAdjointFormulation,
	t: float,
	internal_state: np.ndarray,
	step: float,
) -> np.ndarray:
	"""Apply the current twelve-stage BM4 map without stage observations."""
	value = np.asarray(internal_state, dtype=float)
	result = _advance_composition(
		prepared,
		t,
		value,
		step,
		step_index=0,
		stage_observer=None,
		formulation_name="GCExtendedFormulation",
		method_name="BM4",
	)
	if result.shape != value.shape or not np.all(np.isfinite(result)):
		raise ValueError("The BM4 base map changed shape or became non-finite.")
	return np.asarray(result, dtype=float)


def _central_difference_jacobian(
	map_state: Callable[[np.ndarray], np.ndarray],
	state: np.ndarray,
	*,
	relative_step: float,
) -> np.ndarray:
	"""Differentiate the doubled BM4 map with centered finite differences."""
	value = np.asarray(state, dtype=float)
	dimension = value.size
	jacobian = np.empty((dimension, dimension), dtype=float)
	for column in range(dimension):
		increment = relative_step * max(1.0, abs(float(value[column])))
		perturbation = np.zeros_like(value)
		perturbation[column] = increment
		forward = np.asarray(map_state(value + perturbation), dtype=float)
		backward = np.asarray(map_state(value - perturbation), dtype=float)
		if forward.shape != value.shape or backward.shape != value.shape:
			raise ValueError("The differentiated BM4 map changed the state shape.")
		jacobian[:, column] = (forward - backward) / (2.0 * increment)
	if not np.all(np.isfinite(jacobian)):
		raise ValueError("The BM4 map Jacobian contains non-finite values.")
	return jacobian


def _projection_matrices(physical_size: int) -> tuple[np.ndarray, np.ndarray]:
	"""Return the diagonal constraint G and normal embedding N=G^T."""
	identity = np.eye(physical_size)
	constraint = np.concatenate((identity, -identity), axis=1)
	normal = np.concatenate((identity, -identity), axis=0)
	return constraint, normal


def _bm4_evaluation(
	prepared: PreparedDirectAdjointFormulation,
	t: float,
	state: np.ndarray,
	step: float,
	multiplier: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Evaluate BM4 from the symmetrically displaced duplicated input."""
	internal_input = np.concatenate((state + multiplier, state - multiplier))
	return internal_input, _bm4_map(prepared, t, internal_input, step)


def _bm4_map_jacobian(
	prepared: PreparedDirectAdjointFormulation,
	t: float,
	internal_input: np.ndarray,
	step: float,
	*,
	relative_step: float,
) -> np.ndarray:
	"""Differentiate one fixed-time complete BM4 map."""
	return _central_difference_jacobian(
		lambda candidate: _bm4_map(prepared, t, candidate, step),
		internal_input,
		relative_step=relative_step,
	)


def _solve_reduced_projected_bm4_step(
	prepared: PreparedDirectAdjointFormulation,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	jacobian_relative_step: float,
) -> _ProjectedBM4Step:
	"""Solve the reduced Hairer projection multiplier equation around BM4."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The BM4 physical state must be a finite, non-empty vector.")
	physical_size = value.size
	constraint, normal = _projection_matrices(physical_size)
	multiplier = np.zeros_like(value)
	state_scale = max(1.0, float(np.linalg.norm(value, ord=np.inf)))
	threshold = absolute_tolerance + relative_tolerance * state_scale

	for iteration in range(max_iterations + 1):
		internal_input, mapped = _bm4_evaluation(
			prepared,
			t,
			value,
			step,
			multiplier,
		)
		residual = constraint @ mapped + 2.0 * multiplier
		residual_norm = float(np.linalg.norm(residual, ord=np.inf))
		if residual_norm <= threshold:
			corrected = mapped + normal @ multiplier
			projected_state = (
				corrected[:physical_size] + corrected[physical_size:]
			) / 2.0
			return _ProjectedBM4Step(
				state=np.asarray(projected_state),
				multiplier=multiplier.copy(),
				iterations=iteration,
				residual_norm=residual_norm,
			)
		if iteration == max_iterations:
			break
		base_jacobian = _bm4_map_jacobian(
			prepared,
			t,
			internal_input,
			step,
			relative_step=jacobian_relative_step,
		)
		residual_jacobian = (
			constraint @ base_jacobian @ normal + 2.0 * np.eye(physical_size)
		)
		try:
			correction = np.linalg.solve(residual_jacobian, residual)
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				"The reduced BM4 projection Jacobian is singular at "
				f"t={t:.16g} with step={step:.16g}."
			) from exc
		multiplier = multiplier - correction

	raise RuntimeError(
		"BM4 implicit formulation 1 did not converge at "
		f"t={t:.16g} with step={step:.16g}: residual norm "
		f"{residual_norm:.3e} exceeds {threshold:.3e} after "
		f"{max_iterations} Newton iterations."
	)


def _solve_simultaneous_projected_bm4_step(
	prepared: PreparedDirectAdjointFormulation,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	jacobian_relative_step: float,
) -> _ProjectedBM4Step:
	"""Solve the simultaneous output-multiplier Hairer projection around BM4."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The BM4 physical state must be a finite, non-empty vector.")
	physical_size = value.size
	internal_size = 2 * physical_size
	constraint, normal = _projection_matrices(physical_size)
	multiplier = np.zeros_like(value)
	internal_input, mapped = _bm4_evaluation(
		prepared,
		t,
		value,
		step,
		multiplier,
	)
	output = mapped.copy()
	state_scale = max(1.0, float(np.linalg.norm(value, ord=np.inf)))
	threshold = absolute_tolerance + relative_tolerance * state_scale

	for iteration in range(max_iterations + 1):
		map_defect = output - normal @ multiplier - mapped
		constraint_defect = constraint @ output
		residual = np.concatenate((map_defect, constraint_defect))
		residual_norm = float(np.linalg.norm(residual, ord=np.inf))
		if residual_norm <= threshold:
			projected_state = (
				output[:physical_size] + output[physical_size:]
			) / 2.0
			return _ProjectedBM4Step(
				state=np.asarray(projected_state),
				multiplier=multiplier.copy(),
				iterations=iteration,
				residual_norm=residual_norm,
			)
		if iteration == max_iterations:
			break
		base_jacobian = _bm4_map_jacobian(
			prepared,
			t,
			internal_input,
			step,
			relative_step=jacobian_relative_step,
		)
		identity_internal = np.eye(internal_size)
		zero = np.zeros((physical_size, physical_size), dtype=float)
		newton_jacobian = np.block(
			[
				[identity_internal, -(identity_internal + base_jacobian) @ normal],
				[constraint, zero],
			]
		)
		try:
			increment = np.linalg.solve(newton_jacobian, -residual)
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				"The simultaneous BM4 projection Jacobian is singular at "
				f"t={t:.16g} with step={step:.16g}."
			) from exc
		output = output + increment[:internal_size]
		multiplier = multiplier + increment[internal_size:]
		internal_input, mapped = _bm4_evaluation(
			prepared,
			t,
			value,
			step,
			multiplier,
		)

	raise RuntimeError(
		"BM4 implicit formulation 2 did not converge at "
		f"t={t:.16g} with step={step:.16g}: simultaneous residual norm "
		f"{residual_norm:.3e} exceeds {threshold:.3e} after "
		f"{max_iterations} Newton iterations."
	)


def _integrate_implicit_bm4(
	method: _ImplicitBM4,
	problem: InitialValueProblem,
	request: SimulationRequest,
) -> IntegrationData:
	"""Integrate physical states with the selected projected BM4 formulation."""
	prepared = GCExtendedFormulation(
		coupling_frequency=method.coupling_frequency
	).prepare(problem, track_energy=False)
	iteration_counts: list[int] = []
	residual_norms: list[float] = []
	multiplier_norms: list[float] = []

	def advance(
		t: float,
		state: np.ndarray,
		step: float,
		step_index: int,
		observe: bool,
	) -> np.ndarray:
		def apply_step(candidate: np.ndarray) -> np.ndarray:
			return type(method)._step_solver(
				prepared,
				t,
				candidate,
				step,
				absolute_tolerance=method.newton_absolute_tolerance,
				relative_tolerance=method.newton_relative_tolerance,
				max_iterations=method.newton_max_iterations,
				jacobian_relative_step=method.newton_jacobian_relative_step,
			).state

		state_before = np.asarray(state, dtype=float)
		result = type(method)._step_solver(
			prepared,
			t,
			state_before,
			step,
			absolute_tolerance=method.newton_absolute_tolerance,
			relative_tolerance=method.newton_relative_tolerance,
			max_iterations=method.newton_max_iterations,
			jacobian_relative_step=method.newton_jacobian_relative_step,
		)
		if observe:
			iteration_counts.append(result.iterations)
			residual_norms.append(result.residual_norm)
			multiplier_norms.append(
				float(np.linalg.norm(result.multiplier, ord=np.inf))
			)
			if method.step_observer is not None:
				method.step_observer(
					IntegrationStep(
						dynamics_name=prepared.dynamics_name,
						method_name=type(method).__name__,
						step_index=step_index,
						time=t + step,
						duration=step,
						state_before=state_before.copy(),
						state_after=result.state.copy(),
						map_state=apply_step,
					)
				)
		return result.state

	history, step_count = integrate_fixed_grid(
		problem.initial_state,
		request,
		advance,
		progress=method.progress,
		label=type(method).__name__,
	)
	diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
		"step_count": step_count,
		"newton_iterations": np.asarray(iteration_counts, dtype=int),
		"newton_residual_norms": np.asarray(residual_norms, dtype=float),
		"projection_multiplier_norms": np.asarray(multiplier_norms, dtype=float),
		"newton_absolute_tolerance": method.newton_absolute_tolerance,
		"newton_relative_tolerance": method.newton_relative_tolerance,
		"newton_max_iterations": method.newton_max_iterations,
		"newton_jacobian_relative_step": method.newton_jacobian_relative_step,
		"coupling_frequency": method.coupling_frequency,
		"projection_solver_formulation": type(method)._solver_formulation,
	}
	return IntegrationData(
		t=request.output_times,
		states=np.asarray(history),
		diagnostics=diagnostics,
	)


@dataclass(frozen=True, slots=True)
class _ImplicitBM4:
	"""Shared configuration for Hairer-projected BM4 methods."""

	coupling_frequency: float = np.pi / 8
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	progress: bool = False
	step_observer: StepObserver | None = None

	_step_solver: ClassVar[Callable[..., _ProjectedBM4Step]] = (
		_solve_reduced_projected_bm4_step
	)
	_solver_formulation: ClassVar[str] = "bm4_implicit_1_reduced"

	def __post_init__(self) -> None:
		"""Validate coupling and nonlinear-solver controls."""
		object.__setattr__(
			self,
			"coupling_frequency",
			_nonnegative_finite(self.coupling_frequency, "coupling_frequency"),
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
			_positive_integer(self.newton_max_iterations, "newton_max_iterations"),
		)
		object.__setattr__(
			self,
			"newton_jacobian_relative_step",
			_positive_finite(
				self.newton_jacobian_relative_step,
				"newton_jacobian_relative_step",
			),
		)

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate one GC problem with the selected projected BM4 solve."""
		return _integrate_implicit_bm4(self, problem, request)


__all__: list[str] = []
