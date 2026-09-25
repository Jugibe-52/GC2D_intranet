"""Forward/backward tangent comparisons for implicit ABBA steps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Literal

import numpy as np

from diagnostics._validation import positive_integer as _positive_integer

from dynamics import GuidingCenterJacobianSystem
from methods.extended.configuration import ABBA_PROJECTION_FORMULATIONS
from contracts.observation import (
	ABBA4ImplicitIntegrationStep,
	ABBA2ImplicitIntegrationStep,
	IntegrationStep,
)
from methods._nonlinear import NONLINEAR_SOLVERS, NonlinearSolver
from methods._nonlinear import SolverOptions
from methods.extended.configuration import _validate_projection_formulation
from methods.extended.observations import bind_event_builder
from methods.extended.core.records import StepResult
from contracts.problem import InitialValueProblem
from formulations.gc import GCDoubledMaps
from initial_conditions import GCInitialConfiguration
from methods.extended.core.composition import ABBA2, ABBA4, ABBA6
from methods.extended.core.projection import solve_projection
from contracts.observation import ABBA6ImplicitIntegrationStep

from .jacobians import implicit_function_step_jacobian
from .trajectory_symplecticity.jacobians import (
	abba4_implicit_step_particle_jacobians,
	abba6_implicit_step_particle_jacobians,
)


_ObservedStep = ABBA2ImplicitIntegrationStep | ABBA4ImplicitIntegrationStep | ABBA6ImplicitIntegrationStep


def _dense_component_major_jacobian(blocks: np.ndarray) -> np.ndarray:
	"""Expand independent planar blocks into the packed physical layout."""
	values = np.asarray(blocks, dtype=float)
	if values.ndim != 3 or values.shape[1:] != (2, 2):
		raise ValueError("Planar Jacobian blocks must have shape (N, 2, 2).")
	particle_count = values.shape[0]
	result = np.zeros((2 * particle_count, 2 * particle_count), dtype=float)
	for particle in range(particle_count):
		indices = (particle, particle_count + particle)
		result[np.ix_(indices, indices)] = values[particle]
	if not np.all(np.isfinite(result)):
		raise ValueError("The complete ABBA Jacobian is non-finite.")
	return result


def _complete_step_jacobian(step: _ObservedStep) -> np.ndarray:
	"""Return the exact complete-map tangent for ABBA2 or an outer-projected composition."""
	if isinstance(step, ABBA6ImplicitIntegrationStep):
		return _dense_component_major_jacobian(abba6_implicit_step_particle_jacobians(step))
	if isinstance(step, ABBA4ImplicitIntegrationStep):
		return _dense_component_major_jacobian(
			abba4_implicit_step_particle_jacobians(step)
		)
	return implicit_function_step_jacobian(step)


@dataclass(frozen=True, slots=True)
class ImplicitABBAReversibilitySample:
	"""Forward and independently recomputed backward data for one ABBA step."""

	observation_index: int
	step_index: int
	start_time: float
	end_time: float
	duration: float
	method_name: str
	formulation_name: str
	state_before: np.ndarray
	state_after: np.ndarray
	backward_state: np.ndarray
	velocity_before: np.ndarray
	velocity_after: np.ndarray
	forward_jacobian: np.ndarray
	backward_jacobian: np.ndarray
	jacobian_composition_defect: np.ndarray
	forward_action_on_initial_velocity: np.ndarray
	backward_action_on_final_velocity: np.ndarray
	endpoint_velocity_action_difference: np.ndarray
	forward_increment: np.ndarray
	backward_increment: np.ndarray
	increment_direct_difference: np.ndarray
	increment_closure: np.ndarray
	backward_state_error: np.ndarray
	jacobian_composition_defect_norm: float
	endpoint_velocity_action_difference_norm: float
	forward_increment_norm: float
	backward_increment_norm: float
	increment_direct_difference_norm: float
	increment_closure_norm: float
	increment_closure_scale: float
	normalized_increment_closure: float
	backward_state_error_norm: float


def _positive_finite(value: float, name: str) -> float:
	"""Normalize one strictly positive finite solver parameter."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be positive and finite.")
	result = float(value)
	if not np.isfinite(result) or result <= 0.0:
		raise ValueError(f"`{name}` must be positive and finite.")
	return result


def _finite_vector(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
	"""Validate one packed vector without changing its coordinate layout."""
	result = np.asarray(value, dtype=float)
	if result.shape != shape or not np.all(np.isfinite(result)):
		raise ValueError(f"`{name}` must be finite and have shape {shape}.")
	return result


class ImplicitABBAReversibilityObserver:
	"""Compare exact forward and independently solved backward ABBA tangents.

	For each selected accepted step, the observer solves a genuine signed reverse
	step from the accepted endpoint with the same nonlinear-solver configuration.
	It therefore does not define the backward tangent as the algebraic inverse of
	the forward tangent. The resulting ``J_minus @ J_plus - I`` defect measures
	the actual nonlinear solves, analytic tangents, and floating-point arithmetic.
	"""

	def __init__(
		self,
		*,
		newton_absolute_tolerance: float = 1e-13,
		newton_relative_tolerance: float = 1e-12,
		newton_max_iterations: int = 12,
		nonlinear_solver: NonlinearSolver = "newton",
		sample_every: int = 1,
		verbose: bool = False,
	) -> None:
		"""Configure independent reverse solves and in-memory observations."""
		self.newton_absolute_tolerance = _positive_finite(
			newton_absolute_tolerance,
			"newton_absolute_tolerance",
		)
		self.newton_relative_tolerance = _positive_finite(
			newton_relative_tolerance,
			"newton_relative_tolerance",
		)
		self.newton_max_iterations = _positive_integer(
			newton_max_iterations,
			"newton_max_iterations",
		)
		if nonlinear_solver not in NONLINEAR_SOLVERS:
			raise ValueError("Unknown nonlinear solver for implicit ABBA reversibility.")
		self.nonlinear_solver = nonlinear_solver
		self.sample_every = _positive_integer(sample_every, "sample_every")
		self.verbose = bool(verbose)
		self._expected_step = 0
		self._samples: list[ImplicitABBAReversibilitySample] = []

	@property
	def samples(self) -> tuple[ImplicitABBAReversibilitySample, ...]:
		"""Return all retained forward/backward comparison samples."""
		return tuple(self._samples)

	def _solve_reverse_step(
		self, step: _ObservedStep, dynamics: GuidingCenterJacobianSystem,
	) -> _ObservedStep:
		"""Recompute one signed outer projection with the observed method's recipe."""
		if step.formulation_name not in ABBA_PROJECTION_FORMULATIONS:
			raise TypeError("The observed ABBA step has an unknown formulation.")
		formulation = _validate_projection_formulation(step.formulation_name)
		recipe = ABBA2
		order: Literal[2, 4, 6] = 2
		if isinstance(step, ABBA4ImplicitIntegrationStep):
			recipe, order = ABBA4, 4
		elif isinstance(step, ABBA6ImplicitIntegrationStep):
			recipe, order = ABBA6, 6
		problem = InitialValueProblem(dynamics, GCInitialConfiguration(step.state_after))
		options = SolverOptions(self.nonlinear_solver, self.newton_absolute_tolerance,
			self.newton_relative_tolerance, self.newton_max_iterations)
		project = partial(solve_projection, GCDoubledMaps(problem, coupling_frequency=None),
			recipe, options, formulation)
		result = project(step.time, step.state_after, -step.duration)
		builder = bind_event_builder(dynamics, step.method_name, formulation, order=order,
			coefficients=tuple(2 * c for c in recipe.coefficients[::2]), project=project)
		event = builder(step.time, -step.duration, step.step_index, step.state_after,
			StepResult(result.state, (result,)))
		assert isinstance(event, (ABBA2ImplicitIntegrationStep, ABBA4ImplicitIntegrationStep,
			ABBA6ImplicitIntegrationStep))
		return replace(event, time=float(step.start_time))

	def __call__(self, step: IntegrationStep) -> None:
		"""Observe one consecutive accepted implicit-ABBA step."""
		if not isinstance(
			step,
			(ABBA2ImplicitIntegrationStep, ABBA4ImplicitIntegrationStep, ABBA6ImplicitIntegrationStep),
		):
			raise TypeError(
				"ImplicitABBAReversibilityObserver requires "
				"ABBA2, ABBA4 or ABBA6 integration-step data."
			)
		if step.step_index != self._expected_step:
			raise ValueError("Implicit ABBA steps must be observed consecutively.")
		self._expected_step += 1
		if step.step_index % self.sample_every:
			return
		if step.nonlinear_solver != self.nonlinear_solver:
			raise ValueError(
				"Forward and backward ABBA steps must use the same nonlinear solver."
			)
		dynamics = step.dynamics
		if not isinstance(dynamics, GuidingCenterJacobianSystem):
			raise TypeError(
				"Implicit ABBA reversibility requires GuidingCenterJacobianSystem dynamics."
			)

		state_before = np.asarray(step.state_before, dtype=float)
		if (
			state_before.ndim != 1
			or state_before.size == 0
			or state_before.size % 2
			or not np.all(np.isfinite(state_before))
		):
			raise ValueError("The observed physical state must be finite and planar.")
		shape = state_before.shape
		state_after = _finite_vector(step.state_after, shape, "state_after")
		if not np.isclose(
			step.time,
			step.start_time + step.duration,
			rtol=0.0,
			atol=float(
				64.0
				* np.finfo(float).eps
				* max(1.0, abs(step.time), abs(step.start_time))
			),
		):
			raise ValueError("The observed ABBA step times are inconsistent.")

		forward_jacobian = _complete_step_jacobian(step)
		reverse_step = self._solve_reverse_step(step, dynamics)
		backward_jacobian = _complete_step_jacobian(reverse_step)
		backward_state = _finite_vector(
			reverse_step.state_after,
			shape,
			"backward_state",
		)
		velocity_before = _finite_vector(
			dynamics.vector_field(float(step.start_time), state_before),
			shape,
			"velocity_before",
		)
		velocity_after = _finite_vector(
			dynamics.vector_field(float(step.time), state_after),
			shape,
			"velocity_after",
		)

		identity = np.eye(state_before.size)
		jacobian_defect = backward_jacobian @ forward_jacobian - identity
		forward_action = forward_jacobian @ velocity_before
		backward_final_action = backward_jacobian @ velocity_after
		action_difference = forward_action - backward_final_action
		duration = float(step.duration)
		forward_increment = (
			duration * velocity_before
			+ 0.5 * duration**2 * forward_action
		)
		backward_increment = (
			-duration * velocity_after
			+ 0.5 * duration**2 * backward_final_action
		)
		increment_direct_difference = forward_increment - backward_increment
		increment_closure = forward_increment + backward_increment
		backward_state_error = backward_state - state_before
		forward_increment_norm = float(np.linalg.norm(forward_increment))
		backward_increment_norm = float(np.linalg.norm(backward_increment))
		increment_closure_norm = float(np.linalg.norm(increment_closure))
		increment_closure_scale = max(
			forward_increment_norm,
			backward_increment_norm,
		)
		normalized_increment_closure = (
			increment_closure_norm / increment_closure_scale
			if increment_closure_scale > 0.0
			else 0.0
		)

		sample = ImplicitABBAReversibilitySample(
			observation_index=len(self._samples),
			step_index=step.step_index,
			start_time=float(step.start_time),
			end_time=float(step.time),
			duration=duration,
			method_name=step.method_name,
			formulation_name=step.formulation_name,
			state_before=state_before.copy(),
			state_after=state_after.copy(),
			backward_state=backward_state.copy(),
			velocity_before=velocity_before.copy(),
			velocity_after=velocity_after.copy(),
			forward_jacobian=forward_jacobian.copy(),
			backward_jacobian=backward_jacobian.copy(),
			jacobian_composition_defect=jacobian_defect.copy(),
			forward_action_on_initial_velocity=forward_action.copy(),
			backward_action_on_final_velocity=backward_final_action.copy(),
			endpoint_velocity_action_difference=action_difference.copy(),
			forward_increment=forward_increment.copy(),
			backward_increment=backward_increment.copy(),
			increment_direct_difference=increment_direct_difference.copy(),
			increment_closure=increment_closure.copy(),
			backward_state_error=backward_state_error.copy(),
			jacobian_composition_defect_norm=float(
				np.linalg.norm(jacobian_defect, ord="fro")
			),
			endpoint_velocity_action_difference_norm=float(
				np.linalg.norm(action_difference)
			),
			forward_increment_norm=forward_increment_norm,
			backward_increment_norm=backward_increment_norm,
			increment_direct_difference_norm=float(
				np.linalg.norm(increment_direct_difference)
			),
			increment_closure_norm=increment_closure_norm,
			increment_closure_scale=increment_closure_scale,
			normalized_increment_closure=normalized_increment_closure,
			backward_state_error_norm=float(np.linalg.norm(backward_state_error)),
		)
		self._samples.append(sample)
		if self.verbose:
			print(
				f"[implicit-abba-reversibility] step={step.step_index:05d} "
				f"t={step.time:.6g} ||J- J+ - I||="
				f"{sample.jacobian_composition_defect_norm:.3e} "
				f"relative Delta closure={sample.normalized_increment_closure:.3e}"
			)


__all__ = [
	"ImplicitABBAReversibilityObserver",
	"ImplicitABBAReversibilitySample",
]
