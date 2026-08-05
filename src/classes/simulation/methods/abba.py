"""ABBA integration with Hairer's symmetric projection for GC dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from classes.dynamics import GuidingCenterJacobianSystem

from .._fixed import integrate_fixed_grid
from .._result import IntegrationData
from ..observation import IntegrationStep, StepObserver
from ..problem import InitialValueProblem
from ..request import SimulationRequest


@dataclass(frozen=True, slots=True)
class _ProjectedStep:
	"""Converged physical state and nonlinear-solve diagnostics for one step."""

	state: np.ndarray
	multiplier: np.ndarray
	ideal_state_jacobian: np.ndarray | None
	iterations: int
	residual_norm: float


@dataclass(frozen=True, slots=True)
class _ResidualEvaluation:
	"""ABBA stages and the exact reduced residual Jacobian at one multiplier."""

	u_final: np.ndarray
	v_final: np.ndarray
	residual: np.ndarray
	jacobian: np.ndarray
	abba_jacobian: np.ndarray


@dataclass(frozen=True, slots=True)
class _ABBAStages:
	"""State points traversed by one explicit endpoint-time ABBA map."""

	u_initial: np.ndarray
	v_initial: np.ndarray
	u_first: np.ndarray
	v_final: np.ndarray
	u_final: np.ndarray
	residual: np.ndarray


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


def _checked_vector_field(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate one GC vector field without allowing a layout change."""
	result = np.asarray(dynamics.vector_field(t, state), dtype=float)
	if result.shape != state.shape or not np.all(np.isfinite(result)):
		raise ValueError("The GC vector field changed shape or became non-finite.")
	return result


def _checked_vector_field_jacobian(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate batched exact GC Jacobians with one matrix per particle."""
	result = np.asarray(
		dynamics.particle_vector_field_jacobians(t, state),
		dtype=float,
	)
	particle_count = state.size // dynamics.state_dimension
	expected_shape = (particle_count, dynamics.state_dimension, dynamics.state_dimension)
	if result.shape != expected_shape or not np.all(np.isfinite(result)):
		raise ValueError(
			"The GC vector-field Jacobian changed shape or became non-finite."
		)
	return result


def _evaluate_stages(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	multiplier: np.ndarray,
) -> _ABBAStages:
	"""Apply the four explicit ABBA stages and evaluate equation (11)."""
	half_step = step / 2.0
	final_time = t + step

	# G^T mu displaces the two copies in opposite physical directions.
	u_initial = state + multiplier
	v_initial = state - multiplier
	u_first = u_initial + half_step * _checked_vector_field(
		dynamics,
		t,
		v_initial,
	)

	v_first = v_initial + half_step * _checked_vector_field(
		dynamics,
		t,
		u_first,
	)

	v_final = v_first + half_step * _checked_vector_field(
		dynamics,
		final_time,
		u_first,
	)

	u_final = u_first + half_step * _checked_vector_field(
		dynamics,
		final_time,
		v_final,
	)

	residual = u_final - v_final + 2.0 * multiplier
	return _ABBAStages(
		u_initial=u_initial,
		v_initial=v_initial,
		u_first=u_first,
		v_final=v_final,
		u_final=u_final,
		residual=residual,
	)


def _differentiate_stages(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	stages: _ABBAStages,
) -> _ResidualEvaluation:
	"""Evaluate equations (15) and (17) at the traversed ABBA stage points."""
	half_step = step / 2.0
	final_time = t + step
	w_1 = _checked_vector_field_jacobian(dynamics, t, stages.v_initial)
	w_2 = _checked_vector_field_jacobian(dynamics, t, stages.u_first)
	w_3 = _checked_vector_field_jacobian(dynamics, final_time, stages.u_first)
	w_4 = _checked_vector_field_jacobian(dynamics, final_time, stages.v_final)
	particle_count = state.size // dynamics.state_dimension
	identity = np.broadcast_to(
		np.eye(dynamics.state_dimension),
		(particle_count, dynamics.state_dimension, dynamics.state_dimension),
	)
	central_jacobian = w_2 + w_3
	# Matrix products retain their order because the stage Jacobians need not commute.
	jacobian = (
		4.0 * identity
		- half_step * (w_1 + w_2 + w_3 + w_4)
		+ half_step**2
		* (w_4 @ central_jacobian + central_jacobian @ w_1)
		- half_step**3 * (w_4 @ central_jacobian @ w_1)
	)
	top_left = identity + half_step**2 * (w_4 @ central_jacobian)
	top_right = (
		half_step * (w_1 + w_4)
		+ half_step**3 * (w_4 @ central_jacobian @ w_1)
	)
	bottom_left = half_step * central_jacobian
	bottom_right = identity + half_step**2 * (central_jacobian @ w_1)
	abba_jacobian = np.concatenate(
		(
			np.concatenate((top_left, top_right), axis=-1),
			np.concatenate((bottom_left, bottom_right), axis=-1),
		),
		axis=-2,
	)
	return _ResidualEvaluation(
		u_final=stages.u_final,
		v_final=stages.v_final,
		residual=stages.residual,
		jacobian=jacobian,
		abba_jacobian=abba_jacobian,
	)


def _evaluate_residual(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	multiplier: np.ndarray,
) -> _ResidualEvaluation:
	"""Apply ABBA and evaluate its exact residual and map Jacobians."""
	stages = _evaluate_stages(dynamics, t, state, step, multiplier)
	return _differentiate_stages(dynamics, t, state, step, stages)


def _dense_component_major_jacobian(blocks: np.ndarray) -> np.ndarray:
	"""Expand independent particle Jacobians into the packed physical layout."""
	particle_count = blocks.shape[0]
	result = np.zeros((2 * particle_count, 2 * particle_count), dtype=float)
	for particle in range(particle_count):
		indices = (particle, particle_count + particle)
		result[np.ix_(indices, indices)] = blocks[particle]
	return result


def _ideal_projected_state_jacobian(
	evaluation: _ResidualEvaluation,
) -> np.ndarray:
	"""Differentiate the exact-root projected map by the implicit theorem.

	This is the tangent of the mathematical method defined by ``R(mu) = 0``. It
	does not differentiate a finite Newton iteration or its stopping decision.
	"""
	particle_count = evaluation.jacobian.shape[0]
	identity = np.broadcast_to(np.eye(2), (particle_count, 2, 2))
	embedding = np.concatenate((identity, identity), axis=-2)
	constraint_direction = np.concatenate((identity, -identity), axis=-2)
	constraint = np.concatenate((identity, -identity), axis=-1)
	average = np.concatenate((identity, identity), axis=-1) / 2.0
	abba_jacobian = evaluation.abba_jacobian
	residual_state_jacobian = constraint @ abba_jacobian @ embedding
	try:
		multiplier_state_jacobian = -np.linalg.solve(
			evaluation.jacobian,
			residual_state_jacobian,
		)
	except np.linalg.LinAlgError as exc:
		raise RuntimeError(
			"The ABBA projection Jacobian is singular while differentiating the step."
		) from exc
	input_state_jacobian = (
		embedding + constraint_direction @ multiplier_state_jacobian
	)
	physical_blocks = average @ abba_jacobian @ input_state_jacobian
	return _dense_component_major_jacobian(physical_blocks)


def _solve_projected_step(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	compute_ideal_state_jacobian: bool = True,
) -> _ProjectedStep:
	"""Solve Hairer's two-sided projection with exact reduced Newton steps."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The ABBA physical state must be a finite, non-empty vector.")
	multiplier = np.zeros_like(value)
	state_scale = max(1.0, float(np.linalg.norm(value, ord=np.inf)))
	threshold = absolute_tolerance + relative_tolerance * state_scale

	for iteration in range(max_iterations + 1):
		stages = _evaluate_stages(
			dynamics,
			t,
			value,
			step,
			multiplier,
		)
		residual_norm = float(np.linalg.norm(stages.residual, ord=np.inf))
		if residual_norm <= threshold:
			# Both projected copies agree to the requested tolerance. Their mean is
			# the numerically neutral representative of the physical diagonal state.
			first_copy = stages.u_final + multiplier
			second_copy = stages.v_final - multiplier
			projected_state = (first_copy + second_copy) / 2.0
			ideal_state_jacobian = None
			if compute_ideal_state_jacobian:
				evaluation = _differentiate_stages(
					dynamics,
					t,
					value,
					step,
					stages,
				)
				ideal_state_jacobian = _ideal_projected_state_jacobian(evaluation)
			return _ProjectedStep(
				state=np.asarray(projected_state),
				multiplier=multiplier.copy(),
				ideal_state_jacobian=ideal_state_jacobian,
				iterations=iteration,
				residual_norm=residual_norm,
			)
		if iteration == max_iterations:
			break
		evaluation = _differentiate_stages(
			dynamics,
			t,
			value,
			step,
			stages,
		)
		# The packed residual is component-major; Newton systems are independent
		# two-dimensional solves for the individual GC particles.
		residual_blocks = evaluation.residual.reshape(
			dynamics.state_dimension,
			-1,
		).T
		try:
			correction_blocks = np.linalg.solve(
				evaluation.jacobian,
				residual_blocks[..., None],
			)[..., 0]
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				"The ABBA projection Jacobian is singular at "
				f"t={t:.16g} with step={step:.16g}."
			) from exc
		correction = correction_blocks.T.reshape(-1)
		multiplier = multiplier - correction

	raise RuntimeError(
		"The ABBA symmetric projection did not converge at "
		f"t={t:.16g} with step={step:.16g}: "
		f"residual norm {residual_norm:.3e} exceeds {threshold:.3e} after "
		f"{max_iterations} Newton iterations."
	)


@dataclass(frozen=True, slots=True)
class SymmetricProjectedABBA:
	"""Second-order ABBA method closed by Hairer's symmetric projection.

	The explicit A-B-B-A map evolves two GC copies at the two step endpoints.
	A reduced Newton solve then finds the opposite input/output corrections that
	return the result to the physical diagonal. The residual Jacobian is evaluated
	exactly from spatial second derivatives of the effective potential.
	"""

	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	progress: bool = False
	step_observer: StepObserver | None = None

	def __post_init__(self) -> None:
		"""Validate the nonlinear solver configuration."""
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

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate one GC problem and retain nonlinear-solve diagnostics."""
		dynamics = problem.dynamics
		if not isinstance(dynamics, GuidingCenterJacobianSystem):
			raise TypeError(
				"SymmetricProjectedABBA requires GuidingCenterJacobianSystem."
			)
		if dynamics.state_dimension != 2:
			raise TypeError(
				"SymmetricProjectedABBA requires planar two-component dynamics."
			)
		# Preflight the exact Hessian capability before the integration grid advances.
		_checked_vector_field_jacobian(
			dynamics,
			request.t_span[0],
			problem.initial_state,
		)

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
				"""Apply this fixed-time projected ABBA map to one candidate."""
				return _solve_projected_step(
					dynamics,
					t,
					candidate,
					step,
					absolute_tolerance=self.newton_absolute_tolerance,
					relative_tolerance=self.newton_relative_tolerance,
					max_iterations=self.newton_max_iterations,
					compute_ideal_state_jacobian=False,
				).state

			state_before = np.asarray(state, dtype=float)
			result = _solve_projected_step(
				dynamics,
				t,
				state_before,
				step,
				absolute_tolerance=self.newton_absolute_tolerance,
				relative_tolerance=self.newton_relative_tolerance,
				max_iterations=self.newton_max_iterations,
				compute_ideal_state_jacobian=False,
			)
			if observe:
				iteration_counts.append(result.iterations)
				residual_norms.append(result.residual_norm)
				multiplier_norms.append(
					float(np.linalg.norm(result.multiplier, ord=np.inf))
				)
				if self.step_observer is not None:
					self.step_observer(
						IntegrationStep(
							dynamics_name=type(dynamics).__name__,
							method_name=type(self).__name__,
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
			progress=bool(self.progress),
			label=type(self).__name__,
		)
		diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
			"step_count": step_count,
			"newton_iterations": np.asarray(iteration_counts, dtype=int),
			"newton_residual_norms": np.asarray(residual_norms, dtype=float),
			"projection_multiplier_norms": np.asarray(
				multiplier_norms,
				dtype=float,
			),
			"newton_absolute_tolerance": self.newton_absolute_tolerance,
			"newton_relative_tolerance": self.newton_relative_tolerance,
			"newton_max_iterations": self.newton_max_iterations,
		}
		return IntegrationData(
			t=request.output_times,
			states=np.asarray(history),
			diagnostics=diagnostics,
		)


__all__ = ["SymmetricProjectedABBA"]
