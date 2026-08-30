"""Implicit full-diagonal projection after duplicating ``(z, t, k)``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from dynamics import GuidingCenterDynamics

from .._fixed import integrate_fixed_grid
from .._result import IntegrationData
from ..formulations import gc_coupling_matrix
from ..observation import (
	FullyExtendedBaseMap,
	FullyExtendedImplicitIntegrationStep,
	StepObserver,
)
from ..problem import InitialValueProblem
from ..request import SimulationRequest
from ._nonlinear import (
	NonlinearSolver,
	_solve_broyden,
	_validate_nonlinear_solver,
)
from .bm4._core import _BM4_ORDERS, _BM4_STAGES


_ExtendedMap = Callable[[np.ndarray], np.ndarray]
_ExtendedJacobian = Callable[[np.ndarray], np.ndarray]
_Variant = Literal["abba", "abba4", "bm4"]
_CUBE_ROOT_TWO = float(np.cbrt(2.0))
_GAMMA = 1.0 / (2.0 - _CUBE_ROOT_TWO)
_DELTA = -_CUBE_ROOT_TWO / (2.0 - _CUBE_ROOT_TWO)
_ABBA4_COEFFICIENTS = np.asarray((_GAMMA, _DELTA, _GAMMA), dtype=float)
_IDENTITY_4 = np.eye(4)
_IDENTITY_8 = np.eye(8)
_DIAGONAL_EMBEDDING = np.vstack((_IDENTITY_4, _IDENTITY_4))
_ANTIDIAGONAL_EMBEDDING = np.vstack((_IDENTITY_4, -_IDENTITY_4))
_COPY_DIFFERENCE = np.hstack((_IDENTITY_4, -_IDENTITY_4))
_COPY_AVERAGE = 0.5 * np.hstack((_IDENTITY_4, _IDENTITY_4))


@dataclass(frozen=True, slots=True)
class _AnalyticExtendedMap:
	"""An ``R^8`` map together with its exact stage-product Jacobian."""

	map_state: _ExtendedMap
	jacobian_state: _ExtendedJacobian


def _positive_finite(value: float, name: str) -> float:
	"""Normalize one strictly positive finite method parameter."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be positive and finite.")
	result = float(value)
	if not np.isfinite(result) or result <= 0.0:
		raise ValueError(f"`{name}` must be positive and finite.")
	return result


def _nonnegative_finite(value: float, name: str) -> float:
	"""Normalize one non-negative finite method parameter."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be non-negative and finite.")
	result = float(value)
	if not np.isfinite(result) or result < 0.0:
		raise ValueError(f"`{name}` must be non-negative and finite.")
	return result


def _positive_integer(value: int, name: str) -> int:
	"""Normalize one strictly positive integer method parameter."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 1
	):
		raise ValueError(f"`{name}` must be a positive integer.")
	return int(value)


def _checked_extended_state(state: np.ndarray, *, duplicated: bool) -> np.ndarray:
	"""Validate one physical ``R^4`` or duplicated ``R^8`` extended state."""
	value = np.asarray(state, dtype=float)
	expected = (8,) if duplicated else (4,)
	if value.shape != expected or not np.all(np.isfinite(value)):
		space = "duplicated R^8" if duplicated else "physical R^4"
		raise ValueError(f"The {space} state must be finite with shape {expected}.")
	return value.copy()


def _extended_vector_field(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate ``X_K=(f(t,z), 1, -partial_t h)`` in ``(x,y,t,k)`` order."""
	value = _checked_extended_state(state, duplicated=False)
	physical = np.asarray(dynamics.vector_field(float(value[2]), value[:2]), dtype=float)
	momentum = np.asarray(
		dynamics.extended_momentum_derivative(float(value[2]), value[:2]),
		dtype=float,
	)
	if physical.shape != (2,) or momentum.size != 1:
		raise ValueError("The one-particle extended GC field changed shape.")
	return np.asarray((physical[0], physical[1], 1.0, momentum.reshape(-1)[0]))


def _extended_vector_field_jacobian(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
) -> np.ndarray:
	"""Return the analytic ``4 x 4`` Jacobian of ``X_K``.

	The effective potential has harmonic time dependence, hence
	``partial_tt h = -h``. Mixed space-time derivatives are evaluated directly
	from the same periodic spline used by the vector field.
	"""
	value = _checked_extended_state(state, duplicated=False)
	potential = dynamics.effective_potential
	if potential.interpolation_order < 3:
		raise ValueError(
			"Analytic full-state Jacobians require interpolation_order >= 3."
		)
	time = float(value[2])
	x = np.asarray([value[0]])
	y = np.asarray([value[1]])
	spatial = np.asarray(
		dynamics.particle_vector_field_jacobians(time, value[:2]),
		dtype=float,
	)[0]
	h_tx = float(potential.evaluate(time, x, y, dx=1, dt=1)[0])
	h_ty = float(potential.evaluate(time, x, y, dy=1, dt=1)[0])
	h_tt = -float(potential.evaluate(time, x, y)[0])
	result = np.zeros((4, 4), dtype=float)
	result[:2, :2] = spatial
	result[:2, 2] = (-h_ty, h_tx)
	result[3, :3] = (-h_tx, -h_ty, -h_tt)
	return result


def _flow_first_jacobian(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	duration: float,
) -> np.ndarray:
	"""Return the analytic Jacobian of the first-Hamiltonian shear."""
	value = _checked_extended_state(state, duplicated=True)
	result = _IDENTITY_8.copy()
	result[4:, :4] = duration * _extended_vector_field_jacobian(
		dynamics,
		value[:4],
	)
	return result


def _flow_second_jacobian(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	duration: float,
) -> np.ndarray:
	"""Return the analytic Jacobian of the second-Hamiltonian shear."""
	value = _checked_extended_state(state, duplicated=True)
	result = _IDENTITY_8.copy()
	result[:4, 4:] = duration * _extended_vector_field_jacobian(
		dynamics,
		value[4:],
	)
	return result


def _flow_first(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	duration: float,
) -> np.ndarray:
	"""Flow ``K(Z_1)`` exactly: hold the first copy and shear the second."""
	value = _checked_extended_state(state, duplicated=True)
	value[4:] += duration * _extended_vector_field(dynamics, value[:4])
	return value


def _flow_second(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	duration: float,
) -> np.ndarray:
	"""Flow ``K(Z_2)`` exactly: hold the second copy and shear the first."""
	value = _checked_extended_state(state, duplicated=True)
	value[:4] += duration * _extended_vector_field(dynamics, value[4:])
	return value


def _couple_physical_copies(
	state: np.ndarray,
	*,
	duration: float,
	frequency: float,
) -> np.ndarray:
	"""Apply the exact GC binding flow to ``z`` while leaving both ``(t,k)`` pairs."""
	value = _checked_extended_state(state, duplicated=True)
	physical = np.asarray((value[0], value[1], value[4], value[5]))
	coupled = gc_coupling_matrix(duration, frequency) @ physical
	value[[0, 1, 4, 5]] = coupled
	return value


def _physical_coupling_jacobian(
	*,
	duration: float,
	frequency: float,
) -> np.ndarray:
	"""Embed the exact physical binding matrix into the full ``R^8`` state."""
	indices = np.asarray((0, 1, 4, 5))
	result = _IDENTITY_8.copy()
	result[np.ix_(indices, indices)] = gc_coupling_matrix(duration, frequency)
	return result


def _abba_base_map(
	dynamics: GuidingCenterDynamics,
	duration: float,
) -> _AnalyticExtendedMap:
	"""Return the palindromic four-shear full-state ABBA map on ``R^8``."""
	half_step = duration / 2.0

	def map_state(candidate: np.ndarray) -> np.ndarray:
		value = _flow_second(dynamics, candidate, half_step)
		value = _flow_first(dynamics, value, half_step)
		value = _flow_first(dynamics, value, half_step)
		return _flow_second(dynamics, value, half_step)

	def jacobian_state(candidate: np.ndarray) -> np.ndarray:
		value = _checked_extended_state(candidate, duplicated=True)
		total = _IDENTITY_8.copy()
		for flow, jacobian in (
			(_flow_second, _flow_second_jacobian),
			(_flow_first, _flow_first_jacobian),
			(_flow_first, _flow_first_jacobian),
			(_flow_second, _flow_second_jacobian),
		):
			factor = jacobian(dynamics, value, half_step)
			value = flow(dynamics, value, half_step)
			total = factor @ total
		return total

	return _AnalyticExtendedMap(
		map_state=map_state,
		jacobian_state=jacobian_state,
	)


def _bm4_direct_map(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	*,
	duration: float,
	frequency: float,
) -> np.ndarray:
	"""Apply first-copy shear, second-copy shear, then physical binding."""
	value = _flow_first(dynamics, state, duration)
	value = _flow_second(dynamics, value, duration)
	return _couple_physical_copies(
		value,
		duration=duration,
		frequency=frequency,
	)


def _bm4_direct_jacobian(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	*,
	duration: float,
	frequency: float,
) -> np.ndarray:
	"""Return the analytic direct-stage product in flow order."""
	value = _checked_extended_state(state, duplicated=True)
	first = _flow_first_jacobian(dynamics, value, duration)
	value = _flow_first(dynamics, value, duration)
	second = _flow_second_jacobian(dynamics, value, duration)
	coupling = _physical_coupling_jacobian(
		duration=duration,
		frequency=frequency,
	)
	return np.asarray(coupling @ second @ first)


def _bm4_adjoint_map(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	*,
	duration: float,
	frequency: float,
) -> np.ndarray:
	"""Reverse the direct-map factor order to obtain its exact adjoint."""
	value = _couple_physical_copies(
		state,
		duration=duration,
		frequency=frequency,
	)
	value = _flow_second(dynamics, value, duration)
	return _flow_first(dynamics, value, duration)


def _bm4_adjoint_jacobian(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	*,
	duration: float,
	frequency: float,
) -> np.ndarray:
	"""Return the analytic adjoint-stage product in flow order."""
	value = _checked_extended_state(state, duplicated=True)
	coupling = _physical_coupling_jacobian(
		duration=duration,
		frequency=frequency,
	)
	value = _couple_physical_copies(
		value,
		duration=duration,
		frequency=frequency,
	)
	second = _flow_second_jacobian(dynamics, value, duration)
	value = _flow_second(dynamics, value, duration)
	first = _flow_first_jacobian(dynamics, value, duration)
	return np.asarray(first @ second @ coupling)


def _bm4_base_map(
	dynamics: GuidingCenterDynamics,
	duration: float,
	*,
	frequency: float,
) -> _AnalyticExtendedMap:
	"""Return the twelve-stage fourth-order full-state BM4 map on ``R^8``."""

	def map_state(candidate: np.ndarray) -> np.ndarray:
		value = _checked_extended_state(candidate, duplicated=True)
		for coefficient, order in zip(_BM4_STAGES, _BM4_ORDERS, strict=True):
			stage_duration = float(coefficient * duration)
			if int(order) == 0:
				value = _bm4_direct_map(
					dynamics,
					value,
					duration=stage_duration,
					frequency=frequency,
				)
			else:
				value = _bm4_adjoint_map(
					dynamics,
					value,
					duration=stage_duration,
					frequency=frequency,
				)
		return value

	def jacobian_state(candidate: np.ndarray) -> np.ndarray:
		value = _checked_extended_state(candidate, duplicated=True)
		total = _IDENTITY_8.copy()
		for coefficient, order in zip(_BM4_STAGES, _BM4_ORDERS, strict=True):
			stage_duration = float(coefficient * duration)
			if int(order) == 0:
				factor = _bm4_direct_jacobian(
					dynamics,
					value,
					duration=stage_duration,
					frequency=frequency,
				)
				value = _bm4_direct_map(
					dynamics,
					value,
					duration=stage_duration,
					frequency=frequency,
				)
			else:
				factor = _bm4_adjoint_jacobian(
					dynamics,
					value,
					duration=stage_duration,
					frequency=frequency,
				)
				value = _bm4_adjoint_map(
					dynamics,
					value,
					duration=stage_duration,
					frequency=frequency,
				)
			total = factor @ total
		return total

	return _AnalyticExtendedMap(
		map_state=map_state,
		jacobian_state=jacobian_state,
	)


@dataclass(frozen=True, slots=True)
class _FullProjectedStep:
	"""One converged full-diagonal projection and its base-map snapshots."""

	state: np.ndarray
	multiplier: np.ndarray
	internal_input: np.ndarray
	mapped: np.ndarray
	base_map: _AnalyticExtendedMap
	residual_jacobian: np.ndarray
	iterations: int
	residual_evaluations: int
	residual_norm: float


def _solve_full_projection(
	state: np.ndarray,
	base_map: _AnalyticExtendedMap,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver,
	context: str,
) -> _FullProjectedStep:
	"""Solve the four-component symmetric projection onto ``Z_1=Z_2``."""
	value = _checked_extended_state(state, duplicated=False)
	multiplier = np.zeros(4, dtype=float)
	threshold = absolute_tolerance + relative_tolerance * max(
		1.0,
		float(np.linalg.norm(value, ord=np.inf)),
	)
	evaluation_count = 0

	def evaluate(candidate: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
		nonlocal evaluation_count
		evaluation_count += 1
		unknown = np.asarray(candidate, dtype=float)
		internal_input = np.concatenate((value + unknown, value - unknown))
		mapped = np.asarray(base_map.map_state(internal_input), dtype=float)
		if mapped.shape != (8,) or not np.all(np.isfinite(mapped)):
			raise ValueError("The full duplicated base map returned an invalid state.")
		residual = mapped[:4] - mapped[4:] + 2.0 * unknown
		return residual, (internal_input, mapped)

	def analytic_residual_jacobian(internal_input: np.ndarray) -> np.ndarray:
		return np.asarray(
			_COPY_DIFFERENCE
			@ base_map.jacobian_state(np.asarray(internal_input))
			@ _ANTIDIAGONAL_EMBEDDING
			+ 2.0 * _IDENTITY_4,
		)

	if nonlinear_solver == "broyden":
		initial_jacobian = analytic_residual_jacobian(
			np.concatenate((value, value))
		)
		result = _solve_broyden(
			evaluate,
			multiplier,
			initial_jacobian,
			tolerance=threshold,
			max_iterations=max_iterations,
			context=context,
		)
		internal_input, mapped = result.payload
		corrected_first = mapped[:4] + result.unknown
		corrected_second = mapped[4:] - result.unknown
		return _FullProjectedStep(
			state=np.asarray((corrected_first + corrected_second) / 2.0),
			multiplier=np.asarray(result.unknown).copy(),
			internal_input=np.asarray(internal_input).copy(),
			mapped=np.asarray(mapped).copy(),
			base_map=base_map,
			residual_jacobian=analytic_residual_jacobian(internal_input),
			iterations=result.iterations,
			residual_evaluations=result.residual_evaluations,
			residual_norm=float(np.linalg.norm(result.residual, ord=np.inf)),
		)
	if nonlinear_solver != "newton":
		raise ValueError("Unknown nonlinear solver for the full extended projection.")

	for iteration in range(max_iterations + 1):
		residual, payload = evaluate(multiplier)
		internal_input, mapped = payload
		residual_norm = float(np.linalg.norm(residual, ord=np.inf))
		if residual_norm <= threshold:
			corrected_first = mapped[:4] + multiplier
			corrected_second = mapped[4:] - multiplier
			return _FullProjectedStep(
				state=np.asarray((corrected_first + corrected_second) / 2.0),
				multiplier=multiplier.copy(),
				internal_input=np.asarray(internal_input).copy(),
				mapped=np.asarray(mapped).copy(),
				base_map=base_map,
				residual_jacobian=analytic_residual_jacobian(internal_input),
				iterations=iteration,
				residual_evaluations=evaluation_count,
				residual_norm=residual_norm,
			)
		if iteration == max_iterations:
			break
		jacobian = analytic_residual_jacobian(internal_input)
		try:
			correction = np.linalg.solve(jacobian, residual)
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(f"The projection Jacobian is singular for {context}.") from exc
		multiplier = multiplier - correction

	raise RuntimeError(
		f"{context} did not converge: residual {residual_norm:.3e} exceeds "
		f"{threshold:.3e} after {max_iterations} iterations."
	)


@dataclass(frozen=True, slots=True)
class _AcceptedFullSubstep:
	"""One signed full-state ABBA solve inside a triple-jump step."""

	start_state: np.ndarray
	duration: float
	result: _FullProjectedStep


@dataclass(frozen=True, slots=True)
class _FullMethodStep:
	"""One complete method step and the projected base solves it contains."""

	state: np.ndarray
	substeps: tuple[_AcceptedFullSubstep, ...]


def _projected_substep_jacobian(result: _FullProjectedStep) -> np.ndarray:
	"""Differentiate one converged full projection by the implicit-function theorem."""
	base_jacobian = result.base_map.jacobian_state(result.internal_input)
	state_jacobian = _COPY_DIFFERENCE @ base_jacobian @ _DIAGONAL_EMBEDDING
	try:
		multiplier_jacobian = -np.linalg.solve(
			result.residual_jacobian,
			state_jacobian,
		)
	except np.linalg.LinAlgError as exc:
		raise RuntimeError("The accepted projection Jacobian is singular.") from exc
	return np.asarray(
		_COPY_AVERAGE
		@ base_jacobian
		@ (
			_DIAGONAL_EMBEDDING
			+ _ANTIDIAGONAL_EMBEDDING @ multiplier_jacobian
		)
	)


def _method_step_jacobian(result: _FullMethodStep) -> np.ndarray:
	"""Compose the analytic projected tangents in accepted flow order."""
	total = _IDENTITY_4.copy()
	for accepted in result.substeps:
		total = _projected_substep_jacobian(accepted.result) @ total
	return total


def _solve_method_step(
	method: _FullyExtendedImplicitMethod,
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	duration: float,
) -> _FullMethodStep:
	"""Solve one ABBA, ABBA4, or BM4 full-state projected step."""
	value = _checked_extended_state(state, duplicated=False)
	if method._variant == "abba4":
		current = value
		accepted: list[_AcceptedFullSubstep] = []
		for coefficient in _ABBA4_COEFFICIENTS:
			substep_duration = float(coefficient * duration)
			base_map = _abba_base_map(dynamics, substep_duration)
			result = _solve_full_projection(
				current,
				base_map,
				absolute_tolerance=method.newton_absolute_tolerance,
				relative_tolerance=method.newton_relative_tolerance,
				max_iterations=method.newton_max_iterations,
				nonlinear_solver=method.nonlinear_solver,
				context=(
					f"ABBA4FullyExtendedImplicit substep at t={current[2]:.16g} "
					f"with duration={substep_duration:.16g}"
				),
			)
			accepted.append(
				_AcceptedFullSubstep(
					start_state=current.copy(),
					duration=substep_duration,
					result=result,
				)
			)
			current = result.state
		return _FullMethodStep(state=current, substeps=tuple(accepted))

	if method._variant == "bm4":
		frequency = getattr(method, "coupling_frequency", None)
		if frequency is None:
			raise RuntimeError(
				"The fully extended BM4 variant requires `coupling_frequency`."
			)
		base_map = _bm4_base_map(
			dynamics,
			duration,
			frequency=float(frequency),
		)
		map_name = "fully_extended_bm4_cycle"
	else:
		base_map = _abba_base_map(dynamics, duration)
		map_name = "fully_extended_abba_map"
	result = _solve_full_projection(
		value,
		base_map,
		absolute_tolerance=method.newton_absolute_tolerance,
		relative_tolerance=method.newton_relative_tolerance,
		max_iterations=method.newton_max_iterations,
		nonlinear_solver=method.nonlinear_solver,
		context=(
			f"{map_name} at t={value[2]:.16g} with duration={duration:.16g}"
		),
	)
	return _FullMethodStep(
		state=result.state,
		substeps=(
			_AcceptedFullSubstep(
				start_state=value.copy(),
				duration=duration,
				result=result,
			),
		),
	)


def _base_observation(
	accepted: _AcceptedFullSubstep,
	*,
	map_name: str,
) -> FullyExtendedBaseMap:
	"""Build one immutable accepted ``R^8`` base-map snapshot."""
	result = accepted.result
	return FullyExtendedBaseMap(
		map_name=map_name,
		start_time=float(accepted.start_state[2]),
		duration=float(accepted.duration),
		state_before=result.internal_input.copy(),
		state_after=result.mapped.copy(),
		map_state=result.base_map.map_state,
		jacobian_state=result.base_map.jacobian_state,
		projection_multiplier=result.multiplier.copy(),
		residual_jacobian=result.residual_jacobian.copy(),
	)


def _integrate_fully_extended(
	method: _FullyExtendedImplicitMethod,
	problem: InitialValueProblem,
	request: SimulationRequest,
) -> IntegrationData:
	"""Integrate a one-particle GC problem through the physical ``R^4`` state."""
	method_name = type(method).__name__
	if not isinstance(problem.dynamics, GuidingCenterDynamics):
		raise TypeError(f"{method_name} requires GuidingCenterDynamics.")
	physical_initial = np.asarray(problem.initial_state, dtype=float)
	if physical_initial.shape != (2,):
		raise ValueError(f"{method_name} requires exactly one GC particle.")
	dynamics = problem.dynamics
	initial_extended = np.concatenate(
		(physical_initial, (float(request.t_span[0]), 0.0))
	)
	iteration_counts: list[int] = []
	residual_evaluations: list[int] = []
	residual_norms: list[float] = []
	projection_norms: list[float] = []

	def advance(
		time: float,
		state: np.ndarray,
		step: float,
		step_index: int,
		observe: bool,
	) -> np.ndarray:
		value = _checked_extended_state(state, duplicated=False)
		tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(time))
		if not np.isclose(float(value[2]), time, rtol=0.0, atol=float(tolerance)):
			raise RuntimeError("The internal and integration-grid times diverged.")

		def map_state(candidate: np.ndarray) -> np.ndarray:
			return _solve_method_step(method, dynamics, candidate, step).state

		result = _solve_method_step(method, dynamics, value, step)
		if observe:
			iterations = sum(item.result.iterations for item in result.substeps)
			evaluations = sum(
				item.result.residual_evaluations for item in result.substeps
			)
			worst_residual = max(item.result.residual_norm for item in result.substeps)
			max_multiplier = max(
				float(np.linalg.norm(item.result.multiplier, ord=np.inf))
				for item in result.substeps
			)
			iteration_counts.append(iterations)
			residual_evaluations.append(evaluations)
			residual_norms.append(worst_residual)
			projection_norms.append(max_multiplier)
			if method.step_observer is not None:
				base_name = (
					"fully_extended_bm4_cycle"
					if method._variant == "bm4"
					else "fully_extended_abba_map"
				)
				base_maps = tuple(
					_base_observation(item, map_name=base_name)
					for item in result.substeps
				)
				method.step_observer(
					FullyExtendedImplicitIntegrationStep(
						dynamics_name=type(dynamics).__name__,
						method_name=method_name,
						step_index=step_index,
						start_time=time,
						time=time + step,
						duration=step,
						state_before=value.copy(),
						state_after=result.state.copy(),
						map_state=map_state,
						dynamics=dynamics,
						formulation_name="fully_duplicated_z_t_k_projection",
						nonlinear_solver=method.nonlinear_solver,
						newton_iterations=iterations,
						residual_evaluations=evaluations,
						newton_residual_norm=worst_residual,
						newton_tolerance=(
							method.newton_absolute_tolerance
							+ method.newton_relative_tolerance
							* max(1.0, float(np.linalg.norm(value, ord=np.inf)))
						),
						projection_multiplier_norm=max_multiplier,
						multiplier=result.substeps[-1].result.multiplier.copy(),
						jacobian=_method_step_jacobian(result),
						base_maps=base_maps,
					)
				)
		return result.state

	extended_history, step_count = integrate_fixed_grid(
		initial_extended,
		request,
		advance,
		progress=method.progress,
		label=method_name,
	)
	physical_hamiltonian = np.asarray(
		[
			float(
				np.asarray(
					dynamics.hamiltonian(
						float(extended_history[2, index]),
						extended_history[:2, index],
					)
				).reshape(-1)[0]
			)
			for index in range(extended_history.shape[1])
		]
	)
	generalized_energy = physical_hamiltonian + extended_history[3]
	diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
		"step_count": step_count,
		"nonlinear_solver": method.nonlinear_solver,
		"newton_iterations": np.asarray(iteration_counts, dtype=int),
		"residual_evaluations": np.asarray(residual_evaluations, dtype=int),
		"newton_residual_norms": np.asarray(residual_norms, dtype=float),
		"projection_multiplier_norms": np.asarray(projection_norms, dtype=float),
		"newton_absolute_tolerance": method.newton_absolute_tolerance,
		"newton_relative_tolerance": method.newton_relative_tolerance,
		"newton_max_iterations": method.newton_max_iterations,
		"projection_jacobian": "analytic_stage_product",
		"projection_formulation": "full_state_multiplier",
		"state_extension": "fully_extended",
		"extended_time": np.asarray(extended_history[2]),
		"extended_momentum": np.asarray(extended_history[3]),
		"extended_momentum_normalization": "direct_k",
		"physical_hamiltonian": physical_hamiltonian,
		"generalized_energy": generalized_energy,
		"generalized_energy_error": generalized_energy - generalized_energy[0],
	}
	if method._variant == "bm4":
		frequency = getattr(method, "coupling_frequency", None)
		if frequency is None:
			raise RuntimeError(
				"The fully extended BM4 variant requires `coupling_frequency`."
			)
		diagnostics["coupling_frequency"] = float(frequency)
	return IntegrationData(
		t=request.output_times,
		states=np.asarray(extended_history[:2]),
		diagnostics=diagnostics,
	)


@dataclass(frozen=True, slots=True)
class _FullyExtendedImplicitMethod:
	"""Shared configuration for full ``(z,t,k)`` duplication and projection."""

	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 20
	nonlinear_solver: NonlinearSolver = "newton"
	progress: bool = False
	step_observer: StepObserver | None = None

	_variant: ClassVar[_Variant] = "abba"

	def __post_init__(self) -> None:
		"""Validate nonlinear controls shared by all full-state variants."""
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
		"""Integrate one GC problem with complete extended-state projection."""
		return _integrate_fully_extended(self, problem, request)


__all__: list[str] = []
