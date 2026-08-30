"""Implicit full-diagonal projection after duplicating ``(z, t, k)``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, TypeAlias

import numpy as np

from dynamics import GuidingCenterDynamics

from .._fixed import integrate_fixed_grid
from .._result import IntegrationData
from ..formulations import gc_coupling_matrix
from ..observation import (
	FullyExtendedBaseMap,
	FullyExtendedImplicitIntegrationStep,
	IntegrationStep,
	StepObserver,
)
from ..problem import InitialValueProblem
from ..request import SimulationRequest
from ._abba_coefficients import _ABBA4_COEFFICIENTS, _ABBA6_COEFFICIENTS
from ._nonlinear import (
	NonlinearSolver,
	_solve_broyden,
	_validate_nonlinear_solver,
)
from .bm4._core import _BM4_ORDERS, _BM4_STAGES


_ExtendedMap = Callable[[np.ndarray], np.ndarray]
_ExtendedJacobian = Callable[[np.ndarray], np.ndarray]
ProjectionFormulation: TypeAlias = Literal[
	"reduced_multiplier",
	"simultaneous_state_multiplier",
]
_Variant = Literal["bm4"]
_ABBAVariant = Literal[
	"abba",
	"abba4",
	"abba4_single_projection",
	"abba6",
]
_IDENTITY_4 = np.eye(4)
_IDENTITY_8 = np.eye(8)
_DIAGONAL_EMBEDDING = np.vstack((_IDENTITY_4, _IDENTITY_4))
_ANTIDIAGONAL_EMBEDDING = np.vstack((_IDENTITY_4, -_IDENTITY_4))
_COPY_DIFFERENCE = np.hstack((_IDENTITY_4, -_IDENTITY_4))
_COPY_AVERAGE = 0.5 * np.hstack((_IDENTITY_4, _IDENTITY_4))


def _validate_abba_projection_formulation(
	value: str,
) -> ProjectionFormulation:
	"""Validate the ABBA projection axis without importing its public package."""
	if value == "reduced_multiplier":
		return "reduced_multiplier"
	if value == "simultaneous_state_multiplier":
		return "simultaneous_state_multiplier"
	raise ValueError(
		"`projection_formulation` must be 'reduced_multiplier' or "
		"'simultaneous_state_multiplier'."
	)


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


def _synchronized_extended_time(
	state: np.ndarray,
	expected_time: float,
	*,
	context: str,
) -> np.ndarray:
	"""Validate and pin the exactly solvable time coordinate to its grid value.

	The extended Hamiltonian has ``dt/ds = 1``.  Its accepted time is therefore
	known analytically, while repeated floating-point stage additions can otherwise
	accumulate enough roundoff to trip a long-integration consistency guard.
	"""
	value = _checked_extended_state(state, duplicated=False)
	expected = float(expected_time)
	if not np.isfinite(expected):
		raise ValueError("The expected extended time must be finite.")
	tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(expected))
	if not np.isclose(
		float(value[2]),
		expected,
		rtol=0.0,
		atol=float(tolerance),
	):
		raise RuntimeError(f"{context} and integration-grid times diverged.")
	value[2] = expected
	return value


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

	Mixed and second time derivatives are evaluated through the effective
	potential contract. This supports static means and arbitrary positive-frequency
	HDF5 modes in addition to the normalized single-frequency potential.
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
	h_tt = float(potential.evaluate(time, x, y, dt=2)[0])
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


def _composed_abba_base_map(
	dynamics: GuidingCenterDynamics,
	duration: float,
	coefficients: np.ndarray,
) -> _AnalyticExtendedMap:
	"""Compose unprojected full-state ABBA maps without diagonal projection."""
	composition = np.asarray(coefficients, dtype=float)
	if (
		composition.ndim != 1
		or composition.size == 0
		or not np.all(np.isfinite(composition))
	):
		raise ValueError("ABBA composition coefficients must be finite and non-empty.")
	base_maps = tuple(
		_abba_base_map(dynamics, float(coefficient * duration))
		for coefficient in composition
	)

	def map_state(candidate: np.ndarray) -> np.ndarray:
		value = _checked_extended_state(candidate, duplicated=True)
		for base_map in base_maps:
			value = np.asarray(base_map.map_state(value), dtype=float)
		return value

	def jacobian_state(candidate: np.ndarray) -> np.ndarray:
		value = _checked_extended_state(candidate, duplicated=True)
		total = _IDENTITY_8.copy()
		for base_map in base_maps:
			factor = np.asarray(base_map.jacobian_state(value), dtype=float)
			value = np.asarray(base_map.map_state(value), dtype=float)
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
	residual_jacobian: np.ndarray | None
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


def _full_reduced_residual_jacobian(
	base_map: _AnalyticExtendedMap,
	internal_input: np.ndarray,
) -> np.ndarray:
	"""Differentiate the four-component full-state multiplier residual."""
	return np.asarray(
		_COPY_DIFFERENCE
		@ base_map.jacobian_state(np.asarray(internal_input, dtype=float))
		@ _ANTIDIAGONAL_EMBEDDING
		+ 2.0 * _IDENTITY_4
	)


def _centered_map_jacobian(
	map_state: _ExtendedMap,
	state: np.ndarray,
) -> np.ndarray:
	"""Differentiate one finite state map without requiring GC Hessians."""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The differentiated state must be a finite vector.")
	scale = float(np.cbrt(np.finfo(float).eps))
	jacobian = np.empty((value.size, value.size), dtype=float)
	for column in range(value.size):
		increment = scale * max(1.0, abs(float(value[column])))
		perturbation = np.zeros_like(value)
		perturbation[column] = increment
		forward = np.asarray(map_state(value + perturbation), dtype=float)
		backward = np.asarray(map_state(value - perturbation), dtype=float)
		if forward.shape != value.shape or backward.shape != value.shape:
			raise ValueError("The differentiated map changed the state shape.")
		jacobian[:, column] = (forward - backward) / (2.0 * increment)
	if not np.all(np.isfinite(jacobian)):
		raise ValueError("The numerical map Jacobian contains non-finite values.")
	return jacobian


def _base_map_jacobian(
	base_map: _AnalyticExtendedMap,
	state: np.ndarray,
) -> np.ndarray:
	"""Use the stage product when available, otherwise a diagnostic fallback."""
	try:
		return np.asarray(base_map.jacobian_state(state), dtype=float)
	except ValueError as exc:
		if "Analytic full-state Jacobians require interpolation_order >= 3" not in str(
			exc
		):
			raise
	return _centered_map_jacobian(base_map.map_state, state)


def _solve_abba_full_reduced_projection(
	state: np.ndarray,
	base_map: _AnalyticExtendedMap,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver,
	context: str,
	require_tangent: bool,
) -> _FullProjectedStep:
	"""Solve the reduced ``R^4`` projection without differentiating Broyden."""
	value = _checked_extended_state(state, duplicated=False)
	multiplier = np.zeros(4, dtype=float)
	threshold = absolute_tolerance + relative_tolerance * max(
		1.0,
		float(np.linalg.norm(value, ord=np.inf)),
	)

	def evaluate(candidate: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
		unknown = np.asarray(candidate, dtype=float)
		internal_input = np.concatenate((value + unknown, value - unknown))
		mapped = np.asarray(base_map.map_state(internal_input), dtype=float)
		if mapped.shape != (8,) or not np.all(np.isfinite(mapped)):
			raise ValueError("The full duplicated ABBA map returned an invalid state.")
		residual = mapped[:4] - mapped[4:] + 2.0 * unknown
		return residual, (internal_input, mapped)

	def accepted(
		unknown: np.ndarray,
		internal_input: np.ndarray,
		mapped: np.ndarray,
		*,
		iterations: int,
		residual_evaluations: int,
		residual_norm: float,
	) -> _FullProjectedStep:
		corrected_first = mapped[:4] + unknown
		corrected_second = mapped[4:] - unknown
		residual_jacobian = (
			_full_reduced_residual_jacobian(base_map, internal_input)
			if require_tangent
			else None
		)
		return _FullProjectedStep(
			state=np.asarray((corrected_first + corrected_second) / 2.0),
			multiplier=np.asarray(unknown).copy(),
			internal_input=np.asarray(internal_input).copy(),
			mapped=np.asarray(mapped).copy(),
			base_map=base_map,
			residual_jacobian=residual_jacobian,
			iterations=iterations,
			residual_evaluations=residual_evaluations,
			residual_norm=residual_norm,
		)

	if nonlinear_solver == "broyden":
		# D Psi is deliberately not evaluated here. Four times the identity is the
		# exact residual Jacobian for the identity-map linearization and provides a
		# deterministic residual-only Broyden start.
		result = _solve_broyden(
			evaluate,
			multiplier,
			4.0 * _IDENTITY_4,
			tolerance=threshold,
			max_iterations=max_iterations,
			context=context,
		)
		internal_input, mapped = result.payload
		return accepted(
			result.unknown,
			internal_input,
			mapped,
			iterations=result.iterations,
			residual_evaluations=result.residual_evaluations,
			residual_norm=float(np.linalg.norm(result.residual, ord=np.inf)),
		)
	if nonlinear_solver != "newton":
		raise ValueError("Unknown nonlinear solver for full-state ABBA projection.")

	for iteration in range(max_iterations + 1):
		residual, payload = evaluate(multiplier)
		internal_input, mapped = payload
		residual_norm = float(np.linalg.norm(residual, ord=np.inf))
		if residual_norm <= threshold:
			return accepted(
				multiplier,
				internal_input,
				mapped,
				iterations=iteration,
				residual_evaluations=iteration + 1,
				residual_norm=residual_norm,
			)
		if iteration == max_iterations:
			break
		jacobian = _full_reduced_residual_jacobian(base_map, internal_input)
		try:
			correction = np.linalg.solve(jacobian, residual)
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				f"The reduced full-state projection Jacobian is singular for {context}."
			) from exc
		multiplier = multiplier - correction

	raise RuntimeError(
		f"{context} did not converge: reduced residual {residual_norm:.3e} "
		f"exceeds {threshold:.3e} after {max_iterations} iterations."
	)


def _solve_abba_full_simultaneous_projection(
	state: np.ndarray,
	base_map: _AnalyticExtendedMap,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver,
	context: str,
	require_tangent: bool,
) -> _FullProjectedStep:
	"""Solve the coupled final-copy and multiplier equations in ``R^12``."""
	value = _checked_extended_state(state, duplicated=False)
	threshold = absolute_tolerance + relative_tolerance * max(
		1.0,
		float(np.linalg.norm(value, ord=np.inf)),
	)
	multiplier = np.zeros(4, dtype=float)
	internal_input = np.concatenate((value, value))
	mapped = np.asarray(base_map.map_state(internal_input), dtype=float)
	if mapped.shape != (8,) or not np.all(np.isfinite(mapped)):
		raise ValueError("The full duplicated ABBA map returned an invalid state.")
	first_output = mapped[:4].copy()
	second_output = mapped[4:].copy()

	def evaluate_unknown(
		unknown: np.ndarray,
	) -> tuple[
		np.ndarray,
		tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
	]:
		first = np.asarray(unknown[:4], dtype=float)
		second = np.asarray(unknown[4:8], dtype=float)
		candidate_multiplier = np.asarray(unknown[8:], dtype=float)
		candidate_input = np.concatenate(
			(value + candidate_multiplier, value - candidate_multiplier)
		)
		candidate_mapped = np.asarray(base_map.map_state(candidate_input), dtype=float)
		if candidate_mapped.shape != (8,) or not np.all(np.isfinite(candidate_mapped)):
			raise ValueError("The full duplicated ABBA map returned an invalid state.")
		residual = np.concatenate(
			(
				first - candidate_multiplier - candidate_mapped[:4],
				second + candidate_multiplier - candidate_mapped[4:],
				first - second,
			)
		)
		return residual, (
			candidate_input,
			candidate_mapped,
			first,
			second,
			candidate_multiplier,
		)

	def accepted(
		candidate_input: np.ndarray,
		candidate_mapped: np.ndarray,
		first: np.ndarray,
		second: np.ndarray,
		candidate_multiplier: np.ndarray,
		*,
		iterations: int,
		residual_evaluations: int,
		residual_norm: float,
	) -> _FullProjectedStep:
		residual_jacobian = (
			_full_reduced_residual_jacobian(base_map, candidate_input)
			if require_tangent
			else None
		)
		return _FullProjectedStep(
			state=np.asarray((first + second) / 2.0),
			multiplier=np.asarray(candidate_multiplier).copy(),
			internal_input=np.asarray(candidate_input).copy(),
			mapped=np.asarray(candidate_mapped).copy(),
			base_map=base_map,
			residual_jacobian=residual_jacobian,
			iterations=iterations,
			residual_evaluations=residual_evaluations,
			residual_norm=residual_norm,
		)

	initial_unknown = np.concatenate((first_output, second_output, multiplier))
	initial_residual = np.concatenate((np.zeros(8), first_output - second_output))
	if nonlinear_solver == "broyden":
		# The identity-map approximation gives -(I + D Psi) N = -2 N.
		# It avoids every analytic-Jacobian call before or during Broyden.
		initial_jacobian = np.block(
			[
				[_IDENTITY_8, -2.0 * _ANTIDIAGONAL_EMBEDDING],
				[_COPY_DIFFERENCE, np.zeros((4, 4), dtype=float)],
			]
		)
		result = _solve_broyden(
			evaluate_unknown,
			initial_unknown,
			initial_jacobian,
			tolerance=threshold,
			max_iterations=max_iterations,
			context=context,
			initial_evaluation=(
				initial_residual,
				(internal_input, mapped, first_output, second_output, multiplier),
			),
		)
		candidate_input, candidate_mapped, first, second, candidate_multiplier = (
			result.payload
		)
		return accepted(
			candidate_input,
			candidate_mapped,
			first,
			second,
			candidate_multiplier,
			iterations=result.iterations,
			residual_evaluations=result.residual_evaluations,
			residual_norm=float(np.linalg.norm(result.residual, ord=np.inf)),
		)
	if nonlinear_solver != "newton":
		raise ValueError("Unknown nonlinear solver for full-state ABBA projection.")

	for iteration in range(max_iterations + 1):
		if iteration == 0:
			residual = initial_residual
			payload = (
				internal_input,
				mapped,
				first_output,
				second_output,
				multiplier,
			)
		else:
			residual, payload = evaluate_unknown(
				np.concatenate((first_output, second_output, multiplier))
			)
		internal_input, mapped, first_output, second_output, multiplier = payload
		residual_norm = float(np.linalg.norm(residual, ord=np.inf))
		if residual_norm <= threshold:
			return accepted(
				internal_input,
				mapped,
				first_output,
				second_output,
				multiplier,
				iterations=iteration,
				residual_evaluations=iteration + 1,
				residual_norm=residual_norm,
			)
		if iteration == max_iterations:
			break

		base_jacobian = np.asarray(base_map.jacobian_state(internal_input), dtype=float)
		top_right = -(_IDENTITY_8 + base_jacobian) @ _ANTIDIAGONAL_EMBEDDING
		newton_jacobian = np.block(
			[
				[_IDENTITY_8, top_right],
				[_COPY_DIFFERENCE, np.zeros((4, 4), dtype=float)],
			]
		)
		try:
			increment = np.linalg.solve(newton_jacobian, -residual)
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				f"The simultaneous full-state projection Jacobian is singular for {context}."
			) from exc
		first_output = first_output + increment[:4]
		second_output = second_output + increment[4:8]
		multiplier = multiplier + increment[8:]

	raise RuntimeError(
		f"{context} did not converge: simultaneous residual {residual_norm:.3e} "
		f"exceeds {threshold:.3e} after {max_iterations} iterations."
	)


def _solve_abba_full_projection(
	state: np.ndarray,
	base_map: _AnalyticExtendedMap,
	*,
	projection_formulation: ProjectionFormulation,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver,
	context: str,
	require_tangent: bool,
) -> _FullProjectedStep:
	"""Dispatch one full-state ABBA projection by nonlinear formulation."""
	formulation = _validate_abba_projection_formulation(projection_formulation)
	if formulation == "reduced_multiplier":
		return _solve_abba_full_reduced_projection(
			state,
			base_map,
			absolute_tolerance=absolute_tolerance,
			relative_tolerance=relative_tolerance,
			max_iterations=max_iterations,
			nonlinear_solver=nonlinear_solver,
			context=context,
			require_tangent=require_tangent,
		)
	return _solve_abba_full_simultaneous_projection(
		state,
		base_map,
		absolute_tolerance=absolute_tolerance,
		relative_tolerance=relative_tolerance,
		max_iterations=max_iterations,
		nonlinear_solver=nonlinear_solver,
		context=context,
		require_tangent=require_tangent,
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


class _ABBAFullyExtendedMethod(Protocol):
	"""Configuration consumed by the parameterized full-state ABBA runtime."""

	@property
	def newton_absolute_tolerance(self) -> float: ...

	@property
	def newton_relative_tolerance(self) -> float: ...

	@property
	def newton_max_iterations(self) -> int: ...

	@property
	def nonlinear_solver(self) -> NonlinearSolver: ...

	@property
	def progress(self) -> bool: ...

	@property
	def step_observer(self) -> StepObserver | None: ...


class _ABBAFullyExtendedMidpointMethod(Protocol):
	"""Configuration consumed by the full-state arithmetic-mean runtime."""

	@property
	def progress(self) -> bool: ...

	@property
	def step_observer(self) -> StepObserver | None: ...


def _abba_variant_coefficients(variant: _ABBAVariant) -> np.ndarray:
	"""Return signed ABBA durations for one public full-state variant."""
	if variant == "abba":
		return np.asarray((1.0,), dtype=float)
	if variant in ("abba4", "abba4_single_projection"):
		return _ABBA4_COEFFICIENTS
	if variant == "abba6":
		return _ABBA6_COEFFICIENTS
	raise ValueError(f"Unknown fully extended ABBA variant: {variant!r}.")


def _solve_abba_fully_extended_step(
	method: _ABBAFullyExtendedMethod,
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	duration: float,
	*,
	variant: _ABBAVariant,
	projection_formulation: ProjectionFormulation,
	require_tangent: bool,
) -> _FullMethodStep:
	"""Solve one parameterized fully extended ABBA method step."""
	value = _checked_extended_state(state, duplicated=False)
	coefficients = _abba_variant_coefficients(variant)
	method_name = type(method).__name__

	if variant == "abba4_single_projection":
		base_map = _composed_abba_base_map(dynamics, duration, coefficients)
		result = _solve_abba_full_projection(
			value,
			base_map,
			projection_formulation=projection_formulation,
			absolute_tolerance=method.newton_absolute_tolerance,
			relative_tolerance=method.newton_relative_tolerance,
			max_iterations=method.newton_max_iterations,
			nonlinear_solver=method.nonlinear_solver,
			context=(
				f"{method_name} fully extended outer projection at "
				f"t={value[2]:.16g} with duration={duration:.16g}"
			),
			require_tangent=require_tangent,
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

	current = value
	accepted: list[_AcceptedFullSubstep] = []
	for substep_index, coefficient in enumerate(coefficients):
		substep_duration = float(coefficient * duration)
		base_map = _abba_base_map(dynamics, substep_duration)
		result = _solve_abba_full_projection(
			current,
			base_map,
			projection_formulation=projection_formulation,
			absolute_tolerance=method.newton_absolute_tolerance,
			relative_tolerance=method.newton_relative_tolerance,
			max_iterations=method.newton_max_iterations,
			nonlinear_solver=method.nonlinear_solver,
			context=(
				f"{method_name} fully extended substep {substep_index + 1} "
				f"at t={current[2]:.16g} with duration={substep_duration:.16g}"
			),
			require_tangent=require_tangent,
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


def _projected_substep_jacobian(result: _FullProjectedStep) -> np.ndarray:
	"""Differentiate one converged full projection by the implicit-function theorem."""
	base_jacobian = _base_map_jacobian(result.base_map, result.internal_input)
	state_jacobian = _COPY_DIFFERENCE @ base_jacobian @ _DIAGONAL_EMBEDDING
	residual_jacobian = result.residual_jacobian
	if residual_jacobian is None:
		residual_jacobian = np.asarray(
			_COPY_DIFFERENCE @ base_jacobian @ _ANTIDIAGONAL_EMBEDDING
			+ 2.0 * _IDENTITY_4
		)
	try:
		multiplier_jacobian = -np.linalg.solve(
			residual_jacobian,
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
	"""Solve one legacy fully extended BM4 step."""
	value = _checked_extended_state(state, duplicated=False)
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
	result = _solve_full_projection(
		value,
		base_map,
		absolute_tolerance=method.newton_absolute_tolerance,
		relative_tolerance=method.newton_relative_tolerance,
		max_iterations=method.newton_max_iterations,
		nonlinear_solver=method.nonlinear_solver,
		context=(
			"fully_extended_bm4_cycle at "
			f"t={value[2]:.16g} with duration={duration:.16g}"
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
	residual_jacobian = result.residual_jacobian

	def jacobian_state(state: np.ndarray) -> np.ndarray:
		"""Differentiate this base map with an observer-only numerical fallback."""
		return _base_map_jacobian(result.base_map, state)

	if residual_jacobian is None:
		residual_jacobian = np.asarray(
			_COPY_DIFFERENCE
			@ jacobian_state(result.internal_input)
			@ _ANTIDIAGONAL_EMBEDDING
			+ 2.0 * _IDENTITY_4
		)
	return FullyExtendedBaseMap(
		map_name=map_name,
		start_time=float(accepted.start_state[2]),
		duration=float(accepted.duration),
		state_before=result.internal_input.copy(),
		state_after=result.mapped.copy(),
		map_state=result.base_map.map_state,
		jacobian_state=jacobian_state,
		projection_multiplier=result.multiplier.copy(),
		residual_jacobian=residual_jacobian.copy(),
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
		value = _synchronized_extended_time(
			state,
			time,
			context="The internal state",
		)

		def map_state(candidate: np.ndarray) -> np.ndarray:
			candidate_value = _checked_extended_state(candidate, duplicated=False)
			mapped = _solve_method_step(
				method,
				dynamics,
				candidate_value,
				step,
			).state
			return _synchronized_extended_time(
				mapped,
				float(candidate_value[2] + step),
				context="The fully extended map",
			)

		result = _solve_method_step(method, dynamics, value, step)
		expected_time = time + step
		accepted_state = _synchronized_extended_time(
			result.state,
			expected_time,
			context="The fully extended map",
		)
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
				base_maps = tuple(
					_base_observation(
						item,
						map_name="fully_extended_bm4_cycle",
					)
					for item in result.substeps
				)
				method.step_observer(
					FullyExtendedImplicitIntegrationStep(
							dynamics_name=type(dynamics).__name__,
							method_name=method_name,
							step_index=step_index,
							start_time=time,
							time=expected_time,
							duration=step,
							state_before=value.copy(),
							state_after=accepted_state.copy(),
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
		return accepted_state

	extended_history, step_count = integrate_fixed_grid(
		initial_extended,
		request,
		advance,
		progress=method.progress,
		label=method_name,
	)
	extended_history[2] = request.output_times
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


def _fully_extended_energy_diagnostics(
	dynamics: GuidingCenterDynamics,
	extended_history: np.ndarray,
) -> dict[str, np.ndarray | str]:
	"""Return direct-``k`` energy histories shared by full ABBA runtimes."""
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
	return {
		"extended_time": np.asarray(extended_history[2]),
		"extended_momentum": np.asarray(extended_history[3]),
		"extended_momentum_normalization": "direct_k",
		"physical_hamiltonian": physical_hamiltonian,
		"generalized_energy": generalized_energy,
		"generalized_energy_error": generalized_energy - generalized_energy[0],
	}


def _integrate_abba_fully_extended(
	method: _ABBAFullyExtendedMethod,
	problem: InitialValueProblem,
	request: SimulationRequest,
	*,
	variant: _ABBAVariant,
	projection_formulation: ProjectionFormulation,
) -> IntegrationData:
	"""Integrate one ABBA variant with complete ``(z,t,k)`` duplication."""
	method_name = type(method).__name__
	formulation = _validate_abba_projection_formulation(projection_formulation)
	nonlinear_solver = _validate_nonlinear_solver(method.nonlinear_solver)
	if not isinstance(problem.dynamics, GuidingCenterDynamics):
		raise TypeError(f"{method_name} requires GuidingCenterDynamics.")
	physical_initial = np.asarray(problem.initial_state, dtype=float)
	if physical_initial.shape != (2,):
		raise ValueError(f"{method_name} requires exactly one GC particle.")
	dynamics = problem.dynamics
	initial_extended = np.concatenate(
		(physical_initial, (float(request.t_span[0]), 0.0))
	)
	if nonlinear_solver == "newton":
		# Validate the analytic tangent capability before a zero residual can make
		# Newton appear to succeed without ever forming its Jacobian.
		_extended_vector_field_jacobian(dynamics, initial_extended)
	coefficients = _abba_variant_coefficients(variant)
	projection_count = 1 if variant == "abba4_single_projection" else int(
		coefficients.size
	)
	projection_placement = (
		"around_complete_base_composition"
		if variant == "abba4_single_projection"
		else "after_each_abba_map"
	)
	iteration_counts: list[int] = []
	residual_evaluation_counts: list[int] = []
	residual_norms: list[float] = []
	tolerance_values: list[float] = []
	projection_norms: list[float] = []
	substep_iteration_rows: list[list[int]] = []
	substep_evaluation_rows: list[list[int]] = []
	substep_residual_rows: list[list[float]] = []
	substep_tolerance_rows: list[list[float]] = []
	substep_projection_rows: list[list[float]] = []

	def solve_step(
		state: np.ndarray,
		step: float,
		*,
		require_tangent: bool,
	) -> _FullMethodStep:
		return _solve_abba_fully_extended_step(
			method,
			dynamics,
			state,
			step,
			variant=variant,
			projection_formulation=formulation,
			require_tangent=require_tangent,
		)

	def advance(
		time: float,
		state: np.ndarray,
		step: float,
		step_index: int,
		observe: bool,
	) -> np.ndarray:
		value = _synchronized_extended_time(
			state,
			time,
			context="The internal state",
		)

		def map_state(candidate: np.ndarray) -> np.ndarray:
			candidate_value = _checked_extended_state(candidate, duplicated=False)
			mapped = solve_step(
				candidate_value,
				step,
				require_tangent=False,
			).state
			return _synchronized_extended_time(
				mapped,
				float(candidate_value[2] + step),
				context="The fully extended ABBA map",
			)

		# Broyden's nonlinear solve remains residual-only. If an observer requests
		# tangent data, it is assembled after convergence with an analytic stage
		# product when available and a centered diagnostic fallback otherwise.
		needs_tangent = bool(
			observe
			and method.step_observer is not None
			and nonlinear_solver == "newton"
		)
		result = solve_step(value, step, require_tangent=needs_tangent)
		expected_time = time + step
		accepted_state = _synchronized_extended_time(
			result.state,
			expected_time,
			context="The fully extended ABBA map",
		)
		if observe:
			iterations = [item.result.iterations for item in result.substeps]
			evaluations = [
				item.result.residual_evaluations for item in result.substeps
			]
			residuals = [item.result.residual_norm for item in result.substeps]
			multipliers = [
				float(np.linalg.norm(item.result.multiplier, ord=np.inf))
				for item in result.substeps
			]
			tolerances = [
				method.newton_absolute_tolerance
				+ method.newton_relative_tolerance
				* max(1.0, float(np.linalg.norm(item.start_state, ord=np.inf)))
				for item in result.substeps
			]
			worst_substep = int(
				np.argmax(np.asarray(residuals) / np.asarray(tolerances))
			)
			iteration_counts.append(sum(iterations))
			residual_evaluation_counts.append(sum(evaluations))
			residual_norms.append(residuals[worst_substep])
			tolerance_values.append(tolerances[worst_substep])
			projection_norms.append(max(multipliers))
			substep_iteration_rows.append(iterations)
			substep_evaluation_rows.append(evaluations)
			substep_residual_rows.append(residuals)
			substep_tolerance_rows.append(tolerances)
			substep_projection_rows.append(multipliers)
			if method.step_observer is not None:
				base_name = (
					"fully_extended_abba4_unprojected_composition"
					if variant == "abba4_single_projection"
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
							time=expected_time,
							duration=step,
							state_before=value.copy(),
							state_after=accepted_state.copy(),
						map_state=map_state,
						dynamics=dynamics,
						formulation_name=formulation,
						nonlinear_solver=nonlinear_solver,
						newton_iterations=sum(iterations),
						residual_evaluations=sum(evaluations),
						newton_residual_norm=residuals[worst_substep],
						newton_tolerance=tolerances[worst_substep],
						projection_multiplier_norm=max(multipliers),
						multiplier=result.substeps[-1].result.multiplier.copy(),
						jacobian=_method_step_jacobian(result),
						base_maps=base_maps,
					)
				)
		return accepted_state

	extended_history, step_count = integrate_fixed_grid(
		initial_extended,
		request,
		advance,
		progress=bool(method.progress),
		label=method_name,
	)
	extended_history[2] = request.output_times
	iterations = np.asarray(iteration_counts, dtype=int)
	evaluations = np.asarray(residual_evaluation_counts, dtype=int)
	residuals = np.asarray(residual_norms, dtype=float)
	tolerances = np.asarray(tolerance_values, dtype=float)
	projection_values = np.asarray(projection_norms, dtype=float)
	diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
		"step_count": step_count,
		"nonlinear_solver": nonlinear_solver,
		"nonlinear_iterations": iterations,
		"residual_evaluations": evaluations,
		"nonlinear_residual_norms": residuals,
		"nonlinear_tolerances": tolerances,
		"projection_multiplier_norms": projection_values,
		"nonlinear_absolute_tolerance": method.newton_absolute_tolerance,
		"nonlinear_relative_tolerance": method.newton_relative_tolerance,
		"nonlinear_max_iterations": method.newton_max_iterations,
		"newton_iterations": iterations,
		"newton_residual_norms": residuals,
		"newton_absolute_tolerance": method.newton_absolute_tolerance,
		"newton_relative_tolerance": method.newton_relative_tolerance,
		"newton_max_iterations": method.newton_max_iterations,
		"projection_jacobian": (
			"analytic_stage_product"
			if nonlinear_solver == "newton"
			or (
				method.step_observer is not None
				and dynamics.effective_potential.interpolation_order >= 3
			)
			else "centered_difference_observer_fallback"
			if method.step_observer is not None
			else "not_evaluated"
		),
		"projection_formulation": formulation,
		"state_extension": "fully_extended",
		"projection_placement": projection_placement,
		"implicit_substeps_per_step": projection_count,
		"nonlinear_solves_per_step": projection_count,
		"unprojected_abba_maps_per_step": int(coefficients.size),
		"composition_coefficients": coefficients.copy(),
		"substep_nonlinear_iterations": np.asarray(
			substep_iteration_rows,
			dtype=int,
		),
		"substep_residual_evaluations": np.asarray(
			substep_evaluation_rows,
			dtype=int,
		),
		"substep_nonlinear_residual_norms": np.asarray(
			substep_residual_rows,
			dtype=float,
		),
		"substep_nonlinear_tolerances": np.asarray(
			substep_tolerance_rows,
			dtype=float,
		),
		"substep_projection_multiplier_norms": np.asarray(
			substep_projection_rows,
			dtype=float,
		),
		"unprojected_abba_maps_per_residual_evaluation": (
			int(coefficients.size)
			if variant == "abba4_single_projection"
			else 1
		),
	}
	diagnostics.update(
		{
			"accepted_internal_state_dimension": 4,
			"base_splitting_state_dimension": 8,
			"observer_state_dimension": 4,
			"observer_state_kind": "accepted_internal_map",
			"nonlinear_unknown_dimension": (
				4 if formulation == "reduced_multiplier" else 12
			),
		}
	)
	diagnostics.update(_fully_extended_energy_diagnostics(dynamics, extended_history))
	return IntegrationData(
		t=request.output_times,
		states=np.asarray(extended_history[:2]),
		diagnostics=diagnostics,
	)


def _integrate_abba_fully_extended_midpoint(
	method: _ABBAFullyExtendedMidpointMethod,
	problem: InitialValueProblem,
	request: SimulationRequest,
) -> IntegrationData:
	"""Integrate full-state ABBA2 with arithmetic-mean diagonal projection."""
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
	copy_separation_norms: list[float] = []

	def midpoint_step(state: np.ndarray, step: float) -> tuple[np.ndarray, float]:
		value = _checked_extended_state(state, duplicated=False)
		mapped = np.asarray(
			_abba_base_map(dynamics, step).map_state(np.concatenate((value, value))),
			dtype=float,
		)
		accepted_state = _synchronized_extended_time(
			np.asarray((mapped[:4] + mapped[4:]) / 2.0),
			float(value[2] + step),
			context="The fully extended midpoint map",
		)
		return (
			accepted_state,
			float(np.linalg.norm(mapped[:4] - mapped[4:], ord=np.inf)),
		)

	def advance(
		time: float,
		state: np.ndarray,
		step: float,
		step_index: int,
		observe: bool,
	) -> np.ndarray:
		value = _synchronized_extended_time(
			state,
			time,
			context="The internal state",
		)
		state_after, separation = midpoint_step(value, step)
		if observe:
			copy_separation_norms.append(separation)
			if method.step_observer is not None:
				def map_state(candidate: np.ndarray) -> np.ndarray:
					return midpoint_step(candidate, step)[0]

				method.step_observer(
					IntegrationStep(
						dynamics_name=type(dynamics).__name__,
						method_name=method_name,
						step_index=step_index,
						start_time=time,
						time=time + step,
						duration=step,
						state_before=value.copy(),
						state_after=state_after.copy(),
						map_state=map_state,
						dynamics=dynamics,
					)
				)
		return state_after

	extended_history, step_count = integrate_fixed_grid(
		initial_extended,
		request,
		advance,
		progress=bool(method.progress),
		label=method_name,
	)
	extended_history[2] = request.output_times
	diagnostics: dict[str, np.ndarray | float | int | str | bool] = {
		"step_count": step_count,
		"copy_separation_norms": np.asarray(copy_separation_norms, dtype=float),
		"projection_kind": "arithmetic_mean",
		"state_extension": "fully_extended",
		"accepted_internal_state_dimension": 4,
		"base_splitting_state_dimension": 8,
		"observer_state_dimension": 4,
		"observer_state_kind": "accepted_internal_map",
		"nonlinear_unknown_dimension": 0,
		"unprojected_abba_maps_per_step": 1,
		"vector_field_evaluations_per_step": 4,
	}
	diagnostics.update(_fully_extended_energy_diagnostics(dynamics, extended_history))
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

	_variant: ClassVar[_Variant] = "bm4"

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
