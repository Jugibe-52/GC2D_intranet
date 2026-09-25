"""Full-diagonal projection equations with explicit R4/R12 Newton algebra."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from methods._nonlinear import NonlinearSolver, _solve_broyden, _solve_newton
from methods.extended.configuration import (
    ProjectionFormulation,
    _validate_projection_formulation as _validate_abba_projection_formulation,
)
from methods.extended.legacy_full_maps import (
    _AnalyticExtendedMap, _ExtendedMap, _checked_extended_state,
    _IDENTITY_4, _IDENTITY_8, _DIAGONAL_EMBEDDING,
    _ANTIDIAGONAL_EMBEDDING, _COPY_DIFFERENCE, _COPY_AVERAGE,
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

	def update(
		unknown: np.ndarray, residual: np.ndarray, payload: tuple[np.ndarray, np.ndarray],
	) -> np.ndarray:
		internal_input, mapped = payload
		jacobian = _full_reduced_residual_jacobian(base_map, internal_input)
		try:
			correction = np.linalg.solve(jacobian, residual)
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				f"The reduced full-state projection Jacobian is singular for {context}."
			) from exc
		return unknown - correction

	newton_result = _solve_newton(
		evaluate, multiplier, update, tolerance=threshold,
		max_iterations=max_iterations, context=context,
	)
	internal_input, mapped = newton_result.payload
	return accepted(
		newton_result.unknown, internal_input, mapped,
		iterations=newton_result.iterations,
		residual_evaluations=newton_result.residual_evaluations,
		residual_norm=float(np.linalg.norm(newton_result.residual, ord=np.inf)),
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

	def update(
		unknown: np.ndarray, residual: np.ndarray,
		payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
	) -> np.ndarray:
		candidate_input, candidate_mapped, first, second, mu = payload
		base_jacobian = np.asarray(base_map.jacobian_state(candidate_input), dtype=float)
		top_right = -(_IDENTITY_8 + base_jacobian) @ _ANTIDIAGONAL_EMBEDDING
		jacobian = np.block([
			[_IDENTITY_8, top_right],
			[_COPY_DIFFERENCE, np.zeros((4, 4), dtype=float)],
		])
		try:
			increment = np.linalg.solve(jacobian, -residual)
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				f"The simultaneous full-state projection Jacobian is singular for {context}."
			) from exc
		return np.concatenate((first + increment[:4], second + increment[4:8], mu + increment[8:]))

	newton_result = _solve_newton(
		evaluate_unknown, initial_unknown, update, tolerance=threshold,
		max_iterations=max_iterations, context=context,
		initial_evaluation=(
			initial_residual, (internal_input, mapped, first_output, second_output, multiplier),
		),
	)
	candidate_input, candidate_mapped, first, second, mu = newton_result.payload
	return accepted(
		candidate_input, candidate_mapped, first, second, mu,
		iterations=newton_result.iterations,
		residual_evaluations=newton_result.residual_evaluations,
		residual_norm=float(np.linalg.norm(newton_result.residual, ord=np.inf)),
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


__all__: list[str] = []
