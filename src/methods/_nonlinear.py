"""Shared nonlinear-solver primitives for implicit integration methods."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, TypeVar

import numpy as np


NonlinearSolver: TypeAlias = Literal["newton", "broyden"]
NONLINEAR_SOLVERS: tuple[NonlinearSolver, ...] = ("newton", "broyden")

_Payload = TypeVar("_Payload")
_ResidualFunction: TypeAlias = Callable[[np.ndarray], tuple[np.ndarray, _Payload]]


@dataclass(frozen=True, slots=True)
class SolverOptions:
	"""Validated nonlinear options bound once during method preparation."""

	solver: NonlinearSolver
	absolute_tolerance: float
	relative_tolerance: float
	max_iterations: int

	def tolerance(self, state: np.ndarray) -> float:
		"""Scale the accepted residual against the initial numerical state."""
		return self.absolute_tolerance + self.relative_tolerance * max(
			1.0, float(np.linalg.norm(state, ord=np.inf))
		)


@dataclass(frozen=True, slots=True)
class SolveStats:
	"""Work and convergence information for exactly one projection solve."""

	solver: NonlinearSolver
	iterations: int
	residual_evaluations: int
	residual_norm: float
	tolerance: float


@dataclass(frozen=True, slots=True)
class _NonlinearResult(Generic[_Payload]):
	"""Return values and work counts from one converged nonlinear solve.

	The vectors use the solver's packed unknown convention, which depends on
	the method (for example, concatenated particle stages or projection
	multipliers). The payload carries formulation-specific data evaluated at
	that same unknown.
	"""

	unknown: np.ndarray  # Accepted packed solution, with the same shape as the initial guess.
	residual: np.ndarray  # Residual vector evaluated at `unknown`; its infinity norm measures convergence.
	payload: _Payload  # Additional formulation-specific evaluation data paired with `unknown` and `residual`.
	iterations: int  # Accepted nonlinear corrections; zero when the initial guess already meets tolerance.
	residual_evaluations: int  # Number of residual evaluations used, including the converged evaluation.


def _validate_nonlinear_solver(value: str) -> NonlinearSolver:
	"""Return one supported nonlinear-solver name."""
	if value not in NONLINEAR_SOLVERS:
		raise ValueError(
			"`nonlinear_solver` must be either 'newton' or 'broyden'."
		)
	return value


def _checked_residual_evaluation(
	residual_function: _ResidualFunction[_Payload],
	unknown: np.ndarray,
) -> tuple[np.ndarray, _Payload]:
	"""Evaluate a residual without allowing shape changes or non-finite values."""
	residual, payload = residual_function(unknown)
	value = np.asarray(residual, dtype=float)
	if value.shape != unknown.shape or not np.all(np.isfinite(value)):
		raise ValueError(
			"The nonlinear residual must be finite and match the unknown shape."
		)
	return value, payload


def _solve_broyden(
	residual_function: _ResidualFunction[_Payload],
	initial_unknown: np.ndarray,
	initial_jacobian: np.ndarray,
	*,
	tolerance: float,
	max_iterations: int,
	context: str,
	initial_evaluation: tuple[np.ndarray, _Payload] | None = None,
) -> _NonlinearResult[_Payload]:
	"""Solve one residual equation with the good Broyden Jacobian update.

	The initial Jacobian is supplied by the formulation. Every later matrix is
	obtained from exact residual evaluations and the rank-one secant update from
	``docs/models/abba/tex/nonlinear-solvers.tex``. Iterations count accepted
	corrections, so a root at the initial guess reports zero iterations and one
	residual evaluation.
	"""
	unknown = np.asarray(initial_unknown, dtype=float).copy()
	if unknown.ndim != 1 or unknown.size == 0 or not np.all(np.isfinite(unknown)):
		raise ValueError("The initial Broyden unknown must be a finite vector.")
	jacobian = np.asarray(initial_jacobian, dtype=float).copy()
	expected_shape = (unknown.size, unknown.size)
	if jacobian.shape != expected_shape or not np.all(np.isfinite(jacobian)):
		raise ValueError(
			"The initial Broyden Jacobian must be a finite square matrix matching "
			"the unknown."
		)
	if not np.isfinite(tolerance) or tolerance <= 0.0:
		raise ValueError("The Broyden tolerance must be positive and finite.")
	if (
		isinstance(max_iterations, (bool, np.bool_))
		or not isinstance(max_iterations, (int, np.integer))
		or max_iterations < 1
	):
		raise ValueError("The Broyden iteration limit must be a positive integer.")
	if not isinstance(context, str) or not context:
		raise ValueError("The Broyden solve context must be a non-empty string.")

	if initial_evaluation is None:
		residual, payload = _checked_residual_evaluation(
			residual_function,
			unknown,
		)
	else:
		initial_residual, payload = initial_evaluation
		residual = np.asarray(initial_residual, dtype=float)
		if residual.shape != unknown.shape or not np.all(np.isfinite(residual)):
			raise ValueError(
				"The cached initial residual must be finite and match the unknown shape."
			)
	residual_evaluations = 1
	residual_norm = float(np.linalg.norm(residual, ord=np.inf))

	for iteration in range(int(max_iterations) + 1):
		if residual_norm <= tolerance:
			return _NonlinearResult(
				unknown=unknown.copy(),
				residual=residual.copy(),
				payload=payload,
				iterations=iteration,
				residual_evaluations=residual_evaluations,
			)
		if iteration == max_iterations:
			break
		try:
			increment = np.linalg.solve(jacobian, -residual)
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				f"The Broyden Jacobian approximation is singular for {context}."
			) from exc
		if not np.all(np.isfinite(increment)):
			raise RuntimeError(
				f"The Broyden correction became non-finite for {context}."
			)

		next_unknown = unknown + increment
		next_residual, next_payload = _checked_residual_evaluation(
			residual_function,
			next_unknown,
		)
		residual_evaluations += 1
		residual_change = next_residual - residual
		denominator = float(increment @ increment)
		if denominator <= np.finfo(float).tiny:
			next_norm = float(np.linalg.norm(next_residual, ord=np.inf))
			if next_norm > tolerance:
				raise RuntimeError(
					"The Broyden correction became too small before convergence for "
					f"{context}: residual norm {next_norm:.3e} exceeds "
					f"{tolerance:.3e}."
				)
		else:
			jacobian_increment = residual_change - jacobian @ increment
			jacobian = jacobian + np.outer(
				jacobian_increment,
				increment,
			) / denominator

		unknown = next_unknown
		residual = next_residual
		payload = next_payload
		residual_norm = float(np.linalg.norm(residual, ord=np.inf))

	raise RuntimeError(
		f"Broyden did not converge for {context}: residual norm "
		f"{residual_norm:.3e} exceeds {tolerance:.3e} after "
		f"{max_iterations} iterations and {residual_evaluations} residual "
		"evaluations."
	)


def _solve_newton(
	residual_function: _ResidualFunction[_Payload],
	initial_unknown: np.ndarray,
	update: Callable[[np.ndarray, np.ndarray, _Payload], np.ndarray],
	*,
	tolerance: float,
	max_iterations: int,
	context: str,
	initial_evaluation: tuple[np.ndarray, _Payload] | None = None,
) -> _NonlinearResult[_Payload]:
	"""Share convergence and counters while the formulation owns Newton algebra.

	The update callback returns the new unknown. This preserves each method's
	subtraction/addition convention, particle packing and specialized block solve.
	A supplied initial evaluation counts once and is never recomputed.
	"""
	unknown = np.asarray(initial_unknown, dtype=float).copy()
	if unknown.ndim != 1 or unknown.size == 0 or not np.all(np.isfinite(unknown)):
		raise ValueError("The initial Newton unknown must be a finite vector.")
	if not np.isfinite(tolerance) or tolerance <= 0.0:
		raise ValueError("The Newton tolerance must be positive and finite.")
	if (
		isinstance(max_iterations, (bool, np.bool_))
		or not isinstance(max_iterations, (int, np.integer))
		or max_iterations < 1
	):
		raise ValueError("The Newton iteration limit must be a positive integer.")
	for iteration in range(int(max_iterations) + 1):
		if iteration == 0 and initial_evaluation is not None:
			residual, payload = initial_evaluation
			residual = np.asarray(residual, dtype=float)
			if residual.shape != unknown.shape or not np.all(np.isfinite(residual)):
				raise ValueError("The cached Newton residual must be finite and match the unknown.")
		else:
			residual, payload = _checked_residual_evaluation(residual_function, unknown)
		residual_norm = float(np.linalg.norm(residual, ord=np.inf))
		if residual_norm <= tolerance:
			return _NonlinearResult(
				unknown=unknown.copy(), residual=residual.copy(), payload=payload,
				iterations=iteration, residual_evaluations=iteration + 1,
			)
		if iteration == max_iterations:
			break
		next_unknown = np.asarray(update(unknown, residual, payload), dtype=float)
		if next_unknown.shape != unknown.shape or not np.all(np.isfinite(next_unknown)):
			raise RuntimeError(f"The Newton correction became invalid for {context}.")
		unknown = next_unknown
	raise RuntimeError(
		f"Newton did not converge for {context}: residual norm {residual_norm:.3e} "
		f"exceeds {tolerance:.3e} after {max_iterations} iterations."
	)


__all__ = ["NONLINEAR_SOLVERS", "NonlinearSolver"]
