"""Shared nonlinear-solver primitives for implicit integration methods."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Generic, Literal, TypeAlias, TypeVar

import numpy as np

from contracts.nonlinear import NewtonIteration, NewtonObserver


NonlinearSolver: TypeAlias = Literal["newton", "broyden"]
NONLINEAR_SOLVERS: tuple[NonlinearSolver, ...] = ("newton", "broyden")

_Payload = TypeVar("_Payload")
_ResidualFunction: TypeAlias = Callable[[np.ndarray], tuple[np.ndarray, _Payload]]
_IterateObserver: TypeAlias = Callable[[int, np.ndarray, np.ndarray, float], None]


def _bind_newton_observer(
	observer: NewtonObserver | None, *, time: float, duration: float,
	tolerance: float, stage_index: int | None = None,
	multiplier_slice: slice | None = None,
) -> _IterateObserver | None:
	"""Attach method coordinates to evaluated iterates without numerical work."""
	if observer is None:
		return None

	def observe(iteration: int, unknown: np.ndarray, residual: np.ndarray, norm: float) -> None:
		observer(NewtonIteration(
			time=time, duration=duration, iteration=iteration, unknown=unknown,
			residual=residual, residual_norm=norm, tolerance=tolerance,
			multiplier=None if multiplier_slice is None else unknown[multiplier_slice],
			stage_index=stage_index,
		))

	return observe


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


def _validated_initial_unknown(initial_unknown: np.ndarray, solver: str) -> np.ndarray:
	"""Own the finite, nonempty vector whose coordinates the solver will update."""
	unknown = np.asarray(initial_unknown, dtype=float).copy()
	if unknown.ndim != 1 or unknown.size == 0 or not np.all(np.isfinite(unknown)):
		raise ValueError(f"The initial {solver} unknown must be a finite vector.")
	return unknown


def _validate_solver_controls(tolerance: float, max_iterations: int, solver: str) -> None:
	"""Check solve-local stopping controls without coercing their existing types."""
	if not np.isfinite(tolerance) or tolerance <= 0.0:
		raise ValueError(f"The {solver} tolerance must be positive and finite.")
	if (
		isinstance(max_iterations, (bool, np.bool_))
		or not isinstance(max_iterations, (int, np.integer))
		or max_iterations < 1
	):
		raise ValueError(f"The {solver} iteration limit must be a positive integer.")


def _validated_broyden_jacobian(initial_jacobian: np.ndarray, unknown: np.ndarray) -> np.ndarray:
	"""Own a finite square initial approximation in the unknown's coordinates."""
	jacobian = np.asarray(initial_jacobian, dtype=float).copy()
	expected_shape = (unknown.size, unknown.size)
	if jacobian.shape != expected_shape or not np.all(np.isfinite(jacobian)):
		raise ValueError(
			"The initial Broyden Jacobian must be a finite square matrix matching "
			"the unknown."
		)
	return jacobian


def _validated_residual(residual: np.ndarray, unknown: np.ndarray, message: str) -> np.ndarray:
	"""Normalize one finite residual without re-evaluating its paired payload."""
	value = np.asarray(residual, dtype=float)
	if value.shape != unknown.shape or not np.all(np.isfinite(value)):
		raise ValueError(message)
	return value


def _checked_residual_evaluation(
	residual_function: _ResidualFunction[_Payload],
	unknown: np.ndarray,
) -> tuple[np.ndarray, _Payload]:
	"""Evaluate a residual without allowing shape changes or non-finite values."""
	residual, payload = residual_function(unknown)
	value = _validated_residual(
		residual, unknown, "The nonlinear residual must be finite and match the unknown shape.",
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
	unknown = _validated_initial_unknown(initial_unknown, "Broyden")
	jacobian = _validated_broyden_jacobian(initial_jacobian, unknown)
	_validate_solver_controls(tolerance, max_iterations, "Broyden")
	if not isinstance(context, str) or not context:
		raise ValueError("The Broyden solve context must be a non-empty string.")

	if initial_evaluation is None:
		residual, payload = _checked_residual_evaluation(
			residual_function,
			unknown,
		)
	else:
		initial_residual, payload = initial_evaluation
		residual = _validated_residual(
			initial_residual, unknown,
			"The cached initial residual must be finite and match the unknown shape.",
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
	iteration_observer: _IterateObserver | None = None,
	norm: Callable[[np.ndarray], float] | None = None,
	backtracking: bool = False,
	update_returns_correction: bool = False,
) -> _NonlinearResult[_Payload]:
	"""Share convergence and counters while the formulation owns Newton algebra.

	The update callback returns the new unknown. This preserves each method's
	subtraction/addition convention, particle packing and specialized block solve.
	A supplied initial evaluation counts once and is never recomputed.
	"""
	unknown = _validated_initial_unknown(initial_unknown, "Newton")
	_validate_solver_controls(tolerance, max_iterations, "Newton")
	measure = _infinity_norm if norm is None else norm
	if backtracking and not update_returns_correction:
		raise ValueError("Backtracking requires an additive Newton correction.")
	evaluations = 0
	cached = initial_evaluation
	for iteration in range(int(max_iterations) + 1):
		if cached is not None:
			residual, payload = cached
			residual = _validated_residual(
				residual, unknown, "The cached Newton residual must be finite and match the unknown.",
			)
			if iteration == 0:
				evaluations += 1
			cached = None
		else:
			residual, payload = _checked_residual_evaluation(residual_function, unknown)
			evaluations += 1
		residual_norm = measure(residual)
		if iteration_observer is not None:
			iteration_observer(iteration, unknown, residual, residual_norm)
		if residual_norm <= tolerance:
			return _NonlinearResult(
				unknown=unknown.copy(), residual=residual.copy(), payload=payload,
				iterations=iteration, residual_evaluations=evaluations,
			)
		if iteration == max_iterations:
			break
		next_unknown = np.asarray(update(unknown, residual, payload), dtype=float)
		if update_returns_correction:
			correction = next_unknown
			damping = 1.0
			while True:
				next_unknown = unknown + damping * correction
				if not backtracking:
					break
				cached = _checked_residual_evaluation(residual_function, next_unknown)
				evaluations += 1
				if measure(cached[0]) < residual_norm or damping <= 1.0 / 128.0:
					break
				damping /= 2.0
		if next_unknown.shape != unknown.shape or not np.all(np.isfinite(next_unknown)):
			raise RuntimeError(f"The Newton correction became invalid for {context}.")
		unknown = next_unknown
	raise RuntimeError(
		f"Newton did not converge for {context}: residual norm {residual_norm:.3e} "
		f"exceeds {tolerance:.3e} after {max_iterations} iterations."
	)


def _infinity_norm(value: np.ndarray) -> float:
	return float(np.linalg.norm(value, ord=np.inf))


@dataclass(frozen=True, slots=True)
class _SolverConfiguration:
	"""Immutable controls; all work arrays belong to individual solve calls."""

	absolute_tolerance: float = 1e-13
	relative_tolerance: float = 1e-12
	max_iterations: int = 12

	def __post_init__(self) -> None:
		for name in ('absolute_tolerance', 'relative_tolerance'):
			value = getattr(self, name)
			if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
				raise ValueError(f"`{name}` must be positive and finite.")
			object.__setattr__(self, name, float(value))
		_validate_solver_controls(self.absolute_tolerance, self.max_iterations, 'nonlinear')
		object.__setattr__(self, 'max_iterations', int(self.max_iterations))

	def tolerance(self, scale: np.ndarray) -> float:
		"""Use the supplied physical-state scale, independent of root coordinates."""
		return self.absolute_tolerance + self.relative_tolerance * max(1.0, _infinity_norm(scale))


@dataclass(frozen=True, slots=True)
class NewtonSolver(_SolverConfiguration):
	"""Newton iteration with a supplied Jacobian or specialized update."""

	name: ClassVar[Literal['newton']] = 'newton'

	def solve(
		self, evaluate: _ResidualFunction[_Payload], initial: np.ndarray,
		update: Callable[[np.ndarray, np.ndarray, _Payload], np.ndarray] | None = None,
		*, scale: np.ndarray,
		jacobian: Callable[[np.ndarray], np.ndarray] | None = None,
		context: str = 'nonlinear system',
		initial_evaluation: tuple[np.ndarray, _Payload] | None = None,
		iteration_observer: _IterateObserver | None = None,
		norm: Callable[[np.ndarray], float] | None = None,
		backtracking: bool = False,
		update_returns_correction: bool = False,
	) -> _NonlinearResult[_Payload]:
		"""Solve without interpreting stages, particles, multipliers or physical time."""
		if (update is None) == (jacobian is None):
			raise ValueError("Supply exactly one of `update` or `jacobian`.")
		if update is None:
			assert jacobian is not None
			def linear_correction(value: np.ndarray, residual: np.ndarray, payload: _Payload) -> np.ndarray:
				assert jacobian is not None
				return np.asarray(np.linalg.solve(jacobian(value), -residual))
			update = linear_correction
			update_returns_correction = True
		return _solve_newton(evaluate, initial, update, tolerance=self.tolerance(scale),
			max_iterations=self.max_iterations, context=context, initial_evaluation=initial_evaluation,
			iteration_observer=iteration_observer, norm=norm, backtracking=backtracking,
			update_returns_correction=update_returns_correction)


@dataclass(frozen=True, slots=True)
class BroydenSolver(_SolverConfiguration):
	"""Good Broyden iteration initialized by the equation's Jacobian estimate."""

	name: ClassVar[Literal['broyden']] = 'broyden'

	def solve(
		self, evaluate: _ResidualFunction[_Payload], initial: np.ndarray,
		initial_jacobian: np.ndarray, *, scale: np.ndarray,
		context: str = 'nonlinear system',
		initial_evaluation: tuple[np.ndarray, _Payload] | None = None,
	) -> _NonlinearResult[_Payload]:
		"""Keep a fresh Jacobian approximation and iterate state for each solve."""
		return _solve_broyden(evaluate, initial, initial_jacobian, tolerance=self.tolerance(scale),
			max_iterations=self.max_iterations, context=context, initial_evaluation=initial_evaluation)


AlgebraicSolver: TypeAlias = NewtonSolver | BroydenSolver


def configured_solver(name: NonlinearSolver, absolute_tolerance: float,
		relative_tolerance: float, max_iterations: int) -> AlgebraicSolver:
	"""Normalize supported legacy controls into a reusable solver configuration."""
	selected = NewtonSolver if _validate_nonlinear_solver(name) == 'newton' else BroydenSolver
	return selected(absolute_tolerance, relative_tolerance, max_iterations)


def validate_solver(solver: AlgebraicSolver, *, newton_only: bool = False) -> None:
	"""Reject unsupported algorithms or subclass overrides before backend dispatch."""
	if type(solver) not in (NewtonSolver, BroydenSolver):
		raise TypeError("`solver` must be a built-in NewtonSolver or BroydenSolver.")
	if newton_only and type(solver) is not NewtonSolver:
		raise ValueError("This method supports NewtonSolver only.")


__all__ = ["NewtonSolver", "BroydenSolver", "NONLINEAR_SOLVERS", "NonlinearSolver"]
