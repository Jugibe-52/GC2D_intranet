"""Shared nonlinear-solver primitives for implicit integration methods."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import ClassVar, Generic, Literal, TYPE_CHECKING, TypeAlias, TypeVar

import numpy as np
from contracts.arrays import BackendArray
import sys

from contracts.solvers.nonlinear import NewtonIteration, NewtonObserver
from solvers.derivatives import JacobianMethod, centered_difference


if TYPE_CHECKING:
	from solvers.jax_nonlinear import JaxNewtonSolver, JaxBroydenSolver
	from typing import Any


NonlinearSolver: TypeAlias = Literal["newton", "broyden"]
NONLINEAR_SOLVERS: tuple[NonlinearSolver, ...] = ("newton", "broyden")

_Payload = TypeVar("_Payload")
_ResidualFunction: TypeAlias = Callable[[BackendArray], tuple[BackendArray, _Payload]]
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
class SolveStats:
	"""Work and convergence information for exactly one algebraic solve."""

	solver: NonlinearSolver
	iterations: int
	residual_evaluations: int
	residual_norm: float
	tolerance: float
	converged: Any = True


@dataclass(frozen=True, slots=True)
class _NonlinearResult(Generic[_Payload]):
	"""Return values and work counts from one converged nonlinear solve.

	The vectors use the solver's packed unknown convention, which depends on
	the method (for example, concatenated particle stages or projection
	multipliers). The payload carries equation-specific data evaluated at
	that same unknown.
	"""

	unknown: BackendArray  # Accepted packed solution, with the same shape as the initial guess.
	residual: BackendArray  # Residual vector evaluated at `unknown`; its infinity norm measures convergence.
	payload: _Payload  # Additional equation-specific data paired with `unknown` and `residual`.
	iterations: int  # Accepted nonlinear corrections; zero when the initial guess already meets tolerance.
	residual_evaluations: int  # Number of residual evaluations used, including the converged evaluation.
	converged: Any = True


def _validated_initial_unknown(initial_unknown: BackendArray, solver: str) -> np.ndarray:
	"""Own the finite, nonempty vector whose coordinates the solver will update."""
	unknown: np.ndarray = np.asarray(initial_unknown, dtype=float).copy()
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


def _validated_broyden_jacobian(initial_jacobian: BackendArray, unknown: np.ndarray) -> np.ndarray:
	"""Own a finite square initial approximation in the unknown's coordinates."""
	jacobian: np.ndarray = np.asarray(initial_jacobian, dtype=float).copy()
	expected_shape = (unknown.size, unknown.size)
	if jacobian.shape != expected_shape or not np.all(np.isfinite(jacobian)):
		raise ValueError(
			"The initial Broyden Jacobian must be a finite square matrix matching "
			"the unknown."
		)
	return jacobian


def _validated_residual(residual: BackendArray, unknown: np.ndarray, message: str) -> np.ndarray:
	"""Normalize one finite residual without re-evaluating its paired payload."""
	value: np.ndarray = np.asarray(residual, dtype=float)
	if value.shape != unknown.shape or not np.all(np.isfinite(value)):
		raise ValueError(message)
	return value


def _checked_residual_evaluation(
	residual_function: _ResidualFunction[_Payload],
	unknown: np.ndarray,
) -> tuple[np.ndarray, _Payload]:
	"""Evaluate a residual without allowing shape changes or non-finite values."""
	residual: BackendArray
	payload: _Payload
	residual, payload = residual_function(unknown)
	value: np.ndarray = _validated_residual(
		residual, unknown, "The nonlinear residual must be finite and match the unknown shape.",
	)
	return value, payload


def _solve_broyden(
	residual_function: _ResidualFunction[_Payload],
	initial_unknown: BackendArray,
	initial_jacobian: BackendArray,
	*,
	tolerance: float,
	max_iterations: int,
	context: str,
	initial_evaluation: tuple[BackendArray, _Payload] | None = None,
) -> _NonlinearResult[_Payload]:
	"""Solve one residual equation with the good Broyden Jacobian update.

	The initial Jacobian is supplied by the equation. Every later matrix is
	obtained from exact residual evaluations and the rank-one secant update from
	``docs/models/abba/tex/nonlinear-solvers.tex``. Iterations count accepted
	corrections, so a root at the initial guess reports zero iterations and one
	residual evaluation.
	"""
	unknown: np.ndarray = _validated_initial_unknown(initial_unknown, "Broyden")
	jacobian: np.ndarray = _validated_broyden_jacobian(initial_jacobian, unknown)
	_validate_solver_controls(tolerance, max_iterations, "Broyden")
	if not isinstance(context, str) or not context:
		raise ValueError("The Broyden solve context must be a non-empty string.")

	if initial_evaluation is None:
		residual: np.ndarray
		payload: _Payload
		residual, payload = _checked_residual_evaluation(
			residual_function,
			unknown,
		)
	else:
		initial_residual: BackendArray
		initial_residual, payload = initial_evaluation
		residual = _validated_residual(
			initial_residual, unknown,
			"The cached initial residual must be finite and match the unknown shape.",
		)
	residual_evaluations = 1
	residual_norm: float = float(np.linalg.norm(residual, ord=np.inf))

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
			increment: np.ndarray = np.linalg.solve(jacobian, -residual)
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				f"The Broyden Jacobian approximation is singular for {context}."
			) from exc
		if not np.all(np.isfinite(increment)):
			raise RuntimeError(
				f"The Broyden correction became non-finite for {context}."
			)

		next_unknown = unknown + increment
		next_residual: np.ndarray
		next_payload: _Payload
		next_residual, next_payload = _checked_residual_evaluation(
			residual_function,
			next_unknown,
		)
		residual_evaluations += 1
		residual_change = next_residual - residual
		denominator: float = float(increment @ increment)
		if denominator <= np.finfo(float).tiny:
			next_norm: float = float(np.linalg.norm(next_residual, ord=np.inf))
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
	initial_unknown: BackendArray,
	update: Callable[[BackendArray, BackendArray, _Payload], BackendArray],
	*,
	tolerance: float,
	max_iterations: int,
	context: str,
	initial_evaluation: tuple[BackendArray, _Payload] | None = None,
	iteration_observer: _IterateObserver | None = None,
	norm: Callable[[np.ndarray], float] | None = None,
	backtracking: bool = False,
	update_returns_correction: bool = False,
) -> _NonlinearResult[_Payload]:
	"""Share convergence and counters while the equation supplies Newton algebra.

	The update callback returns the new unknown. This preserves each method's
	subtraction/addition convention, particle packing and specialized block solve.
	A supplied initial evaluation counts once and is never recomputed.
	"""
	unknown: np.ndarray = _validated_initial_unknown(initial_unknown, "Newton")
	_validate_solver_controls(tolerance, max_iterations, "Newton")
	measure = _infinity_norm if norm is None else norm
	evaluations = 0
	cached: tuple[BackendArray, _Payload] | None = initial_evaluation
	for iteration in range(int(max_iterations) + 1):
		if cached is not None:
			residual: BackendArray
			payload: _Payload
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
		residual_norm: float = measure(residual)
		if iteration_observer is not None:
			iteration_observer(iteration, unknown, residual, residual_norm)
		if residual_norm <= tolerance:
			return _NonlinearResult(
				unknown=unknown.copy(), residual=residual.copy(), payload=payload,
				iterations=iteration, residual_evaluations=evaluations,
			)
		if iteration == max_iterations:
			break
		next_unknown: np.ndarray = np.asarray(update(unknown, residual, payload), dtype=float)
		if update_returns_correction or backtracking:
			correction = next_unknown if update_returns_correction else next_unknown - unknown
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


def _infinity_norm(value: BackendArray) -> float:
	return float(np.linalg.norm(value, ord=np.inf))


@dataclass(frozen=True, slots=True)
class _SolverConfiguration:
	"""Immutable controls; all work arrays belong to individual solve calls."""

	absolute_tolerance: float = 1e-13
	relative_tolerance: float = 1e-12
	max_iterations: int = 12
	jacobian_method: JacobianMethod = field(default='auto', kw_only=True)
	jacobian_relative_step: float = field(default=float(np.cbrt(np.finfo(float).eps)), kw_only=True)
	backtracking: bool = field(default=False, kw_only=True)
	iteration_observer: NewtonObserver | None = field(default=None, kw_only=True)

	def __post_init__(self) -> None:
		for name in ('absolute_tolerance', 'relative_tolerance', 'jacobian_relative_step'):
			value: Any = getattr(self, name)
			if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
				raise ValueError(f"`{name}` must be positive and finite.")
			object.__setattr__(self, name, float(value))
		_validate_solver_controls(self.absolute_tolerance, self.max_iterations, 'nonlinear')
		object.__setattr__(self, 'max_iterations', int(self.max_iterations))
		if self.jacobian_method not in ('auto', 'analytic', 'finite_difference'):
			raise ValueError("`jacobian_method` must be 'auto', 'analytic', or 'finite_difference'.")
		if not isinstance(self.backtracking, bool):
			raise TypeError("`backtracking` must be a Boolean.")
		if self.iteration_observer is not None and not callable(self.iteration_observer):
			raise TypeError("`iteration_observer` must be callable or None.")

	def tolerance(self, scale: BackendArray) -> float:
		"""Use the supplied physical-state scale, independent of root coordinates."""
		return self.absolute_tolerance + self.relative_tolerance * max(1.0, _infinity_norm(scale))


@dataclass(frozen=True, slots=True)
class NewtonSolver(_SolverConfiguration):
	"""Configured Newton iteration with a residual derivative or specialized update.

	A specialized update retains its equation's packing and derivative target.
	Otherwise analytic or centered residual derivatives follow ``jacobian_method``.
	Backtracking halves a correction until its residual decreases, down to 1/128.
	"""

	name: ClassVar[Literal['newton']] = 'newton'

	def _as_jax(self) -> JaxNewtonSolver:
		"""Prepare the same algebraic controls without modifying this configuration."""
		from dataclasses import fields
		from solvers.jax_nonlinear import JaxNewtonSolver
		if type(self) is JaxNewtonSolver:
			return self
		if type(self) is not NewtonSolver:
			raise TypeError("Solver subclass overrides cannot be prepared automatically.")
		return JaxNewtonSolver(**{item.name: getattr(self, item.name) for item in fields(self) if item.init})


	def solve(
		self, evaluate: _ResidualFunction[_Payload], initial: BackendArray,
		update: Callable[[BackendArray, BackendArray, _Payload], BackendArray] | None = None,
		*, scale: BackendArray,
		jacobian: Callable[[BackendArray], BackendArray] | None = None,
		context: str = 'nonlinear system',
		initial_evaluation: tuple[BackendArray, _Payload] | None = None,
		norm: Callable[[BackendArray], float] | None = None,
		update_returns_correction: bool = False,
		time: float = 0.0, duration: float = 0.0,
		stage_index: int | None = None, multiplier_slice: slice | None = None,
		observe: bool = True,
	) -> _NonlinearResult[_Payload]:
		"""Solve without interpreting stages, particles, multipliers or physical time."""
		if update is not None and jacobian is not None:
			raise ValueError("Supply at most one of `update` or `jacobian`.")
		derivative_evaluations = 0
		if update is None:
			def residual_only(candidate: BackendArray) -> BackendArray:
				nonlocal derivative_evaluations
				derivative_evaluations += 1
				return evaluate(candidate)[0]

			def linear_correction(value: BackendArray, residual: BackendArray, payload: _Payload) -> BackendArray:
				# Resolve only when a correction is needed, after the initial residual.
				if self.jacobian_method == 'analytic' and jacobian is None:
					raise ValueError("Analytic Newton iteration requires a Jacobian.")
				matrix: BackendArray = (
					centered_difference(residual_only, value,
						relative_step=self.jacobian_relative_step)
					if self.jacobian_method == 'finite_difference' or jacobian is None
					else jacobian(value)
				)
				return np.asarray(np.linalg.solve(matrix, -residual))
			update = linear_correction
			update_returns_correction = True
		tolerance = self.tolerance(scale)
		observer: _IterateObserver | None = None
		if observe:
			observer = _bind_newton_observer(self.iteration_observer, time=time,
				duration=duration, tolerance=tolerance, stage_index=stage_index,
				multiplier_slice=multiplier_slice)
		result: _NonlinearResult[_Payload] = _solve_newton(evaluate, initial, update, tolerance=tolerance,
			max_iterations=self.max_iterations, context=context, initial_evaluation=initial_evaluation,
			iteration_observer=observer, norm=norm,
			backtracking=self.backtracking,
			update_returns_correction=update_returns_correction)
		return replace(result, residual_evaluations=result.residual_evaluations + derivative_evaluations)


@dataclass(frozen=True, slots=True)
class BroydenSolver(_SolverConfiguration):
	"""Good Broyden iteration initialized by the equation's Jacobian estimate."""

	name: ClassVar[Literal['broyden']] = 'broyden'

	def _as_jax(self) -> JaxBroydenSolver:
		"""Prepare the same algebraic controls without modifying this configuration."""
		from dataclasses import fields
		from solvers.jax_nonlinear import JaxBroydenSolver
		if type(self) is JaxBroydenSolver:
			return self
		if type(self) is not BroydenSolver:
			raise TypeError("Solver subclass overrides cannot be prepared automatically.")
		return JaxBroydenSolver(**{item.name: getattr(self, item.name) for item in fields(self) if item.init})


	def __post_init__(self) -> None:
		"""Keep the supported undamped Broyden algorithm explicit."""
		_SolverConfiguration.__post_init__(self)
		if self.backtracking or self.iteration_observer is not None:
			raise ValueError("BroydenSolver does not support backtracking or iteration observation.")

	def solve(
		self, evaluate: _ResidualFunction[_Payload], initial: BackendArray,
		initial_jacobian: BackendArray, *, scale: BackendArray,
		context: str = 'nonlinear system',
		initial_evaluation: tuple[BackendArray, _Payload] | None = None,
	) -> _NonlinearResult[_Payload]:
		"""Keep a fresh Jacobian approximation and iterate state for each solve."""
		return _solve_broyden(evaluate, initial, initial_jacobian, tolerance=self.tolerance(scale),
			max_iterations=self.max_iterations, context=context, initial_evaluation=initial_evaluation)


AlgebraicSolver: TypeAlias = NewtonSolver | BroydenSolver


def validate_solver(solver: AlgebraicSolver, *, newton_only: bool = False) -> None:
	"""Reject unsupported algorithms or subclass overrides before backend dispatch."""
	optional: Any = sys.modules.get('solvers.jax_nonlinear')
	allowed: tuple[type, ...] = (NewtonSolver, BroydenSolver)
	if optional is not None:
		allowed += (optional.JaxNewtonSolver, optional.JaxBroydenSolver)
	if type(solver) not in allowed:
		raise TypeError("`solver` must be a built-in NewtonSolver or BroydenSolver.")
	if newton_only and not isinstance(solver, NewtonSolver):
		raise ValueError("This method supports NewtonSolver only.")


__all__ = ["NewtonSolver", "BroydenSolver", "NONLINEAR_SOLVERS", "NonlinearSolver"]
