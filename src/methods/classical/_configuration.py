"""Shared constructor controls for implicit classical Runge--Kutta methods."""

from __future__ import annotations

from methods._validation import _positive_finite, _positive_integer
from solvers.nonlinear import AlgebraicSolver, validate_solver
from ._jacobians import JacobianMethod


def _prepared_newton_controls(
	solver: AlgebraicSolver | None, *, absolute_tolerance: float,
	relative_tolerance: float, jacobian_relative_step: float,
	max_iterations: int, jacobian_method: JacobianMethod,
) -> tuple[float, float, float, int]:
	"""Resolve a supplied solver before validating legacy Newton controls.

	Gauss--Legendre and SDIRK accept Boolean tolerances but reject Boolean
	iteration counts. Differentiation scale errors precede count errors.
	"""
	if solver is not None:
		validate_solver(solver, newton_only=True)
		absolute_tolerance = solver.absolute_tolerance
		relative_tolerance = solver.relative_tolerance
		max_iterations = solver.max_iterations
	absolute_tolerance = _positive_finite(
		absolute_tolerance, "newton_absolute_tolerance", allow_boolean=True,
	)
	relative_tolerance = _positive_finite(
		relative_tolerance, "newton_relative_tolerance", allow_boolean=True,
	)
	jacobian_relative_step = _positive_finite(
		jacobian_relative_step, "newton_jacobian_relative_step", allow_boolean=True,
	)
	max_iterations = _positive_integer(max_iterations, "newton_max_iterations")
	if jacobian_method not in ("auto", "analytic", "finite_difference"):
		raise ValueError(
			"`newton_jacobian_method` must be 'auto', 'analytic', or "
			"'finite_difference'."
		)
	return absolute_tolerance, relative_tolerance, jacobian_relative_step, max_iterations
