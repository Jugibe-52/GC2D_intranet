"""Sixth-order seven-stage composition of reduced implicit ABBA steps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import GuidingCenterJacobianSystem

from ..._result import IntegrationData
from ...observation import ImplicitABBA6IntegrationStep
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from .._nonlinear import NonlinearSolver
from ._implicit import _ImplicitABBA
from .order4_implicit_1 import (
	_ComposedABBAStep,
	_integrate_composed_implicit_abba,
	_solve_composed_abba_step,
)


# Yoshida's real symmetric order-six solution. The palindromic sequence is
# essential for self-adjointness; its negative stages allow all degree-three
# and degree-five composition conditions to vanish.
_ABBA6_COEFFICIENTS = np.asarray(
	(
		0.78451361047755726382,
		0.23557321335935813368,
		-1.17767998417887100695,
		1.31518632068391121889,
		-1.17767998417887100695,
		0.23557321335935813368,
		0.78451361047755726382,
	),
	dtype=float,
)
_COMPOSITION_FORMULATION = "abba6_implicit_1_seven_stage_yoshida"


def _solve_abba6_step(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver,
) -> _ComposedABBAStep:
	"""Compose the seven signed reduced implicit-ABBA maps of ABBA6."""
	return _solve_composed_abba_step(
		dynamics,
		t,
		state,
		step,
		coefficients=_ABBA6_COEFFICIENTS,
		method_name="ABBA6",
		absolute_tolerance=absolute_tolerance,
		relative_tolerance=relative_tolerance,
		max_iterations=max_iterations,
		nonlinear_solver=nonlinear_solver,
	)


@dataclass(frozen=True, slots=True)
class ABBA6(_ImplicitABBA):
	"""Sixth-order symmetric composition of seven implicit ABBA maps.

	Each outer step applies Yoshida's palindromic seven-stage coefficients to
	complete reduced ``ImplicitABBA1`` maps. Two substeps run backward in time.
	Every signed substep solves its own projection multiplier equation with exact
	``2 x 2`` Newton blocks or with good Broyden updates.
	"""

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate a planar GC problem with the sixth-order composition."""
		return _integrate_composed_implicit_abba(
			self,
			problem,
			request,
			coefficients=_ABBA6_COEFFICIENTS,
			composition_formulation=_COMPOSITION_FORMULATION,
			observation_type=ImplicitABBA6IntegrationStep,
		)


__all__ = ["ABBA6"]
