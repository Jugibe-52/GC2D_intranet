"""Public order-6 implicit ABBA configuration on the shared family runtime."""
from __future__ import annotations
from dataclasses import dataclass
from ..._result import IntegrationData
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from ._implicit import _ABBAImplicitConfig
from .preparation import prepare_abba
from .runtime import integrate_abba
from .composition import _solve_abba6_step as _solve_abba6_step
from ._coefficients import _ABBA6_COEFFICIENTS as _ABBA6_COEFFICIENTS


@dataclass(frozen=True, slots=True)
class ABBA6Implicit(_ABBAImplicitConfig):
	"""Sixth-order symmetric composition of seven implicit ABBA maps.

	Each outer step applies Yoshida's palindromic seven-stage coefficients to
	complete ``ABBA2Implicit`` maps. Two substeps run backward in time. Every
	signed substep uses the same selected projection formulation, nonlinear
	solver, and state strategy, and solves an independent projection problem.
	Physical conjugate-momentum tracking is optional and triangular.
	"""

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Prepare the selected step recipe and run the shared ABBA coordinator."""
		prepared = prepare_abba(problem, self, request, order=6)
		return integrate_abba(prepared, request)


__all__ = ["ABBA6Implicit"]
