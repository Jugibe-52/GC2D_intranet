"""Public order-2 implicit ABBA configuration on the shared family runtime."""
from __future__ import annotations
from dataclasses import dataclass
from ..._result import IntegrationData
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from ._implicit import _ABBAImplicitConfig
from .preparation import prepare_abba
from .runtime import integrate_abba


@dataclass(frozen=True, slots=True)
class ABBA2Implicit(_ABBAImplicitConfig):
	"""Second-order implicit ABBA with optional physical energy tracking.

	The two projection formulations define the same accepted map and may be
	solved by Newton or Broyden. ``state_extension`` selects the physical
	or fully duplicated ``(z,t,k)`` map. Physical energy tracking is a
	triangular auxiliary update and does not change the accepted physical map.
	"""

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Prepare the selected step recipe and run the shared ABBA coordinator."""
		prepared = prepare_abba(problem, self, request, order=2)
		return integrate_abba(prepared, request)


__all__ = ["ABBA2Implicit"]
