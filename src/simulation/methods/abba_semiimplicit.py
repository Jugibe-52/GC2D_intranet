"""Semi-implicit ABBA integration with exact physical tangent propagation."""

from __future__ import annotations

from dataclasses import dataclass

from .._result import IntegrationData
from ..problem import InitialValueProblem
from ..request import SimulationRequest
from ._projected_abba import _integrate_projected_abba
from .abba import SymmetricProjectedABBA


@dataclass(frozen=True, slots=True)
class SemiImplicitABBA(SymmetricProjectedABBA):
	"""Advance projected ABBA states and their exact ideal-root tangent.

	The physical trajectory is identical to ``SymmetricProjectedABBA``. After each
	converged projection solve, this variant evaluates the exact state Jacobian by
	the implicit-function formula, propagates the accumulated tangent from the
	initial state, and provides the local matrix to the optional step observer.
	"""

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate one GC problem and propagate its exact discrete tangent."""
		return _integrate_projected_abba(
			problem,
			request,
			method_name=type(self).__name__,
			newton_absolute_tolerance=self.newton_absolute_tolerance,
			newton_relative_tolerance=self.newton_relative_tolerance,
			newton_max_iterations=self.newton_max_iterations,
			progress=self.progress,
			step_observer=self.step_observer,
			exact_tangent=True,
		)


__all__ = ["SemiImplicitABBA"]
