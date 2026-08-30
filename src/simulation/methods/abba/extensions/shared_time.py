"""Shared-time extension of the second-order projected ABBA map."""

from __future__ import annotations

from dataclasses import dataclass

from ...._result import IntegrationData
from ....problem import InitialValueProblem
from ....request import SimulationRequest
from .._implicit import _integrate_projected_abba, _step_solver_for
from ..order2_implicit import ABBA2Implicit


@dataclass(frozen=True, slots=True)
class ABBA2SharedTimeExtendedImplicit(ABBA2Implicit):
	"""Lift ABBA2 to accepted ``(z,t,kappa)`` states and ``R^6`` splittings.

	The physical copies ``u`` and ``v`` are duplicated while one ``(t,k)`` pair
	is shared. The triangular conjugate update cannot feed back into ``z``, so
	the physical trajectory is identical to :class:`ABBA2Implicit` with the same
	projection formulation. Exactly one GC particle is required so the internal
	accepted state is literally in ``R^4`` and the splitting state is in ``R^6``.
	"""

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate the shared-time lift and return its physical trajectory."""
		return _integrate_projected_abba(
			problem,
			request,
			method_name=type(self).__name__,
			step_solver=_step_solver_for(self.projection_formulation),
			projection_formulation=self.projection_formulation,
			newton_absolute_tolerance=self.newton_absolute_tolerance,
			newton_relative_tolerance=self.newton_relative_tolerance,
			newton_max_iterations=self.newton_max_iterations,
			nonlinear_solver=self.nonlinear_solver,
			progress=self.progress,
			step_observer=self.step_observer,
			shared_time_extension=True,
		)


__all__ = ["ABBA2SharedTimeExtendedImplicit"]
