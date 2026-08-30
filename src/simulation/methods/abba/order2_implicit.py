"""Second-order implicit ABBA with orthogonal solver and state-space axes."""

from __future__ import annotations

from dataclasses import dataclass

from .._fully_extended import _integrate_abba_fully_extended
from ..._result import IntegrationData
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from ._implicit import (
	_ABBAImplicitConfig,
	_integrate_projected_abba,
	_step_solver_for,
)


@dataclass(frozen=True, slots=True)
class ABBA2Implicit(_ABBAImplicitConfig):
	"""Second-order implicit ABBA with twelve supported configurations.

	The two projection formulations define the same accepted map and may be
	solved by Newton or Broyden. ``state_extension`` selects the physical,
	shared-time, or fully duplicated ``(z,t,k)`` map; the fully duplicated branch
	has reduced and simultaneous nonlinear workspaces in ``R^4`` and ``R^12``.
	"""

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate one GC problem with the selected three-axis configuration."""
		if self.state_extension == "fully_extended":
			return _integrate_abba_fully_extended(
				self,
				problem,
				request,
				variant="abba",
				projection_formulation=self.projection_formulation,
			)
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
			shared_time_extension=self.state_extension == "shared_time",
		)


__all__ = ["ABBA2Implicit"]
