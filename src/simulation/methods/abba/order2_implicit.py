"""Second-order implicit ABBA with selectable nonlinear formulation."""

from __future__ import annotations

from dataclasses import dataclass

from ..._result import IntegrationData
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from ._implicit import (
	ProjectionFormulation,
	_ABBAImplicitConfig,
	_integrate_projected_abba,
	_step_solver_for,
	_validate_projection_formulation,
)


@dataclass(frozen=True, slots=True)
class ABBA2Implicit(_ABBAImplicitConfig):
	"""Second-order implicit ABBA with equivalent projection formulations.

	``reduced_multiplier`` solves only the projection multiplier with one
	``2 x 2`` block per particle. ``simultaneous_state_multiplier`` solves the
	equivalent final-copy and multiplier equations with ``6 x 6`` blocks. Both
	choices define the same accepted physical map.
	"""

	projection_formulation: ProjectionFormulation = "reduced_multiplier"

	def __post_init__(self) -> None:
		"""Validate common solver controls and the formulation identifier."""
		_ABBAImplicitConfig.__post_init__(self)
		object.__setattr__(
			self,
			"projection_formulation",
			_validate_projection_formulation(self.projection_formulation),
		)

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate one physical GC problem with the selected formulation."""
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
		)


__all__ = ["ABBA2Implicit"]
