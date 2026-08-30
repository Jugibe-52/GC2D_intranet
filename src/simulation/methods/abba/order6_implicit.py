"""Sixth-order seven-stage composition of projected implicit ABBA steps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import GuidingCenterJacobianSystem

from .._fully_extended import _integrate_abba_fully_extended
from ..._result import IntegrationData
from ...observation import ABBA6ImplicitIntegrationStep
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from .._nonlinear import NonlinearSolver
from ._coefficients import _ABBA6_COEFFICIENTS
from ._configuration import ProjectionFormulation
from ._implicit import _ABBAImplicitConfig
from .order4_implicit import (
	_ComposedABBAStep,
	_integrate_composed_implicit_abba,
	_solve_composed_abba_step,
)


_COMPOSITION_POLICY = "project_each_abba_substep"


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
	projection_formulation: ProjectionFormulation = "reduced_multiplier",
) -> _ComposedABBAStep:
	"""Compose the seven signed projected ABBA maps of ABBA6Implicit."""
	return _solve_composed_abba_step(
		dynamics,
		t,
		state,
		step,
		coefficients=_ABBA6_COEFFICIENTS,
		method_name="ABBA6Implicit",
		absolute_tolerance=absolute_tolerance,
		relative_tolerance=relative_tolerance,
		max_iterations=max_iterations,
		nonlinear_solver=nonlinear_solver,
		projection_formulation=projection_formulation,
	)


@dataclass(frozen=True, slots=True)
class ABBA6Implicit(_ABBAImplicitConfig):
	"""Sixth-order symmetric composition of seven implicit ABBA maps.

	Each outer step applies Yoshida's palindromic seven-stage coefficients to
	complete ``ABBA2Implicit`` maps. Two substeps run backward in time. Every
	signed substep uses the same selected projection formulation, nonlinear
	solver, and state extension, and solves an independent projection problem.
	"""

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate a planar GC problem with the sixth-order composition."""
		if self.state_extension == "fully_extended":
			return _integrate_abba_fully_extended(
				self,
				problem,
				request,
				variant="abba6",
				projection_formulation=self.projection_formulation,
			)
		return _integrate_composed_implicit_abba(
			self,
			problem,
			request,
			coefficients=_ABBA6_COEFFICIENTS,
			composition_policy=_COMPOSITION_POLICY,
			observation_type=ABBA6ImplicitIntegrationStep,
		)


__all__ = ["ABBA6Implicit"]
