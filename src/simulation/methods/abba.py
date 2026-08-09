"""Public symmetric-projected ABBA numerical method."""

from __future__ import annotations

from dataclasses import dataclass

from .._result import IntegrationData
from ..observation import StepObserver
from ..problem import InitialValueProblem
from ..request import SimulationRequest
from ._projected_abba import (
	_dense_component_major_jacobian,
	_differentiate_stages,
	_evaluate_residual,
	_evaluate_stages,
	_ideal_projected_state_jacobian,
	_integrate_projected_abba,
	_positive_finite,
	_positive_integer,
	_solve_projected_step,
)


@dataclass(frozen=True, slots=True)
class SymmetricProjectedABBA:
	"""Second-order ABBA method closed by Hairer's symmetric projection."""

	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	progress: bool = False
	step_observer: StepObserver | None = None

	def __post_init__(self) -> None:
		"""Validate the nonlinear projection solver configuration."""
		object.__setattr__(
			self,
			"newton_absolute_tolerance",
			_positive_finite(
				self.newton_absolute_tolerance,
				"newton_absolute_tolerance",
			),
		)
		object.__setattr__(
			self,
			"newton_relative_tolerance",
			_positive_finite(
				self.newton_relative_tolerance,
				"newton_relative_tolerance",
			),
		)
		object.__setattr__(
			self,
			"newton_max_iterations",
			_positive_integer(
				self.newton_max_iterations,
				"newton_max_iterations",
			),
		)

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate one GC problem and retain nonlinear-solve diagnostics."""
		return _integrate_projected_abba(
			problem,
			request,
			method_name=type(self).__name__,
			newton_absolute_tolerance=self.newton_absolute_tolerance,
			newton_relative_tolerance=self.newton_relative_tolerance,
			newton_max_iterations=self.newton_max_iterations,
			progress=self.progress,
			step_observer=self.step_observer,
			exact_tangent=False,
		)


__all__ = ["SymmetricProjectedABBA"]
