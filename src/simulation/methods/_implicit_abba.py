"""Shared configuration and integration contract for implicit ABBA variants."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from .._result import IntegrationData
from ..observation import StepObserver
from ..problem import InitialValueProblem
from ..request import SimulationRequest
from ._projected_abba import (
	_ProjectedStep,
	_integrate_projected_abba,
	_positive_finite,
	_positive_integer,
	_solve_projected_step,
)


@dataclass(frozen=True, slots=True)
class _ImplicitABBA:
	"""Configure a symmetric projected ABBA nonlinear formulation."""

	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	progress: bool = False
	step_observer: StepObserver | None = None

	_step_solver: ClassVar[Callable[..., _ProjectedStep]] = _solve_projected_step
	_solver_formulation: ClassVar[str] = "implicit_1_reduced_equation_11"
	_exact_tangent: ClassVar[bool] = False

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
		"""Integrate one GC problem with the selected implicit formulation."""
		return _integrate_projected_abba(
			problem,
			request,
			method_name=type(self).__name__,
			step_solver=type(self)._step_solver,
			solver_formulation=type(self)._solver_formulation,
			newton_absolute_tolerance=self.newton_absolute_tolerance,
			newton_relative_tolerance=self.newton_relative_tolerance,
			newton_max_iterations=self.newton_max_iterations,
			progress=self.progress,
			step_observer=self.step_observer,
			exact_tangent=type(self)._exact_tangent,
		)


__all__: list[str] = []
