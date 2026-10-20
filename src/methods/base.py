"""Contract implemented by reusable numerical methods."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contracts.execution_options import ExecutionOptions
from states.history import States
from contracts.problem import InitialValueProblem
from request import SimulationRequest


@runtime_checkable
class NumericalMethod(Protocol):
	"""Advance an initial-value problem and return physical integration data."""

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
		*, execution: ExecutionOptions | None = None,
	) -> States:
		"""Integrate one problem under one temporal request."""


__all__ = ["NumericalMethod"]
