"""Contract implemented by reusable numerical methods."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .._result import IntegrationData
from ..problem import InitialValueProblem
from ..request import SimulationRequest


@runtime_checkable
class NumericalMethod(Protocol):
	"""Advance an initial-value problem and return physical integration data."""

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Integrate one problem under one temporal request."""


__all__ = ["NumericalMethod"]
