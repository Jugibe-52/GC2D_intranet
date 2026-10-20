"""Contract implemented by reusable numerical methods."""

from __future__ import annotations

from typing import Any, Literal, Protocol, TYPE_CHECKING, runtime_checkable

from contracts.problem import InitialValueProblem
from dynamics.protocols import DynamicalSystem
from request import SimulationRequest

if TYPE_CHECKING:
	from integration.core import IntegrationMethod


@runtime_checkable
class NumericalMethod(Protocol):
	"""Prepare numerical operations for an executor-owned integration."""

	@property
	def jax_mode(self) -> Literal["unsupported", "device_resident", "hybrid_scipy_controller"]:
		"""Declare an optional execution capability without selecting a device."""
		...

	def new_run(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
		*, evaluated_dynamics: DynamicalSystem | None = None,
	) -> IntegrationMethod[Any]:
		"""Return a fresh run, optionally using executor-prepared field evaluations."""
		...


__all__ = ["NumericalMethod"]
