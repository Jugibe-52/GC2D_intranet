"""Structural contracts consumed by methods.base."""

from __future__ import annotations

from typing import Any, Literal, Protocol, TYPE_CHECKING, runtime_checkable
from contracts.dynamics.protocols import DynamicalSystem
if TYPE_CHECKING:
	from methods.base import IntegrationMethod
	from problem import InitialValueProblem
	from request import SimulationRequest


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


__all__ = ['NumericalMethod']
