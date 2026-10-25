"""Structural preparation contract for function-based numerical methods."""

from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable
from contracts.methods.functions import JacobianStructure, VectorField
if TYPE_CHECKING:
	from methods.base import IntegrationMethod
	from states.initial import InitialConditions
	from request import SimulationRequest


@runtime_checkable
class NumericalMethod(Protocol):
	"""Prepare an isolated numerical run without exposing physical dynamics."""

	def new_run(
		self, f: VectorField, initial_conditions: InitialConditions, request: SimulationRequest,
		*, jacobian: VectorField | None = None, jacobian_structure: JacobianStructure = "dense",
		observe: bool = False, capture_quadrature: bool = False,
	) -> IntegrationMethod[Any]: ...


__all__ = ['NumericalMethod']
