"""Structural preparation contract for function-based numerical methods."""

from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable
from contracts.methods.functions import VectorField
if TYPE_CHECKING:
	from methods.base import IntegrationMethod
	from states.initial import InitialConditions


@runtime_checkable
class NumericalMethod(Protocol):
	"""Prepare an isolated numerical run without exposing physical dynamics."""

	def new_run(
		self, f: VectorField, initial_conditions: InitialConditions,
	) -> IntegrationMethod[Any]: ...


__all__ = ['NumericalMethod']
