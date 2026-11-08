"""Reusable numerical algorithms bound to functions and physical initial data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar, Generic, Self, TYPE_CHECKING, TypeVar

import numpy as np
from types import ModuleType
from contracts.arrays import array_namespace, checked_finite

from contracts.methods.base import NumericalMethod
from contracts.methods.functions import VectorField
from contracts.methods.step import StepInfo, StepResult
from contracts.states.history import DiagnosticValue
from states.initial import InitialConditions
from request import DEFAULT_INTEGRATION_STEP
from methods._validation import _positive_finite

if TYPE_CHECKING:
	from contracts.execution.controllers import StepController

Detail = TypeVar("Detail")


def _readonly_metadata(metadata: Mapping[str, DiagnosticValue]) -> Mapping[str, DiagnosticValue]:
	"""Copy metadata arrays so reusable run preparation cannot mutate them."""
	owned: dict[str, np.ndarray | float | int | str] = dict(metadata)
	value: np.ndarray | float | int | str
	for name, value in owned.items():
		if isinstance(value, np.ndarray):
			value = value.copy()
			value.setflags(write=False)
			owned[name] = value
	return MappingProxyType(owned)


class IntegrationMethod(ABC, Generic[Detail]):
	"""A numerical algorithm with fresh resources for each function binding."""

	f: VectorField
	jacobian: VectorField | None
	metadata: Mapping[str, DiagnosticValue]
	_status: str = "configuration"
	copies: ClassVar[int] = 1

	@property
	def method_name(self) -> str:
		return type(self).__name__

	def new_run(
		self, f: VectorField, initial_conditions: InitialConditions,
		*, jacobian: VectorField | None = None,
	) -> Self:
		"""Bind fresh numerical resources without retaining physical initial data."""
		if not is_dataclass(self):
			raise TypeError("Numerical methods must declare their options as dataclass fields.")
		if not callable(f) or (jacobian is not None and not callable(jacobian)):
			raise TypeError("Supply a field function and an optional Jacobian function.")
		if not isinstance(initial_conditions, InitialConditions):
			raise TypeError("Supply physical InitialConditions for method preparation.")
		method: Self = replace(self)
		method.f = f
		method.jacobian = jacobian
		method.metadata = {}
		method.initialize(initial_conditions)
		method.metadata = _readonly_metadata({**method.metadata, **method._state_metadata(initial_conditions.values.size)})
		method._status = "ready"
		return method

	@abstractmethod
	def initialize(self, initial_conditions: InitialConditions) -> None:
		"""Prepare method-specific work after the common function binding."""

	@abstractmethod
	def advance(self, time: Any, state: Any, step: Any, *, observe: bool = True) -> StepResult[Detail]:
		"""Return a physical step and optional quadrature data without observing it."""

	@abstractmethod
	def build_observation(self, info: StepInfo, result: StepResult[Detail]) -> Any:
		"""Build numerical snapshots; execution supplies physical provenance."""

	def _state_metadata(self, size: int) -> dict[str, DiagnosticValue]:
		"""Describe physical storage and temporary copies independently of observers."""
		return {
			'state_formulation': 'physical' if self.copies == 1 else 'duplicated',
			'accepted_internal_state_dimension': size,
			'base_splitting_state_dimension': self.copies * size,
			'spatial_state_dimension': self.copies * size,
			'workspace_state_dimension': self.copies * size,
			'time_state_dimension': 1,
			'observer_state_dimension': size,
			'observer_state_kind': 'physical_map', 'energy_feedback': False,
		}

	def _checked_state(self, state: Any) -> Any:
		"""Validate a finite, nonempty physical vector."""
		xp: ModuleType = array_namespace(state)
		physical: np.ndarray = xp.asarray(state)
		physical = checked_finite(physical, 'A step requires a finite nonempty physical vector.', ValueError)
		if physical.ndim != 1 or not physical.size:
			raise ValueError('A step requires a finite nonempty physical vector.')
		return physical

	def controller(self) -> StepController[Detail]:
		"""Select fixed scheduling; extensions may supply another controller."""
		from execution._host import FixedStepController
		return FixedStepController()


@dataclass(kw_only=True)
class FixedStepMethod(IntegrationMethod[Detail], Generic[Detail]):
	"""Reusable fixed-step configuration in the problem's runtime time units.

	The controller clips only the final accepted interval. Shadow sampling and
	final clipping supply effective durations to advance without changing step.
	"""

	step: float = DEFAULT_INTEGRATION_STEP

	def __post_init__(self) -> None:
		"""Reject invalid nominal steps before constructing a run."""
		self.step = _positive_finite(self.step, "step")


__all__ = ["NumericalMethod", "IntegrationMethod", "FixedStepMethod"]
