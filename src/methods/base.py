"""Reusable numerical algorithms bound to functions and physical initial data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar, Generic, Literal, Self, TYPE_CHECKING, TypeVar

import numpy as np

from contracts.dynamics.coordinates import StateLayout
from contracts.methods.base import NumericalMethod
from contracts.methods.functions import JacobianStructure, VectorField
from contracts.methods.step import StepInfo, StepResult
from contracts.states.history import DiagnosticValue
from states.initial import InitialConditions
from request import DEFAULT_INTEGRATION_STEP, SimulationRequest
from methods._validation import _positive_finite

if TYPE_CHECKING:
	from contracts.execution.controllers import StepController

Detail = TypeVar("Detail")


def _readonly_initial_state(state: np.ndarray) -> np.ndarray:
	"""Own a finite, nonempty internal vector before a run becomes ready."""
	initial: np.ndarray = np.array(state, dtype=float, copy=True)
	if initial.ndim != 1 or initial.size == 0 or not np.all(np.isfinite(initial)):
		raise ValueError("The initial internal state must be a finite vector.")
	initial.setflags(write=False)
	return initial


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
	jacobian_structure: JacobianStructure
	initial_state: np.ndarray
	layout: StateLayout
	physical_size: int
	particle_count: int
	metadata: Mapping[str, DiagnosticValue]
	request: SimulationRequest
	progress: bool
	observe: bool = False
	capture_quadrature: bool = False
	_status: str = "configuration"
	copies: ClassVar[int] = 1
	requires_planar: ClassVar[bool] = False
	jax_mode: ClassVar[Literal["unsupported", "device_resident", "hybrid_scipy_controller"]] = "unsupported"

	@property
	def method_name(self) -> str:
		return type(self).__name__

	def new_run(
		self, f: VectorField, initial_conditions: InitialConditions, request: SimulationRequest,
		*, jacobian: VectorField | None = None, jacobian_structure: JacobianStructure = "dense",
		observe: bool = False, capture_quadrature: bool = False,
	) -> Self:
		"""Copy constructor controls and own physical input; never retain a problem."""
		if not is_dataclass(self):
			raise TypeError("Numerical methods must declare their options as dataclass fields.")
		if not callable(f) or (jacobian is not None and not callable(jacobian)):
			raise TypeError("Supply a field function and an optional Jacobian function.")
		if not isinstance(initial_conditions, InitialConditions):
			raise TypeError("Supply physical InitialConditions for method preparation.")
		if jacobian_structure not in ("dense", "particle"):
			raise ValueError("Jacobian structure must be dense or particle.")
		method: Self = replace(self)
		method.f = f
		method.jacobian = jacobian if jacobian is not None else getattr(method, "jacobian", None)
		method.jacobian_structure = jacobian_structure
		method.layout = initial_conditions.layout
		method.initial_state = _readonly_initial_state(initial_conditions.values)
		method.physical_size = method.initial_state.size
		method.particle_count = initial_conditions.particle_count
		method.request = request
		method.observe, method.capture_quadrature = bool(observe), bool(capture_quadrature)
		method.metadata = {}
		method.initialize(request)
		method.metadata = _readonly_metadata({**method.metadata, **method._state_metadata()})
		method._status = "ready"
		return method

	@abstractmethod
	def initialize(self, request: SimulationRequest) -> None:
		"""Prepare method-specific work after the common function binding."""

	@abstractmethod
	def advance(self, time: float, state: np.ndarray, step: float, *, observe: bool = True) -> StepResult[Detail]:
		"""Return a physical step and optional quadrature data without observing it."""

	@abstractmethod
	def build_observation(self, info: StepInfo, result: StepResult[Detail]) -> Any:
		"""Build numerical snapshots; execution supplies physical provenance."""

	def _state_metadata(self) -> dict[str, DiagnosticValue]:
		"""Describe physical storage and temporary copies independently of observers."""
		return {
			'state_formulation': 'physical' if self.copies == 1 else 'duplicated',
			'accepted_internal_state_dimension': self.physical_size,
			'base_splitting_state_dimension': self.copies * self.physical_size,
			'spatial_state_dimension': self.copies * self.physical_size,
			'workspace_state_dimension': self.copies * self.physical_size,
			'time_state_dimension': 1,
			'observer_state_dimension': self.physical_size,
			'observer_state_kind': 'physical_map', 'energy_feedback': False,
		}

	def _checked_state(self, state: np.ndarray) -> np.ndarray:
		"""Reject packed workspaces and nonfinite physical inputs."""
		physical: np.ndarray = np.asarray(state)
		if physical.shape != self.initial_state.shape or not np.all(np.isfinite(physical)):
			raise ValueError('A step requires a finite physical vector with the initial shape.')
		return physical

	def controller(self) -> StepController[Detail]:
		"""Select fixed scheduling; adaptive methods supply their live controller."""
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
