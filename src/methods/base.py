"""Contract implemented by reusable numerical methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import is_dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar, Generic, Literal, Protocol, Self, TYPE_CHECKING, TypeVar, runtime_checkable

import numpy as np

from contracts.result import DiagnosticValue
from contracts.step import StepInfo, StepResult
from formulations.state import PhysicalFormulation

from problem import InitialValueProblem
from dynamics.protocols import DynamicalSystem
from request import SimulationRequest

if TYPE_CHECKING:
	from execution._host import StepController


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
	"""A numerical algorithm whose run instances own their formulation and resources.

	Public dataclass constructors describe reusable controls. ``new_run`` creates
	another instance of the same class and initializes it for one problem. No
	context object or collection of bound callbacks sits between it and the driver.
	"""

	initial_state: np.ndarray
	metadata: Mapping[str, DiagnosticValue]
	problem: InitialValueProblem
	request: SimulationRequest
	progress: bool
	step_observer: Callable[[Any], None] | None
	_status: str = "configuration"
	state_formulation: PhysicalFormulation | None = None
	# Capabilities select a driver; methods never resolve an execution device.
	jax_mode: ClassVar[Literal["unsupported", "device_resident", "hybrid_scipy_controller"]] = "unsupported"
	evaluated_dynamics: DynamicalSystem

	@property
	def method_name(self) -> str:
		return type(self).__name__

	def new_run(self, problem: InitialValueProblem, request: SimulationRequest,
	            *, evaluated_dynamics: DynamicalSystem | None = None) -> Self:
		"""Create an isolated run of this class, copying only constructor options.

		Dataclass replacement reconstructs the configuration and excludes all
		``init=False`` resources. Observers and immutable problem data are shared
		intentionally; solver state and accumulated histories are never copied.
		"""
		if not is_dataclass(self):
			raise TypeError("Numerical methods must declare their options as dataclass fields.")
		method: Self = replace(self)
		method.evaluated_dynamics = problem.dynamics if evaluated_dynamics is None else evaluated_dynamics
		method.problem = problem
		method.request = request
		method.metadata = {}
		method.initialize(problem, request)
		if method.state_formulation is not None:
			# Keep algorithm metadata on the method and state-layout metadata on
			# the formulation, then expose both through this run's single mapping.
			method.metadata = {**method.metadata, **method.state_formulation.metadata()}
		method.initial_state = _readonly_initial_state(method.initial_state)
		method.metadata = _readonly_metadata(method.metadata)
		method._status = "ready"
		return method

	@abstractmethod
	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Initialize this run's formulation and initial state; called by new_run."""

	@abstractmethod
	def advance(self, time: float, state: np.ndarray, step: float) -> StepResult[Detail]:
		"""Advance without observing or collecting; return local work and details.

		Fixed methods evaluate a map of duration ``step`` on the supplied state.
		Adaptive methods advance their live solver with ``step`` as an upper bound;
		their controller reports the actual accepted endpoint.
		"""

	@abstractmethod
	def build_observation(self, info: StepInfo, result: StepResult[Detail]) -> Any:
		"""Construct an independent method-specific event for an accepted step."""

	def export_history(self, times: np.ndarray, history: np.ndarray) -> tuple[np.ndarray, dict[str, DiagnosticValue]]:
		"""Extract physical samples and auxiliary diagnostics from internal history."""
		if self.state_formulation is None:
			return history, {}
		return self.state_formulation.extract_history(times, history)

	def controller(self) -> StepController[Detail]:
		"""Select fixed scheduling; adaptive methods override this operation."""
		from execution._host import FixedStepController
		return FixedStepController()


__all__ = ["NumericalMethod", "IntegrationMethod"]
