"""Contract implemented by reusable numerical methods."""

from __future__ import annotations

from contracts.methods.base import NumericalMethod

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, is_dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar, Generic, Literal, Self, TYPE_CHECKING, TypeVar

import numpy as np

from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from formulations.state import SingleFormulation
from states.energy import energy_diagnostics
from contracts.dynamics.protocols import HamiltonianSystem

from problem import InitialValueProblem
from contracts.dynamics.protocols import DynamicalSystem
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
	"""A numerical algorithm whose run instances own their formulation and resources.

	Public dataclass constructors describe reusable controls. ``new_run`` creates
	another instance of the same class and initializes it for one problem. No
	context object or collection of bound callbacks sits between it and the driver.
	"""

	initial_state: np.ndarray
	initial_momentum: np.ndarray | None = None
	track_energy: bool = False
	metadata: Mapping[str, DiagnosticValue]
	problem: InitialValueProblem
	request: SimulationRequest
	progress: bool
	step_observer: Callable[[Any], None] | None
	_status: str = "configuration"
	state_formulation: SingleFormulation | None = None
	_energy_dynamics: HamiltonianSystem | None = None
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
			# Report accepted physical size separately from temporary workspace size.
			method.metadata = {**method.metadata, **method._state_metadata()}
		method.initial_state = _readonly_initial_state(method.initial_state)
		if method.initial_momentum is not None:
			method.initial_momentum = _readonly_initial_state(method.initial_momentum)
		method.metadata = _readonly_metadata(method.metadata)
		method._status = "ready"
		return method

	@abstractmethod
	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Initialize this run's formulation and initial state; called by new_run."""

	@abstractmethod
	def advance(self, time: float, state: np.ndarray, step: float, *, momentum: np.ndarray | None = None) -> StepResult[Detail]:
		"""Advance without observing or collecting; return local work and details.

		Fixed methods evaluate a map of duration ``step`` on the supplied state.
		Adaptive methods advance their live solver with ``step`` as an upper bound;
		their controller reports the actual accepted endpoint.
		"""

	@abstractmethod
	def build_observation(self, info: StepInfo, result: StepResult[Detail]) -> Any:
		"""Construct an independent method-specific event for an accepted step."""

	def _state_metadata(self) -> dict[str, DiagnosticValue]:
		"""Describe actual accepted storage, temporary work and separate auxiliaries."""
		assert self.state_formulation is not None
		name: str = 'physical' if self.state_formulation.copies == 1 else 'duplicated'
		return {
			'state_formulation': name + ('_with_energy' if self.track_energy else ''),
			'track_energy': self.track_energy,
			'accepted_internal_state_dimension': self.state_formulation.physical_size,
			'base_splitting_state_dimension': self.state_formulation.spatial_size,
			'spatial_state_dimension': self.state_formulation.spatial_size,
			'workspace_state_dimension': self.state_formulation.spatial_size,
			'time_state_dimension': 1,
			'momentum_state_dimension': self.problem.particle_count if self.track_energy else 0,
			'observer_state_dimension': self.state_formulation.physical_size,
			'observer_state_kind': 'physical_map', 'energy_feedback': False,
		}

	def _initialize_state(self, problem: InitialValueProblem, request: SimulationRequest,
		*, energy_dynamics: DynamicalSystem | None = None,
	) -> np.ndarray:
		"""Own physical initial values and a separate zero physical momentum."""
		dynamics: DynamicalSystem = problem.dynamics if energy_dynamics is None else energy_dynamics
		self._energy_dynamics = None
		self.initial_momentum = None
		if self.track_energy:
			if not isinstance(dynamics, HamiltonianSystem):
				raise TypeError('Energy tracking requires HamiltonianSystem.')
			self._energy_dynamics = dynamics
			self.initial_momentum = np.zeros(problem.particle_count)
		return problem.initial_state

	def _checked_state(self, state: np.ndarray, momentum: np.ndarray | None) -> np.ndarray:
		"""Reject packed copies or missing auxiliaries before numerical work."""
		physical: np.ndarray = np.asarray(state)
		if physical.shape != self.initial_state.shape or not np.all(np.isfinite(physical)):
			raise ValueError('A step requires a finite physical vector with the initial shape.')
		if self.track_energy:
			if momentum is None or momentum.shape != (self.problem.particle_count,) or not np.all(np.isfinite(momentum)):
				raise ValueError('Supply finite momentum with one value per particle for an energy-tracked step.')
		elif momentum is not None:
			raise ValueError('Momentum requires energy tracking.')
		return physical

	def _momentum_rate(self, time: float, physical: np.ndarray) -> np.ndarray:
		"""Evaluate -partial_t H in the physical Hamiltonian normalization."""
		assert self._energy_dynamics is not None
		rate: np.ndarray = np.asarray(self._energy_dynamics.extended_momentum_derivative(time, physical), dtype=float)
		if rate.shape != (self.problem.particle_count,) or not np.all(np.isfinite(rate)):
			raise ValueError('The momentum derivative must be finite with one value per particle.')
		return rate

	def _finish_momentum(self, momentum: np.ndarray | None, increment: np.ndarray | None) -> np.ndarray | None:
		"""Accumulate accepted-stage quadrature without retaining mutable run state."""
		if not self.track_energy:
			return None
		if momentum is None or increment is None:
			raise ValueError('An energy-tracked step requires momentum and its increment.')
		return np.asarray(momentum + increment)

	def diagnose_history(self, times: np.ndarray, physical: np.ndarray,
		auxiliary: dict[str, np.ndarray], *, energy: np.ndarray | None = None,
	) -> dict[str, DiagnosticValue]:
		"""Evaluate saved physical energy once and assemble supplied auxiliaries."""
		if 'extended_momentum' not in auxiliary:
			return {}
		if energy is None:
			assert self._energy_dynamics is not None
			energy = np.asarray(self._energy_dynamics.hamiltonian(times, physical), dtype=float)
		return energy_diagnostics(times, physical, auxiliary, energy=energy)

	def controller(self) -> StepController[Detail]:
		"""Select fixed scheduling; adaptive methods override this operation."""
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
