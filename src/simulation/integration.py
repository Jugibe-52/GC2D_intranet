"""Method instances, step control and per-run collection for all integrators.

Numerical maps own a complete step. Controllers own the accepted time grid and
sampling rule. The coordinator owns collection, optional observation and output.
Fixed maps use the common controller; adaptive adapters supply a fresh controller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, is_dataclass, replace
from types import MappingProxyType
from typing import Any, Generic, Protocol, Self, TypeVar

import numpy as np

from ._fixed import _Progress, _step_count
from ._result import DiagnosticValue, IntegrationData
from .problem import InitialValueProblem
from .request import SimulationRequest
from .formulations.state import PhysicalFormulation


Detail = TypeVar("Detail")
StepValue = np.ndarray | float | int
NEWTON_ALIASES: Mapping[str, str] = MappingProxyType({
	"newton_iterations": "nonlinear_iterations",
	"newton_residual_norms": "nonlinear_residual_norms",
	"newton_absolute_tolerance": "nonlinear_absolute_tolerance",
	"newton_relative_tolerance": "nonlinear_relative_tolerance",
	"newton_max_iterations": "nonlinear_max_iterations",
})


@dataclass(frozen=True, slots=True)
class StepResult(Generic[Detail]):
	"""One complete internal-state map and its small, observer-free metrics.

	``state`` uses the method's packed internal coordinates. ``details``
	belongs to that method and is never interpreted or retained by the collector.
	Statistics describe this advance only; shadow statistics are discarded.
	"""

	state: np.ndarray
	statistics: Mapping[str, StepValue]
	details: Detail


@dataclass(frozen=True, slots=True)
class StepInfo:
	"""Accepted interval and an independent internal input snapshot."""

	index: int
	time: float
	duration: float
	end_time: float
	state_before: np.ndarray


class StepController(Protocol[Detail]):
	"""Deliver accepted intervals and evaluate output samples within them.

	An adaptive implementation must handle its retries internally and expose
	accepted intervals only. Its sampling operation can use dense output. The
	SciPy-backed methods provide adaptive control; other methods use fixed steps.
	"""

	def steps(
		self, method: IntegrationMethod[Detail], request: SimulationRequest,
	) -> Iterator[tuple[StepInfo, StepResult[Detail]]]: ...

	def sample(
		self, method: IntegrationMethod[Detail], info: StepInfo,
		result: StepResult[Detail], time: float,
	) -> np.ndarray: ...


class FixedStepController(Generic[Detail]):
	"""Use the established uniform grid and independent shortened shadow maps."""

	def steps(
		self, method: IntegrationMethod[Detail], request: SimulationRequest,
	) -> Iterator[tuple[StepInfo, StepResult[Detail]]]:
		"""Advance main nodes without observing, collecting or exporting data."""
		t0, tf = request.t_span
		count = _step_count(tf - t0, request.max_step)
		h = (tf - t0) / count
		state = method.initial_state.copy()
		for index in range(count):
			time = t0 + index * h
			info = StepInfo(index, time, h, t0 + (index + 1) * h, state.copy())
			result = method.advance(time, state, h)
			state = np.asarray(result.state)
			yield info, result

	def sample(
		self, method: IntegrationMethod[Detail], info: StepInfo,
		result: StepResult[Detail], time: float,
	) -> np.ndarray:
		"""Compute a sample without touching main state, metrics or observers."""
		return method.advance(info.time, info.state_before.copy(), time - info.time).state


class IntegrationCollector:
	"""Own output samples and small per-step metrics for exactly one run."""

	def __init__(self, initial: np.ndarray, times: np.ndarray) -> None:
		self.history = np.empty((initial.size, times.size), dtype=initial.dtype)
		self.history[:, 0] = initial
		self._rows: dict[str, list[StepValue]] = {}
		self._shapes: dict[str, tuple[int, ...]] = {}
		self._starts: list[float] = []
		self._ends: list[float] = []
		self._sizes: list[float] = []

	def record_step(self, info: StepInfo, statistics: Mapping[str, StepValue]) -> None:
		"""Copy one stable metric schema, independently of observation events."""
		if not self._starts:
			self._rows = {key: [] for key in statistics}
			self._shapes = {key: np.shape(value) for key, value in statistics.items()}
		if statistics.keys() != self._rows.keys():
			raise ValueError("Step statistic names changed during integration.")
		for key, value in statistics.items():
			if np.shape(value) != self._shapes[key]:
				raise ValueError(f"The shape of step statistic {key!r} changed.")
			self._rows[key].append(value.copy() if isinstance(value, np.ndarray) else value)
		self._starts.append(info.time)
		self._ends.append(info.end_time)
		self._sizes.append(info.duration)

	def finalize(self) -> dict[str, DiagnosticValue]:
		"""Keep step times separate from the requested trajectory sample times."""
		result: dict[str, DiagnosticValue] = {
			key: np.asarray(values) for key, values in self._rows.items()
		}
		result.update({
			"step_count": len(self._starts),
			"step_start_times": np.asarray(self._starts),
			"step_times": np.asarray(self._ends),
			"step_sizes": np.asarray(self._sizes),
		})
		return result


def integrate_method(
	method: IntegrationMethod[Detail],
	*, controller: StepController[Detail] | None = None,
) -> IntegrationData:
	"""Coordinate accepted steps, sampling, observation and physical output.

	The supplied controller is per-run; the default is fixed. Neither the driver
	nor the collector branches on the numerical method or its state extension.
	Observer lifecycle and persistence remain owned by the caller.
	"""
	if method._status != "ready":
		raise RuntimeError("Integration requires a fresh method.new_run(problem, request).")
	request = method.request
	selected = controller if controller is not None else method.controller()
	method._status = "running"
	collector = IntegrationCollector(method.initial_state, request.output_times)
	t0, tf = request.t_span
	# Allow only float round-off when comparing interval and sampling endpoints;
	# this is unrelated to an integrator's error or nonlinear-solver tolerance.
	tolerance = 16 * np.finfo(float).eps * max(1.0, abs(t0), abs(tf))
	sampling_tolerance = getattr(selected, 'sampling_tolerance', tolerance)
	sample_endpoints = getattr(selected, 'sample_endpoints', False)
	# Estimate the fixed-step count for the display; its percentage uses accepted time.
	progress = _Progress(method.method_name, _step_count(tf - t0, request.max_step), t_span=request.t_span) if method.progress else None
	output_index = 1
	shadow_count = 0
	previous_end = t0
	try:
		for info, result in selected.steps(method, request):
			if (info.time != previous_end or info.end_time <= info.time
				or info.end_time > tf + tolerance):
				raise ValueError("The controller returned an invalid accepted interval.")
			if result.state.shape != method.initial_state.shape or not np.all(np.isfinite(result.state)):
				raise ValueError("A numerical step changed shape or became non-finite.")
			collector.record_step(info, result.statistics)
			if method.step_observer is not None:
				method.step_observer(method.build_observation(info, result))
			while output_index < request.output_times.size and request.output_times[output_index] <= info.end_time + sampling_tolerance:
				target = float(request.output_times[output_index])
				at_end = abs(target - info.end_time) <= sampling_tolerance
				at_start = abs(target - info.time) <= sampling_tolerance
				if at_end and not sample_endpoints:
					sample = result.state
				elif at_start and not sample_endpoints:
					sample = info.state_before
				else:
					sample = np.asarray(selected.sample(method, info, result, target))
					if not at_end and not at_start:
						shadow_count += 1
				if sample.shape != method.initial_state.shape or not np.all(np.isfinite(sample)):
					raise ValueError("An output sample changed shape or became non-finite.")
				collector.history[:, output_index] = sample
				output_index += 1
			previous_end = info.end_time
			if progress is not None:
				progress.update(info.end_time)
	finally:
		method._status = "finished"
		if progress is not None:
			progress.close()
	if output_index != request.output_times.size or abs(previous_end - tf) > tolerance:
		raise RuntimeError("The integration did not cover the requested interval.")
	states, auxiliary = method.export_history(request.output_times, collector.history)
	diagnostics = dict(method.metadata)
	diagnostics.update(collector.finalize())
	diagnostics.update(auxiliary)
	diagnostics["output_interpolation_count"] = shadow_count
	for alias, canonical in method.diagnostic_aliases.items():
		diagnostics[alias] = diagnostics[canonical]
	return IntegrationData(t=request.output_times, states=states, diagnostics=diagnostics)


class IntegrationMethod(ABC, Generic[Detail]):
	"""A numerical algorithm whose run instances own their formulation and resources.

	Public dataclass constructors describe reusable controls. ``new_run`` creates
	another instance of the same class and initializes it for one problem. No
	context object or collection of bound callbacks sits between it and the driver.
	"""

	initial_state: np.ndarray
	metadata: Mapping[str, DiagnosticValue]
	diagnostic_aliases: Mapping[str, str]
	problem: InitialValueProblem
	request: SimulationRequest
	progress: bool
	step_observer: Callable[[Any], None] | None
	_status: str = "configuration"
	state_formulation: PhysicalFormulation | None = None

	@property
	def method_name(self) -> str:
		return type(self).__name__

	def new_run(self, problem: InitialValueProblem, request: SimulationRequest) -> Self:
		"""Create an isolated run of this class, copying only constructor options.

		Dataclass replacement reconstructs the configuration and excludes all
		``init=False`` resources. Observers and immutable problem data are shared
		intentionally; solver state and accumulated histories are never copied.
		"""
		if not is_dataclass(self):
			raise TypeError("Numerical methods must declare their options as dataclass fields.")
		method = replace(self)
		method.problem = problem
		method.request = request
		method.metadata = {}
		method.diagnostic_aliases = {}
		method.initialize(problem, request)
		if method.state_formulation is not None:
			method.metadata = {**method.metadata, **method.state_formulation.metadata()}
		initial = np.array(method.initial_state, dtype=float, copy=True)
		if initial.ndim != 1 or initial.size == 0 or not np.all(np.isfinite(initial)):
			raise ValueError("The initial internal state must be a finite vector.")
		initial.setflags(write=False)
		method.initial_state = initial
		metadata = dict(method.metadata)
		for name, value in metadata.items():
			if isinstance(value, np.ndarray):
				value = value.copy()
				value.setflags(write=False)
				metadata[name] = value
		method.metadata = MappingProxyType(metadata)
		method.diagnostic_aliases = MappingProxyType(dict(method.diagnostic_aliases))
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
		return FixedStepController()

	def integrate(self, problem: InitialValueProblem, request: SimulationRequest) -> IntegrationData:
		"""Run a fresh instance through the shared controller and collector."""
		return integrate_method(self.new_run(problem, request))


__all__ = [
	"FixedStepController", "IntegrationMethod", "IntegrationCollector",
	"StepController", "StepInfo", "StepResult", "integrate_method",
]
