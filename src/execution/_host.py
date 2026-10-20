"""Host time stepping, sampling and result collection inside execution.

Controllers provide accepted intervals and samples; numerical maps remain in
methods. This private driver serves fixed and adaptive runs without method dispatch.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, Generic, Protocol, TYPE_CHECKING, TypeVar

import numpy as np

from contracts.result import DiagnosticValue
from contracts.step import StepInfo, StepResult
from execution.state import IntegrationState
from execution._fixed import _Progress, _step_count
from request import SimulationRequest
from states.history import States

if TYPE_CHECKING:
	from methods.base import IntegrationMethod


Detail = TypeVar("Detail")
NEWTON_ALIASES: Mapping[str, str] = MappingProxyType({
	"newton_iterations": "nonlinear_iterations",
	"newton_residual_norms": "nonlinear_residual_norms",
	"newton_absolute_tolerance": "nonlinear_absolute_tolerance",
	"newton_relative_tolerance": "nonlinear_relative_tolerance",
	"newton_max_iterations": "nonlinear_max_iterations",
})


def _time_tolerance(t_span: tuple[float, float]) -> float:
	"""Allow float round-off when matching requested and accepted times."""
	t0, tf = t_span
	return float(16 * np.finfo(float).eps * max(1.0, abs(t0), abs(tf)))


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
	) -> np.ndarray:
		"""Return a state at any requested time in the interval, including its ends."""
		...


class FixedStepController(Generic[Detail]):
	"""Use the established uniform grid and independent shortened shadow maps."""

	def steps(
		self, method: IntegrationMethod[Detail], request: SimulationRequest,
	) -> Iterator[tuple[StepInfo, StepResult[Detail]]]:
		"""Advance main nodes without observing, collecting or exporting data."""
		t0, tf = request.t_span
		count = _step_count(tf - t0, request.max_step)
		h = (tf - t0) / count
		current = IntegrationState(t0, method.initial_state)
		for index in range(count):
			time = current.time
			info = StepInfo(index, time, h, t0 + (index + 1) * h, current.values.copy())
			result = method.advance(time, current.values.copy(), h)
			current = IntegrationState(info.end_time, result.state)
			yield info, result

	def sample(
		self, method: IntegrationMethod[Detail], info: StepInfo,
		result: StepResult[Detail], time: float,
	) -> np.ndarray:
		"""Reuse accepted endpoints or take an independent partial step."""
		tolerance = _time_tolerance(method.request.t_span)
		if abs(time - info.end_time) <= tolerance:
			return result.state
		if abs(time - info.time) <= tolerance:
			return info.state_before
		return method.advance(info.time, info.state_before.copy(), time - info.time).state


def _record_sample(method: IntegrationMethod[Any], result: States, time: float, sample: np.ndarray) -> None:
	"""Export a physical sample before recording; never advance the work state."""
	if sample.shape != method.initial_state.shape or not np.all(np.isfinite(sample)):
		raise ValueError("An output sample changed shape or became non-finite.")
	if method.state_formulation is None:
		result.record(time, sample)
	else:
		physical, auxiliary = method.state_formulation.extract_sample(time, sample)
		result.record(time, physical, auxiliary=auxiliary)


def integrate_method(
	method: IntegrationMethod[Detail],
	*, controller: StepController[Detail] | None = None,
	metadata: Mapping[str, DiagnosticValue] | None = None,
) -> States:
	"""Coordinate accepted steps, sampling, observation and physical output.

	The supplied controller is per-run; the default is fixed. Neither the driver
	nor the result container branches on the numerical method or its state extension.
	Observer lifecycle and persistence remain owned by the caller.
	"""
	if method._status != "ready":
		raise RuntimeError("Integration requires a fresh method.new_run(problem, request).")
	request = method.request
	selected = controller if controller is not None else method.controller()
	method._status = "running"
	saved = States.prepare(request.output_times, method.problem.initial_state, layout=method.problem.layout)
	_record_sample(method, saved, request.t_span[0], method.initial_state)
	t0, tf = request.t_span
	# Allow only float round-off when comparing interval and sampling endpoints;
	# this is unrelated to an integrator's error or nonlinear-solver tolerance.
	tolerance = _time_tolerance(request.t_span)
	sampling_tolerance = getattr(selected, 'sampling_tolerance', tolerance)
	# Estimate the fixed-step count for the display; its percentage uses accepted time.
	progress = _Progress(method.method_name, _step_count(tf - t0, request.max_step), t_span=request.t_span) if method.progress else None
	output_index = 1
	interior_sample_count = 0
	previous_end = t0
	try:
		for info, result in selected.steps(method, request):
			if (info.time != previous_end or info.end_time <= info.time
				or info.end_time > tf + tolerance):
				raise ValueError("The controller returned an invalid accepted interval.")
			if result.state.shape != method.initial_state.shape or not np.all(np.isfinite(result.state)):
				raise ValueError("A numerical step changed shape or became non-finite.")
			saved.record_step(info, result.statistics)
			if method.step_observer is not None:
				method.step_observer(method.build_observation(info, result))
			# One accepted step can cover several requested output times.
			while (
				output_index < request.output_times.size
				and request.output_times[output_index] <= info.end_time + sampling_tolerance
			):
				target = float(request.output_times[output_index])
				sample = np.asarray(selected.sample(method, info, result, target))
				_record_sample(method, saved, target, sample)
				if min(abs(target - info.time), abs(target - info.end_time)) > sampling_tolerance:
					interior_sample_count += 1
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
	diagnostics = {**method.metadata, **(metadata or {})}
	if method.state_formulation is not None:
		diagnostics.update(method.state_formulation.diagnose_history(saved.times, saved.values, dict(saved.auxiliary)))
	diagnostics["output_interpolation_count"] = interior_sample_count
	# Only methods publishing the complete nonlinear contract expose legacy
	# Newton names; a partial set belongs to other nonlinear algorithms.
	return saved.finalize(diagnostics, aliases=NEWTON_ALIASES)


__all__: list[str] = []
