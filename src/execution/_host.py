# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Host time stepping, sampling and result collection inside execution.

Controllers provide accepted intervals and samples; numerical maps remain in
methods. This private driver serves fixed and adaptive runs without method dispatch.
"""

from __future__ import annotations

from contracts.execution.controllers import StepController

from collections.abc import Callable, Iterator, Sequence
from typing import Any, Generic, TYPE_CHECKING, TypeVar
from dataclasses import replace

import math
import sys

import numpy as np

from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from contracts.methods.observation import IntegrationStep
from contracts.methods.quadrature import StageQuadrature
from contracts.dynamics.protocols import DynamicalSystem
from execution.state import IntegrationState
from request import SimulationRequest
from states.history import States

if TYPE_CHECKING:
	from diagnostics.methods.energy import PreparedEnergyObserver
	from methods.base import IntegrationMethod


Detail = TypeVar("Detail")
class _Progress:
	"""Small stderr progress indicator counting complete integration steps."""

	def __init__(self, label: str, total: int, *, t_span: tuple[float, float] | None = None) -> None:
		self.label = label
		self.total = max(total, 1)
		self.every = max(self.total // 100, 1)
		self.steps = 0
		self.t_span = t_span
		self._last_percent = -1

	def update(self, t: float) -> None:
		"""Advance and occasionally redraw the same terminal line."""
		self.steps += 1
		if self.t_span is None and self.steps % self.every and self.steps < self.total:
			return
		fraction: float = min(self.steps / self.total, 1.0)
		if self.t_span is not None:
			start: float
			end: float
			start, end = self.t_span
			fraction = min(max((t - start) / (end - start), 0.0), 1.0)
		percent: int = int(100 * fraction)
		if percent == self._last_percent:
			return
		self._last_percent = percent
		width = 30
		filled: int = int(width * fraction)
		bar = "=" * filled
		if filled < width:
			bar += ">" + " " * (width - filled - 1)
		print(
			f"\r{self.label} [{bar}] {fraction:6.1%} "
			+ (f"({self.steps} accepted steps, t={t:.6g})" if self.t_span is not None
			 else f"({self.steps}/{self.total}, t={t:.6g})"),
			end="",
			file=sys.stderr,
			flush=True,
		)

	def close(self) -> None:
		"""Terminate the in-place progress display."""
		print(file=sys.stderr, flush=True)


def _step_count(duration: float, step: float) -> int:
	"""Count nominal steps, allowing round-off at an integer quotient."""
	ratio: float = duration / step
	if not math.isfinite(ratio):
		raise ValueError("The time span and step produce an unrepresentable step count.")
	nearest: int = round(ratio)
	if abs(ratio - nearest) <= 8 * np.finfo(float).eps * max(1.0, ratio):
		return max(1, nearest)
	return max(1, math.ceil(ratio))


def _fixed_schedule(t_span: tuple[float, float], step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Build a nominal grid with only its final duration clipped to the endpoint.

	Integer-indexed starts avoid cumulative clock drift. All complete maps use
	the configured duration; tiny clock differences remain float round-off.
	"""
	t0: float
	tf: float
	t0, tf = t_span
	if not np.isfinite(step) or step <= 0:
		raise ValueError("`step` must be positive and finite.")
	if t0 + min(step, tf - t0) <= t0:
		raise ValueError("The configured step cannot advance representable time.")
	count: int = _step_count(tf - t0, step)
	starts: np.ndarray = t0 + np.arange(count) * step
	ends: np.ndarray = np.concatenate((starts[1:], np.asarray([tf])))
	durations: np.ndarray = np.full(count, step)
	durations[-1] = tf - starts[-1]
	if np.any(ends <= starts):
		raise ValueError("The configured step cannot advance representable time.")
	return starts, ends, durations


def _time_tolerance(t_span: tuple[float, float]) -> float:
	"""Allow float round-off when matching requested and accepted times."""
	t0: float
	tf: float
	t0, tf = t_span
	return float(16 * np.finfo(float).eps * max(1.0, abs(t0), abs(tf)))


class FixedStepController(Generic[Detail]):
	"""Use a fixed nominal step, a clipped final step and independent shadow maps."""

	def steps(
		self, method: IntegrationMethod[Detail], request: SimulationRequest,
	) -> Iterator[tuple[StepInfo, StepResult[Detail]]]:
		"""Advance main nodes without observing, collecting or exporting data."""
		from methods.base import FixedStepMethod

		if not isinstance(method, FixedStepMethod):
			raise TypeError("FixedStepController requires a FixedStepMethod.")
		starts: np.ndarray
		ends: np.ndarray
		durations: np.ndarray
		starts, ends, durations = _fixed_schedule(request.t_span, method.step)
		current: IntegrationState = IntegrationState(request.t_span[0], method.initial_state)
		for index, (time, end, duration) in enumerate(zip(starts, ends, durations)):
			info: StepInfo = StepInfo(index, float(time), float(duration), float(end), current.values.copy())
			result: StepResult[Detail] = method.advance(info.time, current.values.copy(), info.duration)
			current = IntegrationState(info.end_time, result.state)
			yield info, result

	def sample(
		self, method: IntegrationMethod[Detail], info: StepInfo,
		result: StepResult[Detail], time: float,
	) -> IntegrationState:
		"""Reuse endpoints or advance an independent physical/auxiliary snapshot."""
		tolerance: float = _time_tolerance(method.request.t_span)
		if abs(time - info.end_time) <= tolerance:
			return IntegrationState(time, result.state, result.quadrature)
		if abs(time - info.time) <= tolerance:
			return IntegrationState(time, info.state_before)
		shadow: StepResult[Detail] = method.advance(info.time, info.state_before.copy(), time - info.time)
		return IntegrationState(time, shadow.state, shadow.quadrature)


def _record_sample(method: IntegrationMethod[Any], result: States, sample: IntegrationState,
	energy: PreparedEnergyObserver | None, *, sample_time: float | None = None,
) -> None:
	"""Record physical values and independent observer-owned auxiliaries."""
	method._checked_state(sample.values)
	auxiliary: dict[str, np.ndarray] = {}
	if energy is not None:
		quadrature = sample.quadrature if isinstance(sample.quadrature, StageQuadrature) else None
		auxiliary['extended_momentum'] = energy.sample(sample.time if sample_time is None else sample_time, quadrature)
	result.record(sample.time, sample.values, auxiliary=auxiliary)


def _observation(method: IntegrationMethod[Any], info: StepInfo, result: StepResult[Any],
	source: DynamicalSystem,
) -> Any:
	"""Attach source identity at the execution boundary, including nested stages."""
	event = method.build_observation(info, result)
	updates: dict[str, Any] = {'dynamics_name': type(source).__name__}
	if isinstance(event, IntegrationStep):
		quadrature = result.quadrature
		if isinstance(quadrature, StageQuadrature):
			quadrature = StageQuadrature(quadrature.times, tuple(np.array(value, copy=True) for value in quadrature.states), quadrature.weights, quadrature.reduction_scale)
		updates.update(dynamics=source, quadrature=quadrature)
		if hasattr(event, 'base_stages'):
			updates['base_stages'] = tuple(replace(stage, dynamics=source,
				dynamics_name=type(source).__name__) for stage in event.base_stages)
	return replace(event, **updates)


def run_steps(
	method: IntegrationMethod[Detail], saved: States, *,
	source: DynamicalSystem, observers: Sequence[Callable[[Any], None]] = (),
	energy: PreparedEnergyObserver | None = None,
) -> dict[str, DiagnosticValue]:
	"""Record accepted steps and samples in the executor's open result.

	Controllers own step policy; the executor owns run status and finalization.
	Observer lifecycle and persistence remain owned by the caller.
	"""
	request = method.request
	selected: StepController[Detail] = method.controller()
	_record_sample(method, saved, IntegrationState(request.t_span[0], method.initial_state), energy)
	t0: float
	tf: float
	t0, tf = request.t_span
	# Allow only float round-off when comparing interval and sampling endpoints;
	# this is unrelated to an integrator's error or nonlinear-solver tolerance.
	tolerance: float = _time_tolerance(request.t_span)
	sampling_tolerance: Any | float = getattr(selected, 'sampling_tolerance', tolerance)
	# Progress uses accepted time for both fixed and adaptive controllers.
	progress: _Progress | None = _Progress(method.method_name, 1, t_span=request.t_span) if method.progress else None
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
			method._checked_state(result.state)
			saved.record_step(info, result.statistics)
			if energy is not None:
				if result.quadrature is None:
					raise ValueError("Energy observation requires accepted method quadrature.")
				energy.accept(info.index, info.time, info.end_time, result.quadrature)
			if observers:
				event = _observation(method, info, result, source)
				for observer in observers:
					observer(event)
			# One accepted step can cover several requested output times.
			while (
				output_index < request.output_times.size
				and request.output_times[output_index] <= info.end_time + sampling_tolerance
			):
				target: float = float(request.output_times[output_index])
				sample: IntegrationState = selected.sample(method, info, result, target)
				if sample.time != target:
					raise ValueError("The controller returned a different sample time.")
				query = info.end_time if abs(target - info.end_time) <= sampling_tolerance else (
					info.time if abs(target - info.time) <= sampling_tolerance else target)
				_record_sample(method, saved, sample, energy, sample_time=query)
				if min(abs(target - info.time), abs(target - info.end_time)) > sampling_tolerance:
					interior_sample_count += 1
				output_index += 1
			previous_end = info.end_time
			if progress is not None:
				progress.update(info.end_time)
	finally:
		if progress is not None:
			progress.close()
	if output_index != request.output_times.size or abs(previous_end - tf) > tolerance:
		raise RuntimeError("The integration did not cover the requested interval.")
	diagnostics: dict[str, DiagnosticValue] = {}
	if energy is not None:
		diagnostics.update(energy.diagnose_history(saved.times, saved.values, dict(saved.auxiliary)))
	diagnostics["output_interpolation_count"] = interior_sample_count
	return diagnostics


__all__: list[str] = []
