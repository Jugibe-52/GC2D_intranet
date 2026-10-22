# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Host time stepping, sampling and result collection inside execution.

Controllers provide accepted intervals and samples; numerical maps remain in
methods. This private driver serves fixed and adaptive runs without method dispatch.
"""

from __future__ import annotations

from contracts.execution.controllers import StepController

from collections.abc import Iterator
from typing import Any, Generic, TYPE_CHECKING, TypeVar

import math
import sys

import numpy as np

from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from execution.state import IntegrationState
from request import SimulationRequest
from states.history import States

if TYPE_CHECKING:
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


def _step_count(duration: float, max_step: float) -> int:
	"""Return the fewest uniform steps respecting ``max_step``."""
	ratio = duration / max_step
	return max(1, math.ceil(math.nextafter(ratio, -math.inf)))


def _time_tolerance(t_span: tuple[float, float]) -> float:
	"""Allow float round-off when matching requested and accepted times."""
	t0: float
	tf: float
	t0, tf = t_span
	return float(16 * np.finfo(float).eps * max(1.0, abs(t0), abs(tf)))


class FixedStepController(Generic[Detail]):
	"""Use the established uniform grid and independent shortened shadow maps."""

	def steps(
		self, method: IntegrationMethod[Detail], request: SimulationRequest,
	) -> Iterator[tuple[StepInfo, StepResult[Detail]]]:
		"""Advance main nodes without observing, collecting or exporting data."""
		t0: float
		tf: float
		t0, tf = request.t_span
		count: int = _step_count(tf - t0, request.max_step)
		h = (tf - t0) / count
		current: IntegrationState = IntegrationState(t0, method.initial_state)
		for index in range(count):
			time = current.time
			info: StepInfo = StepInfo(index, time, h, t0 + (index + 1) * h, current.values.copy())
			result: StepResult[Detail] = method.advance(time, current.values.copy(), h)
			current = IntegrationState(info.end_time, result.state)
			yield info, result

	def sample(
		self, method: IntegrationMethod[Detail], info: StepInfo,
		result: StepResult[Detail], time: float,
	) -> np.ndarray:
		"""Reuse accepted endpoints or take an independent partial step."""
		tolerance: float = _time_tolerance(method.request.t_span)
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
		physical: np.ndarray
		auxiliary: dict[str, np.ndarray]
		physical, auxiliary = method.state_formulation.extract_sample(time, sample)
		result.record(time, physical, auxiliary=auxiliary)


def run_steps(
	method: IntegrationMethod[Detail], saved: States,
) -> dict[str, DiagnosticValue]:
	"""Record accepted steps and samples in the executor's open result.

	Controllers own step policy; the executor owns run status and finalization.
	Observer lifecycle and persistence remain owned by the caller.
	"""
	request = method.request
	selected: StepController[Detail] = method.controller()
	_record_sample(method, saved, request.t_span[0], method.initial_state)
	t0: float
	tf: float
	t0, tf = request.t_span
	# Allow only float round-off when comparing interval and sampling endpoints;
	# this is unrelated to an integrator's error or nonlinear-solver tolerance.
	tolerance: float = _time_tolerance(request.t_span)
	sampling_tolerance: Any | float = getattr(selected, 'sampling_tolerance', tolerance)
	# Estimate the fixed-step count for the display; its percentage uses accepted time.
	progress: _Progress | None = _Progress(method.method_name, _step_count(tf - t0, request.max_step), t_span=request.t_span) if method.progress else None
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
				target: float = float(request.output_times[output_index])
				sample: np.ndarray = np.asarray(selected.sample(method, info, result, target))
				_record_sample(method, saved, target, sample)
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
	if method.state_formulation is not None:
		diagnostics.update(method.state_formulation.diagnose_history(saved.times, saved.values, dict(saved.auxiliary)))
	diagnostics["output_interpolation_count"] = interior_sample_count
	return diagnostics


__all__: list[str] = []
