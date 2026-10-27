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

import sys

import numpy as np

from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from contracts.methods.observation import IntegrationStep
from contracts.methods.quadrature import StageQuadrature
from contracts.dynamics.protocols import DynamicalSystem
from execution.state import IntegrationState
from execution._plan import build_fixed_step_plan, _time_tolerance, sample_modes
from contracts.execution.plan import FixedStepPlan, SAMPLE_START, SAMPLE_END, SAMPLE_SHADOW
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


class FixedStepController(Generic[Detail]):
	"""Use a fixed nominal step, a clipped final step and independent shadow maps."""

	def __init__(self, plan: FixedStepPlan | None = None) -> None:
		self.plan = plan

	def steps(
		self, method: IntegrationMethod[Detail], request: SimulationRequest,
	) -> Iterator[tuple[StepInfo, StepResult[Detail]]]:
		"""Advance main nodes without observing, collecting or exporting data."""
		from methods.base import FixedStepMethod

		if not isinstance(method, FixedStepMethod):
			raise TypeError("FixedStepController requires a FixedStepMethod.")
		plan: FixedStepPlan = self.plan if self.plan is not None else build_fixed_step_plan(request, method.step)
		current: IntegrationState = IntegrationState(request.t_span[0], method.initial_state)
		for index, (time, end, duration) in enumerate(zip(plan.step_starts, plan.step_ends, plan.step_sizes)):
			info: StepInfo = StepInfo(index, float(time), float(duration), float(end), current.values.copy())
			result: StepResult[Detail] = method.advance(info.time, current.values.copy(), info.duration)
			current = IntegrationState(info.end_time, np.asarray(result.state))
			yield info, result

	def sample(
		self, method: IntegrationMethod[Detail], info: StepInfo,
		result: StepResult[Detail], time: float,
	) -> IntegrationState:
		"""Reuse endpoints or advance an independent physical/auxiliary snapshot."""
		mode: int = int(sample_modes(np.asarray(time), np.asarray(info.time),
			np.asarray(info.end_time), _time_tolerance(method.request.t_span)))
		return self._sample(method, info, result, time, mode, time - info.time)

	def planned_sample(
		self, method: IntegrationMethod[Detail], info: StepInfo,
		result: StepResult[Detail], index: int,
	) -> IntegrationState:
		"""Apply a precomputed sampling decision without changing the main state."""
		assert self.plan is not None
		return self._sample(method, info, result, float(self.plan.output_times[index]),
			int(self.plan.sample_modes[index]), float(self.plan.sample_durations[index]))

	def _sample(
		self, method: IntegrationMethod[Detail], info: StepInfo,
		result: StepResult[Detail], time: float, mode: int, duration: float,
	) -> IntegrationState:
		if mode == SAMPLE_END:
			return IntegrationState(time, np.asarray(result.state), result.quadrature)
		if mode == SAMPLE_START:
			return IntegrationState(time, info.state_before)
		shadow: StepResult[Detail] = method.advance(info.time, info.state_before.copy(), duration)
		return IntegrationState(time, np.asarray(shadow.state), shadow.quadrature)


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
	source: DynamicalSystem, controller: StepController[Detail], observers: Sequence[Callable[[Any], None]] = (),
	energy: PreparedEnergyObserver | None = None,
) -> dict[str, DiagnosticValue]:
	"""Record accepted steps and samples in the executor's open result.

	Controllers own step policy; the executor owns run status and finalization.
	Observer lifecycle and persistence remain owned by the caller.
	"""
	request = method.request
	selected: StepController[Detail] = controller
	plan: FixedStepPlan | None = selected.plan if type(selected) is FixedStepController else None
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
			# The fixed route consumes the shared plan; custom controllers keep
			# their own accepted intervals and sampling tolerance.
			limit: int = int(plan.sample_offsets[info.index + 1]) if plan is not None else request.output_times.size
			while output_index < limit and (plan is not None or request.output_times[output_index] <= info.end_time + sampling_tolerance):
				target: float = float(request.output_times[output_index])
				sample: IntegrationState
				query: float
				interior: bool
				if plan is not None:
					assert isinstance(selected, FixedStepController)
					sample = selected.planned_sample(method, info, result, output_index)
					mode: int = int(plan.sample_modes[output_index])
					query = info.end_time if mode == SAMPLE_END else info.time if mode == SAMPLE_START else target
					interior = mode == SAMPLE_SHADOW
				else:
					sample = selected.sample(method, info, result, target)
					query = info.end_time if abs(target - info.end_time) <= sampling_tolerance else (
						info.time if abs(target - info.time) <= sampling_tolerance else target)
					interior = min(abs(target - info.time), abs(target - info.end_time)) > sampling_tolerance
				if sample.time != target:
					raise ValueError("The controller returned a different sample time.")
				_record_sample(method, saved, sample, energy, sample_time=query)
				interior_sample_count += int(interior)
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
		diagnostics.update(energy.diagnose_history(saved.times, saved.values.T, dict(saved.auxiliary)))
	diagnostics["output_interpolation_count"] = interior_sample_count
	return diagnostics


__all__: list[str] = []
