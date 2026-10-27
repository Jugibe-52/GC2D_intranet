"""Plan fixed integration and shadow sampling before selecting a worker."""

import math
import numpy as np

from contracts.execution.plan import FixedStepPlan, SAMPLE_START, SAMPLE_END, SAMPLE_SHADOW
from request import SimulationRequest


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


def sample_modes(times: np.ndarray, starts: np.ndarray, ends: np.ndarray, tolerance: float) -> np.ndarray:
	"""Prefer the right endpoint when round-off makes both endpoints eligible."""
	return np.asarray(np.where(np.abs(times - ends) <= tolerance, SAMPLE_END,
		np.where(np.abs(times - starts) <= tolerance, SAMPLE_START, SAMPLE_SHADOW)), dtype=np.int32)


def build_fixed_step_plan(request: SimulationRequest, step: float) -> FixedStepPlan:
	"""Combine a validated output request and nominal step without field calls."""
	starts: np.ndarray
	ends: np.ndarray
	durations: np.ndarray
	starts, ends, durations = _fixed_schedule(request.t_span, step)
	times: np.ndarray = request.output_times
	tolerance: float = _time_tolerance(request.t_span)
	# Assign boundary-near samples to the first eligible interval, exactly as
	# the streaming controller does. The initial output is recorded separately.
	owners: np.ndarray = np.minimum(np.searchsorted(ends + tolerance, times), starts.size - 1)
	modes: np.ndarray = sample_modes(times, starts[owners], ends[owners], tolerance)
	modes[0] = SAMPLE_START
	offsets: np.ndarray = np.concatenate(([1], 1 + np.searchsorted(times[1:], ends + tolerance, side="right"))).astype(np.int32)
	return FixedStepPlan(starts, ends, durations, times, offsets, modes, times - starts[owners])


__all__: list[str] = []
