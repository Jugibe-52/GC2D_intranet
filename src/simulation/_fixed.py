# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Shared fixed-grid stepping and output sampling for numerical methods."""

from __future__ import annotations

from collections.abc import Callable
import math
import sys

import numpy as np

from .request import SimulationRequest


# ``observe`` is false for shadow advances used only to sample output times.
Advance = Callable[[float, np.ndarray, float, int, bool], np.ndarray]


class _Progress:
	"""Small stderr progress indicator counting complete integration steps."""

	def __init__(self, label: str, total: int) -> None:
		self.label = label
		self.total = max(total, 1)
		self.every = max(self.total // 100, 1)
		self.steps = 0

	def update(self, t: float) -> None:
		"""Advance and occasionally redraw the same terminal line."""
		self.steps += 1
		if self.steps % self.every and self.steps < self.total:
			return
		fraction = min(self.steps / self.total, 1.0)
		width = 30
		filled = int(width * fraction)
		bar = "=" * filled
		if filled < width:
			bar += ">" + " " * (width - filled - 1)
		print(
			f"\r{self.label} [{bar}] {fraction:6.1%} "
			f"({self.steps}/{self.total}, t={t:.6g})",
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


def integrate_fixed_grid(
	initial_state: np.ndarray,
	request: SimulationRequest,
	advance: Advance,
	*,
	progress: bool,
	label: str,
) -> tuple[np.ndarray, int]:
	"""Advance on an output-independent grid and evaluate shadow samples.

	Every off-grid output starts from the preceding main-grid node.  It therefore
	does not change the integration trajectory, later samples, or observations.
	"""
	value = np.asarray(initial_state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The numerical initial state must be a finite vector.")
	t0, tf = request.t_span
	times = request.output_times
	step_count = _step_count(tf - t0, request.max_step)
	internal_step = (tf - t0) / step_count
	result = np.empty(value.shape + (times.size,), dtype=value.dtype)
	result[:, 0] = value
	output_index = 1
	time_tolerance = 16 * np.finfo(float).eps * max(1.0, abs(t0), abs(tf))
	progress_bar = _Progress(label, step_count) if progress else None

	try:
		for step_index in range(step_count):
			step_start = t0 + step_index * internal_step
			step_end = t0 + (step_index + 1) * internal_step
			initial_value = np.asarray(value).copy()
			value = np.asarray(
				advance(step_start, value, internal_step, step_index, True)
			)
			if value.shape != initial_value.shape:
				raise ValueError("A numerical step changed the internal state shape.")

			while (
				output_index < times.size
				and float(times[output_index]) <= step_end + time_tolerance
			):
				target = float(times[output_index])
				if abs(target - step_end) <= time_tolerance:
					sample = value
				elif abs(target - step_start) <= time_tolerance:
					sample = initial_value
				else:
					sample = np.asarray(
						advance(
							step_start,
							initial_value.copy(),
							target - step_start,
							step_index,
							False,
						)
					)
					if sample.shape != initial_value.shape:
						raise ValueError(
							"A shadow step changed the internal state shape."
						)
				result[:, output_index] = sample
				output_index += 1

			if progress_bar is not None:
				progress_bar.update(step_end)
	finally:
		if progress_bar is not None:
			progress_bar.close()

	if output_index != times.size:
		raise RuntimeError("The integration grid did not cover every output time.")
	return result, step_count


__all__: list[str] = []
