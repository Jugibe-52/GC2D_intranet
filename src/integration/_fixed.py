# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Shared step-count and progress utilities for the common integrator."""

from __future__ import annotations

import math
import sys


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
		fraction = min(self.steps / self.total, 1.0)
		if self.t_span is not None:
			start, end = self.t_span
			fraction = min(max((t - start) / (end - start), 0.0), 1.0)
		percent = int(100 * fraction)
		if percent == self._last_percent:
			return
		self._last_percent = percent
		width = 30
		filled = int(width * fraction)
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


__all__: list[str] = []
