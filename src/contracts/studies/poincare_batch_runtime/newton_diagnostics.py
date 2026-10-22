"""Structural contracts consumed by studies.poincare_batch_runtime.newton_diagnostics."""

from __future__ import annotations

from typing import TypedDict


class DiagnosticTableMetadata(TypedDict):
	"""Scientific dimensions and time conversion used by the CSV exporters."""

	complete_steps: int
	process_count: int
	steps_per_cycle: int
	time_scale_s: float


__all__ = ['DiagnosticTableMetadata']
