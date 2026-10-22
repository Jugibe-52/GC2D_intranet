"""Structural contracts consumed by visualization.implicit_method_accuracy."""

from __future__ import annotations

from typing import Protocol


class ImplicitMethodAccuracySummaryView(Protocol):
	"""Step-summary fields consumed by implicit accuracy plots."""

	method_name: str
	method_label: str
	designed_order: float
	integration_step: float
	time_integrated_rms_distance: float
	mean_iterations_per_solve: float
	maximum_iterations_per_solve: int
	mean_residual_evaluations_per_solve: float
	maximum_residual_to_tolerance: float
	runtime_seconds: float


class ImplicitMethodAccuracyOrderView(Protocol):
	"""Adjacent-refinement fields consumed by the observed-order plot."""

	method_name: str
	method_label: str
	designed_order: float
	fine_step: float
	time_integrated_rms_order: float
	final_rms_order: float


__all__ = ['ImplicitMethodAccuracySummaryView', 'ImplicitMethodAccuracyOrderView']
