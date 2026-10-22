"""Structural contracts consumed by visualization.hbvm42."""

from __future__ import annotations

from typing import Protocol


class HBVM42EvaluationSummaryView(Protocol):
	"""Structural plotting view of one individual-evaluation row."""

	step: float
	trajectory_rms_error: float
	final_error: float
	median_runtime_seconds: float
	minimum_runtime_seconds: float
	maximum_absolute_energy_error: float
	maximum_relative_energy_error: float
	local_symplecticity_defect: float
	flow_symplecticity_defect: float
	maximum_nonlinear_iterations: int
	mean_nonlinear_iterations: float
	mean_vector_field_evaluations: float


class HBVM42OrderSummaryView(Protocol):
	"""Structural plotting view of one adjacent-step order row."""

	fine_step: float
	trajectory_order_reduction: float
	final_order_reduction: float


class HBVM42BM4SummaryView(Protocol):
	"""Structural plotting view of one comparison row."""

	method: str
	step: float
	final_error: float
	median_runtime_seconds: float
	minimum_runtime_seconds: float


__all__ = ['HBVM42EvaluationSummaryView', 'HBVM42OrderSummaryView', 'HBVM42BM4SummaryView']
