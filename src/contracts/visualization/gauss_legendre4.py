"""Structural contracts consumed by visualization.gauss_legendre4."""

from __future__ import annotations

from typing import Protocol
import numpy as np


class GaussEvaluationSummaryView(Protocol):
	"""Scalar fields consumed by the individual-evaluation plots."""

	integration_step: float
	time_integrated_rms_distance: float
	final_rms_distance: float
	runtime_seconds: float
	runtime_first_quartile_seconds: float
	runtime_third_quartile_seconds: float
	maximum_relative_generalized_energy_error: float
	maximum_local_symplecticity_defect: float
	maximum_accumulated_symplecticity_defect: float
	maximum_finite_difference_symplecticity_defect: float


class GaussOrderView(Protocol):
	"""Observed-order fields consumed by the order-deficit plot."""

	fine_step: float
	time_integrated_rms_order: float
	final_rms_order: float
	time_integrated_order_deficit: float
	final_order_deficit: float
	time_integrated_reduction_detected: bool
	final_reduction_detected: bool


class GaussSymplecticityView(Protocol):
	"""Time-series fields consumed by the geometry audit plot."""

	times: np.ndarray
	local_relative_defects: np.ndarray
	accumulated_relative_defects: np.ndarray
	accumulated_determinant_errors: np.ndarray
	audit_times: np.ndarray
	finite_difference_relative_defects: np.ndarray
	analytic_finite_difference_relative_differences: np.ndarray


class GaussBM4SummaryView(Protocol):
	"""Accuracy and timing fields consumed by the comparison dashboard."""

	method_name: str
	method_label: str
	integration_step: float
	time_integrated_rms_distance: float
	final_rms_distance: float
	runtime_seconds: float
	runtime_first_quartile_seconds: float
	runtime_third_quartile_seconds: float


class GaussBM4OrderView(Protocol):
	"""Observed-order fields consumed by the method comparison."""

	method_name: str
	fine_step: float
	time_integrated_rms_order: float
	final_rms_order: float


__all__ = ['GaussEvaluationSummaryView', 'GaussOrderView', 'GaussSymplecticityView', 'GaussBM4SummaryView', 'GaussBM4OrderView']
