"""Plots for stage-projected BM4 physical symplecticity studies."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from diagnostics.projection import ProjectedAreaRecord


def _positive_for_log(values: np.ndarray) -> np.ndarray:
	"""Replace exact zeros by a local display floor."""
	result = np.asarray(values, dtype=float)
	positive = result[result > 0.0]
	floor = float(np.min(positive)) / 10.0 if positive.size else np.finfo(float).eps
	return np.maximum(result, floor)


def plot_projected_bm4_symplecticity_diagnostics(
	records_by_label: Mapping[str, tuple[ProjectedAreaRecord, ...]],
) -> tuple[Figure, np.ndarray]:
	"""Plot physical area and symplecticity diagnostics over integration time."""
	if not records_by_label:
		raise ValueError("At least one labeled diagnostic series is required.")
	figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
	for label, records in records_by_label.items():
		if not records:
			raise ValueError("Every diagnostic series must contain at least one record.")
		times = np.asarray([record.time for record in records])
		area_errors = np.asarray([record.relative_area_error for record in records])
		local_defects = np.asarray(
			[record.local_relative_defect for record in records]
		)
		flow_defects = np.asarray([record.relative_defect for record in records])
		local_determinants = np.asarray(
			[record.local_determinant_error for record in records]
		)
		flow_determinants = np.asarray(
			[record.determinant_error for record in records]
		)
		axes[0, 0].plot(times, area_errors, label=label)
		axes[0, 1].semilogy(times, _positive_for_log(local_defects), label=label)
		axes[1, 0].semilogy(times, _positive_for_log(flow_defects), label=label)
		axes[1, 1].semilogy(
			times,
			_positive_for_log(local_determinants),
			linestyle="--",
			label=f"{label}, local",
		)
		axes[1, 1].semilogy(
			times,
			_positive_for_log(flow_determinants),
			label=f"{label}, accumulated",
		)

	axes[0, 0].axhline(0.0, color="0.5", linestyle="--", linewidth=1.0)
	axes[0, 0].set(
		title="Relative transported-area error",
		xlabel="$t$",
		ylabel=r"$(A(t)-A(0))/|A(0)|$",
	)
	axes[0, 1].set(
		title="Local complete-step symplecticity defect",
		xlabel="$t$",
		ylabel="relative defect",
	)
	axes[1, 0].set(
		title="Accumulated physical-flow symplecticity defect",
		xlabel="$t$",
		ylabel="relative defect",
	)
	axes[1, 1].set(
		title="Physical-map determinant errors",
		xlabel="$t$",
		ylabel=r"$|\det(J)-1|$",
	)
	for axis in axes.flat:
		axis.grid(alpha=0.25)
		axis.legend(fontsize="small")
	return figure, axes


def plot_projected_bm4_symplecticity_convergence(
	*,
	steps: np.ndarray,
	local_defects: np.ndarray,
	flow_defects: np.ndarray,
	area_errors: np.ndarray,
) -> tuple[Figure, Axes]:
	"""Plot maximum physical errors against the complete BM4 step size."""
	values = tuple(
		np.asarray(series, dtype=float)
		for series in (steps, local_defects, flow_defects, area_errors)
	)
	step_values, local_values, flow_values, area_values = values
	if step_values.ndim != 1 or step_values.size < 2:
		raise ValueError("At least two one-dimensional step values are required.")
	if any(series.shape != step_values.shape for series in values[1:]):
		raise ValueError("Every convergence series must match the step-value shape.")
	if np.any(step_values <= 0.0) or any(
		np.any(series < 0.0) for series in values[1:]
	):
		raise ValueError("Steps must be positive and errors must be non-negative.")

	figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
	axis.loglog(
		step_values,
		_positive_for_log(local_values),
		"o-",
		label="Maximum local-step defect",
	)
	axis.loglog(
		step_values,
		_positive_for_log(flow_values),
		"s-",
		label="Maximum accumulated defect",
	)
	axis.loglog(
		step_values,
		_positive_for_log(area_values),
		"^-",
		label="Maximum relative area error",
	)
	axis.set(
		title="Stage-projected BM4 physical-defect convergence",
		xlabel=r"Complete BM4 step $\Delta t$",
		ylabel="Maximum relative error or defect",
	)
	axis.grid(which="both", alpha=0.25)
	axis.legend()
	axis.invert_xaxis()
	return figure, axis


__all__ = [
	"plot_projected_bm4_symplecticity_convergence",
	"plot_projected_bm4_symplecticity_diagnostics",
]
