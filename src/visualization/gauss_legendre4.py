"""Matplotlib views for Gauss4 evaluation and Gauss4/BM4 comparison studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


GAUSS_BM4_COLORS: Mapping[str, str] = {
	"GaussLegendre4": "tab:blue",
	"BM4Implicit1": "tab:orange",
}


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


def _positive(values: np.ndarray) -> np.ndarray:
	"""Replace exact zeros by a data-dependent floor for logarithmic axes."""
	result = np.asarray(values, dtype=float)
	positive = result[result > 0.0]
	floor = (
		float(np.min(positive)) / 10.0
		if positive.size
		else float(np.finfo(float).eps)
	)
	return np.maximum(result, floor)


def plot_gauss_legendre4_evaluation(
	summaries: Sequence[GaussEvaluationSummaryView],
	*,
	designed_order: float = 4.0,
	reference_floor: float = 0.0,
) -> tuple[Figure, np.ndarray]:
	"""Plot accuracy, runtime, generalized energy, and geometric defects."""
	rows = tuple(sorted(summaries, key=lambda row: row.integration_step, reverse=True))
	if len(rows) < 2:
		raise ValueError("At least two Gauss4 refinement summaries are required.")
	steps = np.asarray([row.integration_step for row in rows], dtype=float)
	time_errors = np.asarray(
		[row.time_integrated_rms_distance for row in rows], dtype=float
	)
	final_errors = np.asarray([row.final_rms_distance for row in rows], dtype=float)
	figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
	axes[0, 0].loglog(steps, time_errors, "o-", label="Time-integrated RMS")
	axes[0, 0].loglog(steps, final_errors, "s--", label="Final RMS")
	guide = time_errors[0] * (steps / steps[0]) ** designed_order
	axes[0, 0].loglog(
		steps,
		guide,
		color="black",
		linestyle=":",
		label=rf"$O(h^{{{designed_order:g}}})$ guide",
	)
	if reference_floor > 0.0:
		axes[0, 0].axhline(
			reference_floor,
			color="0.4",
			linestyle="--",
			label="DOP853/Radau audit floor",
		)
	median = np.asarray([row.runtime_seconds for row in rows])
	first = np.asarray([row.runtime_first_quartile_seconds for row in rows])
	third = np.asarray([row.runtime_third_quartile_seconds for row in rows])
	axes[0, 1].errorbar(
		steps,
		median,
		yerr=np.vstack((median - first, third - median)),
		marker="o",
		capsize=4,
	)
	axes[0, 1].set_xscale("log")
	axes[0, 1].set_yscale("log")
	energy = _positive(
		np.asarray([row.maximum_relative_generalized_energy_error for row in rows])
	)
	axes[1, 0].loglog(steps, energy, "o-", color="tab:green")
	for field, marker, label in (
		("maximum_local_symplecticity_defect", "o", "Ideal local tangent"),
		(
			"maximum_accumulated_symplecticity_defect",
			"s",
			"Ideal accumulated tangent",
		),
		(
			"maximum_finite_difference_symplecticity_defect",
			"^",
			"Finite-stopping map audit",
		),
	):
		values = _positive(np.asarray([getattr(row, field) for row in rows]))
		axes[1, 1].loglog(steps, values, marker=marker, label=label)
	for axis in axes.flat:
		axis.invert_xaxis()
		axis.grid(which="both", alpha=0.25)
	axes[0, 0].set(
		title="Gauss4 trajectory accuracy",
		xlabel="Complete step $h$",
		ylabel="Periodic trajectory distance",
	)
	axes[0, 0].legend(fontsize=8)
	axes[0, 1].set(
		title="Observer-free integration runtime",
		xlabel="Complete step $h$",
		ylabel="Median wall time [s]",
	)
	axes[1, 0].set(
		title="Generalized energy $K=H+k$",
		xlabel="Complete step $h$",
		ylabel="Maximum relative drift",
	)
	axes[1, 1].set(
		title="Physical symplecticity",
		xlabel="Complete step $h$",
		ylabel="Maximum relative defect",
	)
	axes[1, 1].legend(fontsize=8)
	return figure, axes


def plot_gauss_legendre4_observed_order(
	orders: Sequence[GaussOrderView],
	*,
	designed_order: float = 4.0,
	reduction_threshold: float = 0.5,
) -> tuple[Figure, np.ndarray]:
	"""Plot observed accuracy orders and their deficit from the design."""
	rows = tuple(sorted(orders, key=lambda row: row.fine_step, reverse=True))
	if not rows:
		raise ValueError("At least one observed-order row is required.")
	steps = np.asarray([row.fine_step for row in rows])
	figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
	for order_field, deficit_field, flag_field, linestyle, label in (
		(
			"time_integrated_rms_order",
			"time_integrated_order_deficit",
			"time_integrated_reduction_detected",
			"-",
			"Time-integrated RMS",
		),
		(
			"final_rms_order",
			"final_order_deficit",
			"final_reduction_detected",
			"--",
			"Final RMS",
		),
	):
		order_values = np.asarray([getattr(row, order_field) for row in rows])
		deficits = np.asarray([getattr(row, deficit_field) for row in rows])
		mask = np.isfinite(order_values)
		axes[0].plot(steps[mask], order_values[mask], "o", linestyle=linestyle, label=label)
		axes[1].plot(steps[mask], deficits[mask], "o", linestyle=linestyle, label=label)
		flagged = mask & np.asarray([getattr(row, flag_field) for row in rows], dtype=bool)
		axes[1].scatter(steps[flagged], deficits[flagged], marker="x", s=70, color="red")
	axes[0].axhline(designed_order, color="black", linestyle=":", label="Designed order")
	axes[1].axhline(0.0, color="black", linestyle=":")
	axes[1].axhline(
		reduction_threshold,
		color="red",
		linestyle=":",
		label="Reduction threshold",
	)
	axes[0].set(title="Resolved observed order", ylabel="Observed order $p$")
	axes[1].set(
		title="Observed order deficit",
		xlabel="Fine step $h$",
		ylabel=rf"${designed_order:g}-p$",
	)
	axes[1].set_xscale("log")
	axes[1].set_xlim(float(steps[0] * 1.1), float(steps[-1] / 1.1))
	for axis in axes:
		axis.grid(alpha=0.25)
		axis.legend(fontsize=8)
	return figure, axes


def plot_gauss_legendre4_symplecticity(
	series_by_step: Mapping[float, GaussSymplecticityView],
) -> tuple[Figure, np.ndarray]:
	"""Plot ideal-root evolution and sparse finite-stopping-rule audits."""
	if not series_by_step:
		raise ValueError("At least one Gauss4 symplecticity series is required.")
	figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
	for step in sorted(series_by_step, reverse=True):
		series = series_by_step[step]
		label = f"h={step:g}"
		axes[0, 0].semilogy(
			series.times,
			_positive(series.local_relative_defects),
			label=label,
		)
		axes[0, 1].semilogy(
			series.times,
			_positive(series.accumulated_relative_defects),
			label=label,
		)
		axes[1, 0].semilogy(
			series.audit_times,
			_positive(series.finite_difference_relative_defects),
			"o-",
			label=label,
		)
		axes[1, 1].semilogy(
			series.audit_times,
			_positive(series.analytic_finite_difference_relative_differences),
			"o-",
			label=label,
		)
	titles = (
		"Ideal-root local symplecticity defect",
		"Ideal-root accumulated defect",
		"Finite-stopping map symplecticity audit",
		"Analytic versus finite-difference tangent",
	)
	ylabels = (
		"Relative defect",
		"Relative defect",
		"Relative defect",
		"Relative Jacobian difference",
	)
	for axis, title, ylabel in zip(axes.flat, titles, ylabels, strict=True):
		axis.set(title=title, xlabel="$t$", ylabel=ylabel)
		axis.grid(alpha=0.25)
		axis.legend(fontsize=8)
	return figure, axes


def plot_gauss_legendre4_energy(
	times_by_step: Mapping[float, np.ndarray],
	energies_by_step: Mapping[float, np.ndarray],
) -> tuple[Figure, np.ndarray]:
	"""Plot generalized-energy drift over every complete node of each grid."""
	if not times_by_step:
		raise ValueError("At least one generalized-energy time grid is required.")
	if not energies_by_step:
		raise ValueError("At least one generalized-energy history is required.")
	if set(times_by_step) != set(energies_by_step):
		raise ValueError("Energy histories and time grids must use the same steps.")
	figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
	for step in sorted(energies_by_step, reverse=True):
		time_values = np.asarray(times_by_step[step], dtype=float)
		if time_values.ndim != 1 or time_values.size < 2:
			raise ValueError("Energy times must be vectors with at least two samples.")
		energy = np.asarray(energies_by_step[step], dtype=float)
		if energy.ndim == 1:
			energy = energy[np.newaxis, :]
		if energy.ndim != 2 or energy.shape[1] != time_values.size:
			raise ValueError("Generalized-energy histories must align with saved times.")
		error = np.max(np.abs(energy - energy[:, :1]), axis=0)
		scale = np.maximum(np.max(np.abs(energy[:, :1])), np.finfo(float).eps)
		label = f"h={step:g}"
		axes[0].semilogy(time_values, _positive(error), label=label)
		axes[1].semilogy(time_values, _positive(error / scale), label=label)
	axes[0].set(title="Generalized-energy absolute drift", ylabel=r"$|K(t)-K(0)|$")
	axes[1].set(
		title="Generalized-energy relative drift",
		xlabel="$t$",
		ylabel=r"$|K(t)-K(0)|/\max(|K(0)|,\epsilon)$",
	)
	for axis in axes:
		axis.grid(alpha=0.25)
		axis.legend(fontsize=8)
	return figure, axes


def plot_gauss_bm4_accuracy_runtime(
	summaries: Sequence[GaussBM4SummaryView],
	orders: Sequence[GaussBM4OrderView],
	*,
	designed_order: float = 4.0,
	reference_floor: float = 0.0,
) -> tuple[Figure, np.ndarray]:
	"""Plot equal-step accuracy, timing, efficiency, and observed order."""
	rows = tuple(summaries)
	if not rows:
		raise ValueError("At least one Gauss4/BM4 summary is required.")
	methods = tuple(dict.fromkeys(row.method_name for row in rows))
	steps = tuple(sorted({row.integration_step for row in rows}, reverse=True))
	if len(methods) != 2 or len(steps) < 2:
		raise ValueError("The comparison requires two methods and two steps.")
	grouped = {
		method: {row.integration_step: row for row in rows if row.method_name == method}
		for method in methods
	}
	figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
	step_values = np.asarray(steps)
	for method in methods:
		method_rows = grouped[method]
		color = GAUSS_BM4_COLORS.get(method)
		label = method_rows[steps[0]].method_label
		errors = np.asarray(
			[method_rows[step].time_integrated_rms_distance for step in steps]
		)
		runtime = np.asarray([method_rows[step].runtime_seconds for step in steps])
		first = np.asarray(
			[method_rows[step].runtime_first_quartile_seconds for step in steps]
		)
		third = np.asarray(
			[method_rows[step].runtime_third_quartile_seconds for step in steps]
		)
		axes[0, 0].loglog(step_values, errors, "o-", color=color, label=label)
		axes[0, 1].errorbar(
			step_values,
			runtime,
			yerr=np.vstack((runtime - first, third - runtime)),
			marker="o",
			color=color,
			capsize=4,
			label=label,
		)
		axes[1, 0].loglog(runtime, errors, "o-", color=color, label=label)
		method_orders = sorted(
			(row for row in orders if row.method_name == method),
			key=lambda row: row.fine_step,
			reverse=True,
		)
		order_steps = np.asarray([row.fine_step for row in method_orders])
		order_values = np.asarray(
			[row.time_integrated_rms_order for row in method_orders]
		)
		mask = np.isfinite(order_values)
		axes[1, 1].plot(
			order_steps[mask],
			order_values[mask],
			"o-",
			color=color,
			label=label,
		)
	guide_anchor = max(
		grouped[method][steps[0]].time_integrated_rms_distance for method in methods
	)
	axes[0, 0].loglog(
		step_values,
		guide_anchor * (step_values / step_values[0]) ** designed_order,
		color="black",
		linestyle=":",
		label=rf"$O(h^{{{designed_order:g}}})$ guide",
	)
	if reference_floor > 0.0:
		axes[0, 0].axhline(reference_floor, color="0.4", linestyle="--", label="Audit floor")
	axes[1, 1].axhline(designed_order, color="black", linestyle=":", label="Designed order")
	axes[0, 0].set(title="Trajectory accuracy", xlabel="Step $h$", ylabel="Time-integrated RMS distance")
	axes[0, 1].set(title="Observer-free runtime", xlabel="Step $h$", ylabel="Median wall time [s]")
	axes[1, 0].set(title="Accuracy--runtime tradeoff", xlabel="Median wall time [s]", ylabel="Time-integrated RMS distance")
	axes[1, 1].set(title="Resolved observed order", xlabel="Fine step $h$", ylabel="Observed order $p$")
	for axis in (axes[0, 0], axes[0, 1], axes[1, 1]):
		axis.set_xscale("log")
		if axis is axes[1, 1]:
			order_steps = np.asarray([row.fine_step for row in orders], dtype=float)
			axis.set_xlim(
				float(np.max(order_steps) * 1.1),
				float(np.min(order_steps) / 1.1),
			)
		else:
			axis.invert_xaxis()
	axes[0, 1].set_yscale("log")
	for axis in axes.flat:
		axis.grid(which="both", alpha=0.25)
		axis.legend(fontsize=8)
	return figure, axes


__all__ = [
	"GAUSS_BM4_COLORS",
	"plot_gauss_bm4_accuracy_runtime",
	"plot_gauss_legendre4_evaluation",
	"plot_gauss_legendre4_energy",
	"plot_gauss_legendre4_observed_order",
	"plot_gauss_legendre4_symplecticity",
]
