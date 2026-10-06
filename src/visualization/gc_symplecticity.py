"""Plots and animations for physical guiding-centre symplecticity studies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from studies._gc_symplecticity import _summary_solver_values
from visualization.gc_area import animate_gc_area_comparison

if TYPE_CHECKING:
	from studies._gc_symplecticity import GCSymplecticityResult
	from studies.abba_midpoint_symplecticity import ABBA2MidpointSymplecticityResult
	from studies.abba_symplecticity import ABBASymplecticityResult
	from studies.rk4_symplecticity import RK4SymplecticityResult


def plot_gc_symplecticity_diagnostics(
	result: GCSymplecticityResult,
) -> tuple[Figure, np.ndarray]:
	"""Plot area, local and accumulated symplecticity, and determinant drift."""
	figure, axes = plt.subplots(
		2,
		2,
		figsize=(13, 8),
		constrained_layout=True,
	)
	for step in result.steps:
		label = step.label
		records = result.records[label]
		times = np.asarray([record.time for record in records])
		area_errors = np.asarray(
			[record.relative_area_error for record in records]
		)
		local_defects = np.asarray(
			[record.local_relative_defect for record in records]
		)
		flow_defects = np.asarray(
			[record.relative_defect for record in records]
		)
		determinant_errors = np.asarray(
			[record.determinant_error for record in records]
		)
		axes[0, 0].plot(times, area_errors, label=label)
		axes[0, 1].semilogy(
			times[1:],
			_positive_for_log(local_defects[1:]),
			label=label,
		)
		axes[1, 0].semilogy(
			times[1:],
			_positive_for_log(flow_defects[1:]),
			label=label,
		)
		axes[1, 1].semilogy(
			times[1:],
			_positive_for_log(determinant_errors[1:]),
			label=label,
		)

	axes[0, 0].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
	axes[0, 0].set(
		title="Relative transported-area error",
		xlabel="$t$",
		ylabel=r"$(A(t)-A(0))/|A(0)|$",
	)
	axes[0, 1].set(
		title=f"Local {result.method_name} step symplecticity defect",
		xlabel="$t$",
		ylabel=r"$\|J_n^T\Omega J_n-\Omega\|_F/\|\Omega\|_F$",
	)
	axes[1, 0].set(
		title="Accumulated numerical-flow symplecticity defect",
		xlabel="$t$",
		ylabel=r"$\|DG_n^T\Omega DG_n-\Omega\|_F/\|\Omega\|_F$",
	)
	axes[1, 1].set(
		title="Accumulated numerical-flow determinant error",
		xlabel="$t$",
		ylabel=r"$|\det(DG_n)-1|$",
	)
	for axis in axes.flat:
		axis.grid(alpha=0.25)
		axis.legend()
	return figure, axes


def _plot_step_defects(
	result: GCSymplecticityResult,
	*,
	title: str,
	xlabel: str,
) -> tuple[Figure, Axes]:
	"""Plot maximum measured symplecticity defects against step size."""
	summaries = result.summaries()
	steps = np.asarray([row.step for row in summaries])
	local_errors = np.asarray([row.max_local_defect for row in summaries])
	flow_errors = np.asarray([row.max_flow_defect for row in summaries])
	figure, axes = plt.subplots(figsize=(8, 5), constrained_layout=True)
	axes.loglog(steps, local_errors, "o-", label="Maximum local-step defect")
	axes.loglog(steps, flow_errors, "s-", label="Maximum accumulated defect")
	axes.set(
		xlabel=xlabel,
		ylabel="Relative symplecticity defect",
		title=title,
	)
	axes.grid(which="both", alpha=0.25)
	axes.legend()
	axes.invert_xaxis()
	return figure, axes


def plot_gc_solver_diagnostics(
	result: GCSymplecticityResult,
) -> tuple[Figure, np.ndarray]:
	"""Plot main-step nonlinear-solve work, residuals and multipliers."""
	summaries = result.summaries()
	solver_rows = tuple(_summary_solver_values(row) for row in summaries)
	if all(values is None for values in solver_rows):
		raise ValueError(
			f"{result.method_name} does not provide nonlinear-solver diagnostics."
		)
	if any(values is None for values in solver_rows):
		raise ValueError(
			"Nonlinear-solver summaries must be available for every step or none."
		)

	steps = np.asarray([row.step for row in summaries])
	complete_rows = tuple(values for values in solver_rows if values is not None)
	max_iterations = np.asarray([values[0] for values in complete_rows])
	mean_iterations = np.asarray([values[1] for values in complete_rows])
	max_residuals = np.asarray([values[2] for values in complete_rows])
	max_multipliers = np.asarray([values[3] for values in complete_rows])
	figure, axes = plt.subplots(
		1,
		2,
		figsize=(12, 4.5),
		constrained_layout=True,
	)
	axes[0].plot(steps, max_iterations, "o-", label="Maximum iterations")
	axes[0].plot(steps, mean_iterations, "s-", label="Mean iterations")
	axes[0].set(
		xlabel=r"Integration step $\Delta t$",
		ylabel="Newton iterations",
		title="Projection-solve work on main steps",
	)
	axes[1].loglog(
		steps,
		_positive_for_log(max_residuals),
		"o-",
		label="Maximum final residual",
	)
	axes[1].loglog(
		steps,
		_positive_for_log(max_multipliers),
		"s-",
		label=r"Maximum $\|\mu\|_\infty$",
	)
	axes[1].set(
		xlabel=r"Integration step $\Delta t$",
		ylabel="Infinity norm",
		title="Projection residual and multiplier",
	)
	for axis in axes:
		axis.grid(which="both", alpha=0.25)
		axis.legend()
		axis.invert_xaxis()
	return figure, axes


def animate_gc_symplecticity(
	result: GCSymplecticityResult,
	*,
	frames: int | None = None,
	interval: int = 200,
	repeat: bool = True,
) -> FuncAnimation:
	"""Animate GC contours, area errors and accumulated symplecticity."""
	return animate_gc_area_comparison(
		result.dynamics.effective_potential,
		result.area,
		result.solutions,
		diagnostic_times=result.diagnostic_times,
		relative_symplecticity_errors=result.relative_symplecticity_errors,
		frames=frames,
		interval=interval,
		repeat=repeat,
	)


def _positive_for_log(values: np.ndarray) -> np.ndarray:
	"""Replace exact zeros by a local positive floor for logarithmic display."""
	result = np.asarray(values, dtype=float)
	positive = result[result > 0]
	floor = (
		float(np.min(positive)) / 10
		if positive.size
		else float(np.finfo(float).eps)
	)
	return np.maximum(result, floor)


def plot_abba_midpoint_convergence(
	result: ABBA2MidpointSymplecticityResult,
) -> tuple[Figure, Axes]:
	"""Plot defects and pre-projection copy separation against step size."""
	figure, axis = _plot_step_defects(
		result,
		title="Midpoint ABBA symplecticity-defect convergence",
		xlabel=r"Midpoint ABBA step $\Delta t$",
	)
	rows = result.summaries()
	axis.loglog(
		[row.step for row in rows],
		[row.max_copy_separation_norm for row in rows],
		"^-",
		label="Maximum copy separation before averaging",
	)
	axis.set_ylabel("Relative defect or copy-separation norm")
	axis.legend()
	return figure, axis


def plot_abba_defect_floor(result: ABBASymplecticityResult) -> tuple[Figure, Axes]:
	"""Plot measured ABBA defects across steps as a numerical floor."""
	return _plot_step_defects(
		result,
		title="Projected ABBA symplecticity-defect numerical floor",
		xlabel=r"ABBA step $\Delta t$",
	)


def plot_rk4_convergence(result: RK4SymplecticityResult) -> tuple[Figure, Axes]:
	"""Plot RK4 symplecticity defects against the integration step size."""
	return _plot_step_defects(
		result,
		title="RK4 symplecticity-defect convergence",
		xlabel=r"RK4 step $\Delta t$",
	)


__all__ = [
	"plot_gc_symplecticity_diagnostics",
	"plot_gc_solver_diagnostics",
	"animate_gc_symplecticity",
	"plot_abba_midpoint_convergence",
	"plot_abba_defect_floor",
	"plot_rk4_convergence",
]
