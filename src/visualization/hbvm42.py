"""Plots for HBVM(4,2) evaluation and BM4 work--precision comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from simulation import Solution


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


def _positive(values: np.ndarray) -> np.ndarray:
	"""Clamp only exact zeros so log plots retain every finite sample."""
	return np.maximum(np.asarray(values, dtype=float), np.finfo(float).tiny)


def plot_hbvm42_evaluation(
	summaries: Sequence[HBVM42EvaluationSummaryView],
	orders: Sequence[HBVM42OrderSummaryView],
) -> tuple[Figure, np.ndarray]:
	"""Plot accuracy, cost, energy, geometry, order reduction, and solver work."""
	rows = tuple(summaries)
	order_rows = tuple(orders)
	if len(rows) < 2 or len(order_rows) != len(rows) - 1:
		raise ValueError("HBVM evaluation plots require aligned refinement rows.")
	steps = np.asarray([row.step for row in rows])
	figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

	axes[0, 0].loglog(
		steps,
		_positive(np.asarray([row.trajectory_rms_error for row in rows])),
		"o-",
		label="Trajectory RMS",
	)
	axes[0, 0].loglog(
		steps,
		_positive(np.asarray([row.final_error for row in rows])),
		"s--",
		label="Final state",
	)
	axes[0, 0].set(title="Global accuracy", xlabel="Step $h$", ylabel="Error")
	axes[0, 0].legend()

	axes[0, 1].loglog(
		steps,
		_positive(np.asarray([row.median_runtime_seconds for row in rows])),
		"o-",
		label="Median",
	)
	axes[0, 1].loglog(
		steps,
		_positive(np.asarray([row.minimum_runtime_seconds for row in rows])),
		"s--",
		label="Minimum",
	)
	axes[0, 1].set(title="Execution time", xlabel="Step $h$", ylabel="Seconds")
	axes[0, 1].legend()

	axes[0, 2].loglog(
		steps,
		_positive(np.asarray([row.maximum_absolute_energy_error for row in rows])),
		"o-",
		label="Absolute",
	)
	axes[0, 2].loglog(
		steps,
		_positive(np.asarray([row.maximum_relative_energy_error for row in rows])),
		"s--",
		label="Relative",
	)
	axes[0, 2].set(
		title="Maximum Hamiltonian drift",
		xlabel="Step $h$",
		ylabel="Energy error",
	)
	axes[0, 2].legend()

	axes[1, 0].loglog(
		steps,
		_positive(np.asarray([row.local_symplecticity_defect for row in rows])),
		"o-",
		label="One-step map",
	)
	axes[1, 0].loglog(
		steps,
		_positive(np.asarray([row.flow_symplecticity_defect for row in rows])),
		"s--",
		label="Final flow map",
	)
	axes[1, 0].set(
		title="Canonical symplecticity defect",
		xlabel="Step $h$",
		ylabel=r"$\|D\Phi^TJD\Phi-J\|_F/\|J\|_F$",
	)
	axes[1, 0].legend()

	fine_steps = np.asarray([row.fine_step for row in order_rows])
	axes[1, 1].semilogx(
		fine_steps,
		np.asarray([row.trajectory_order_reduction for row in order_rows]),
		"o-",
		label="Trajectory RMS",
	)
	axes[1, 1].semilogx(
		fine_steps,
		np.asarray([row.final_order_reduction for row in order_rows]),
		"s--",
		label="Final state",
	)
	axes[1, 1].axhline(0.0, color="0.4", linestyle=":", linewidth=1)
	axes[1, 1].set(
		title="Order reduction relative to $p=4$",
		xlabel="Fine step $h$",
		ylabel=r"$4-p_{observed}$",
	)
	axes[1, 1].legend()

	axes[1, 2].semilogx(
		steps,
		np.asarray([row.mean_nonlinear_iterations for row in rows]),
		"o-",
		label="Mean Newton iterations",
	)
	axes[1, 2].semilogx(
		steps,
		np.asarray([row.mean_vector_field_evaluations for row in rows]),
		"s--",
		label="Mean field evaluations",
	)
	axes[1, 2].set(
		title="Nonlinear work per step",
		xlabel="Step $h$",
		ylabel="Count",
	)
	axes[1, 2].legend()

	for axis in axes.flat:
		axis.grid(alpha=0.25)
	return figure, axes


def plot_hbvm42_energy_errors(
	solutions: Mapping[float, Solution],
) -> tuple[Figure, Axes]:
	"""Plot signed physical-Hamiltonian drift for every evaluated HBVM step."""
	if not solutions:
		raise ValueError("At least one HBVM solution is required.")
	figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
	for step, solution in solutions.items():
		energy_drift = solution.diagnostics.get("energy_drift")
		if energy_drift is None:
			raise ValueError("Every solution must contain HBVM energy diagnostics.")
		values = np.asarray(energy_drift, dtype=float)
		if values.shape != (1, solution.t.size):
			raise ValueError("The energy plot requires one planar trajectory.")
		axis.plot(solution.t, values[0], label=rf"$h={step:g}$")
	axis.axhline(0.0, color="0.4", linestyle=":", linewidth=1)
	axis.set(
		title="HBVM(4,2) Hamiltonian drift",
		xlabel="$t$",
		ylabel=r"$H(y_n)-H(y_0)$",
	)
	axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
	axis.grid(alpha=0.25)
	axis.legend()
	return figure, axis


def plot_hbvm42_bm4_comparison(
	summaries: Sequence[HBVM42BM4SummaryView],
) -> tuple[Figure, np.ndarray]:
	"""Plot endpoint accuracy, runtime, and work--precision for both methods."""
	rows = tuple(summaries)
	methods = tuple(dict.fromkeys(row.method for row in rows))
	if len(methods) != 2:
		raise ValueError("The comparison plot requires exactly two methods.")
	figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
	markers = ("o", "s")
	for method, marker in zip(methods, markers, strict=True):
		method_rows = tuple(row for row in rows if row.method == method)
		steps = np.asarray([row.step for row in method_rows])
		errors = _positive(np.asarray([row.final_error for row in method_rows]))
		median_times = _positive(
			np.asarray([row.median_runtime_seconds for row in method_rows])
		)
		minimum_times = _positive(
			np.asarray([row.minimum_runtime_seconds for row in method_rows])
		)
		axes[0].loglog(steps, errors, marker=marker, label=method)
		axes[1].loglog(steps, median_times, marker=marker, label=f"{method} median")
		axes[1].loglog(
			steps,
			minimum_times,
			marker=marker,
			linestyle="--",
			alpha=0.75,
			label=f"{method} minimum",
		)
		axes[2].loglog(
			median_times,
			errors,
			marker=marker,
			label=method,
		)
	axes[0].set(title="Final-state accuracy", xlabel="Step $h$", ylabel="Error")
	axes[1].set(title="Execution time", xlabel="Step $h$", ylabel="Seconds")
	axes[2].set(
		title="Work--precision diagram",
		xlabel="Median runtime [s]",
		ylabel="Final-state error",
	)
	for axis in axes:
		axis.grid(alpha=0.25)
		axis.legend()
	return figure, axes


__all__ = [
	"HBVM42BM4SummaryView",
	"HBVM42EvaluationSummaryView",
	"HBVM42OrderSummaryView",
	"plot_hbvm42_bm4_comparison",
	"plot_hbvm42_energy_errors",
	"plot_hbvm42_evaluation",
]
