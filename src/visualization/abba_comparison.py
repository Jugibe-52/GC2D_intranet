"""Plots and animations for completed ABBA method comparisons."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from visualization.gc_area import animate_gc_area_solution

if TYPE_CHECKING:
	from studies.abba_comparison import ABBAComparisonResult


def plot_abba_runtime_comparison(result: ABBAComparisonResult) -> tuple[Figure, Axes]:
	"""Plot simulation runtime after subtracting symplecticity callbacks."""
	rows = result.runtime_summaries()
	figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
	bars = axis.bar(
		[row.method_name for row in rows],
		[row.seconds for row in rows],
		color=("C0", "C1", "C2"),
	)
	axis.bar_label(bars, fmt="%.3f s", padding=3)
	axis.set(
		ylabel="Wall-clock time [s]",
		title="One-pass ABBA runtime excluding symplecticity diagnostics",
	)
	axis.grid(axis="y", alpha=0.25)
	return figure, axis


def plot_abba_trajectory_differences(result: ABBAComparisonResult) -> tuple[Figure, Axes]:
	"""Plot pairwise RMS and maximum periodic particle displacement over time."""
	figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
	floor = np.finfo(float).eps * result.potential.grid.period
	for index, series in enumerate(result.trajectory_difference_series()):
		color = f"C{index}"
		axis.semilogy(
			series.times,
			np.maximum(series.rms_particle_distance, floor),
			linestyle="--",
			color=color,
			label=f"{series.label}: RMS",
		)
		axis.semilogy(
			series.times,
			np.maximum(series.max_particle_distance, floor),
			color=color,
			label=f"{series.label}: maximum",
		)
	axis.set(
		xlabel="$t$",
		ylabel="Periodic particle displacement",
		title="Pairwise difference between ABBA physical trajectories",
	)
	axis.grid(which="both", alpha=0.25)
	axis.legend(fontsize="small")
	return figure, axis


def animate_abba_comparison(
	result: ABBAComparisonResult,
	method_name: str,
	*,
	frames: int | None = None,
	interval: int = 200,
	repeat: bool = True,
) -> FuncAnimation:
	"""Animate one method's contour, area error, and symplecticity defect."""
	if method_name not in result.studies:
		raise KeyError(f"Unknown ABBA method: {method_name}")
	study = result.studies[method_name]
	label = result.config.step_label
	records = study.records[label]
	return animate_gc_area_solution(
		result.potential,
		result.area,
		study.solutions[label],
		frames=frames,
		interval=interval,
		repeat=repeat,
		diagnostic_times=np.asarray([record.time for record in records]),
		relative_symplecticity_errors=np.asarray(
			[record.relative_defect for record in records]
		),
	)


__all__ = [
	"animate_abba_comparison",
	"plot_abba_runtime_comparison",
	"plot_abba_trajectory_differences",
]
