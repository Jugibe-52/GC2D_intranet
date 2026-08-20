"""Accuracy plots for Euler and the two tangent-Taylor ABBA methods."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter
import numpy as np

from studies import (
	TANGENT_TAYLOR_EULER_METHOD_NAMES,
	TangentTaylorEulerAccuracyResult,
)


TANGENT_TAYLOR_EULER_COLORS = {
	"ExplicitEuler": "tab:gray",
	"ImplicitABBA1TangentTaylor": "tab:blue",
	"ABBA4Implicit1TangentTaylor": "tab:orange",
}

_SHORT_LABELS = {
	"ExplicitEuler": "Euler",
	"ImplicitABBA1TangentTaylor": "Implicit-ABBA tangent",
	"ABBA4Implicit1TangentTaylor": "ABBA4 tangent",
}


def plot_tangent_taylor_euler_accuracy(
	result: TangentTaylorEulerAccuracyResult,
) -> tuple[Figure, np.ndarray]:
	"""Plot refinement error and finest-grid error evolution for all methods."""
	if not isinstance(result, TangentTaylorEulerAccuracyResult):
		raise TypeError("`result` must be TangentTaylorEulerAccuracyResult.")
	summaries = {
		(row.step_count, row.method_name): row for row in result.summaries()
	}
	figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
	steps = np.asarray(result.config.integration_steps, dtype=float)
	for method_name in TANGENT_TAYLOR_EULER_METHOD_NAMES:
		color = TANGENT_TAYLOR_EULER_COLORS[method_name]
		short_label = _SHORT_LABELS[method_name]
		# The two tangent-Taylor errors nearly coincide in the intended study.
		# A wider blue trace and a thinner orange trace keep both visible.
		line_width = 3.0 if method_name == "ImplicitABBA1TangentTaylor" else 1.6
		time_errors = np.asarray(
			[
				summaries[(count, method_name)].time_integrated_rms_distance
				for count in result.config.step_counts
			]
		)
		final_errors = np.asarray(
			[
				summaries[(count, method_name)].final_rms_distance
				for count in result.config.step_counts
			]
		)
		axes[0].loglog(
			steps,
			time_errors,
			marker="o",
			color=color,
			linewidth=line_width,
			label=f"{short_label}: time RMS",
		)
		axes[0].loglog(
			steps,
			final_errors,
			marker="s",
			linestyle="--",
			color=color,
			linewidth=line_width,
			label=f"{short_label}: final RMS",
		)
		finest = result.finest_runs[method_name]
		axes[1].semilogy(
			result.reference.times,
			np.where(
				finest.accuracy.rms_distance > 0.0,
				finest.accuracy.rms_distance,
				np.nan,
			),
			color=color,
			linewidth=line_width,
			label=short_label,
		)
	axes[0].set(
		title="Accuracy under nested step refinement",
		xlabel="complete step $h$",
		ylabel="periodic particle RMS error",
	)
	axes[0].grid(which="both", alpha=0.25)
	axes[0].legend(fontsize="x-small")
	finest_step = result.config.integration_steps[-1]
	axes[1].set(
		title=f"RMS error over time at finest step h={finest_step:.4g}",
		xlabel="$t$",
		ylabel="periodic particle RMS error",
	)
	axes[1].grid(which="both", alpha=0.25)
	axes[1].legend(fontsize="small")
	return figure, axes


def plot_tangent_taylor_h_error(
	result: TangentTaylorEulerAccuracyResult,
) -> tuple[Figure, np.ndarray]:
	"""Plot both tangent-Taylor errors against their exact complete step."""
	if not isinstance(result, TangentTaylorEulerAccuracyResult):
		raise TypeError("`result` must be TangentTaylorEulerAccuracyResult.")
	summaries = {
		(row.step_count, row.method_name): row for row in result.summaries()
	}
	method_names = (
		"ImplicitABBA1TangentTaylor",
		"ABBA4Implicit1TangentTaylor",
	)
	# Present fine to coarse so the requested near-doubling sequence reads from
	# left to right on the logarithmic step axis.
	counts = tuple(reversed(result.config.step_counts))
	steps = np.asarray(tuple(reversed(result.config.integration_steps)), dtype=float)
	figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
	metrics = (
		("time_integrated_rms_distance", "Time-integrated RMS error"),
		(
			"final_rms_distance",
			f"Final RMS error at t={result.reference.times[-1]:.4g}",
		),
	)
	for axis, (attribute, title) in zip(axes, metrics):
		guide_errors: np.ndarray | None = None
		for method_name in method_names:
			errors = np.asarray(
				[
					getattr(summaries[(count, method_name)], attribute)
					for count in counts
				],
				dtype=float,
			)
			if guide_errors is None:
				guide_errors = errors
			line_width = 3.2 if method_name == method_names[0] else 1.6
			axis.loglog(
				steps,
				errors,
				marker="o",
				linewidth=line_width,
				color=TANGENT_TAYLOR_EULER_COLORS[method_name],
				label=_SHORT_LABELS[method_name],
			)
		assert guide_errors is not None
		axis.loglog(
			steps,
			guide_errors[0] * steps / steps[0],
			color="black",
			linestyle=":",
			linewidth=1.3,
			label=r"first-order guide $C h$",
		)
		axis.set_xticks(steps, labels=[f"{step:.5f}" for step in steps])
		axis.xaxis.set_minor_formatter(NullFormatter())
		axis.set(
			title=title,
			xlabel="exact complete step $h=2\pi/N$",
			ylabel="periodic particle RMS error",
		)
		axis.grid(which="both", alpha=0.25)
		axis.legend(fontsize="small")
	return figure, axes


__all__ = [
	"TANGENT_TAYLOR_EULER_COLORS",
	"plot_tangent_taylor_euler_accuracy",
	"plot_tangent_taylor_h_error",
]
