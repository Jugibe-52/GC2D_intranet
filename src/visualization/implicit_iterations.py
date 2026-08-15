"""Matplotlib views of implicit-ABBA nonlinear-solver work."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from diagnostics import ImplicitIterationRecord


def _record_arrays(
	records: Sequence[ImplicitIterationRecord],
) -> dict[str, np.ndarray]:
	"""Validate ordered records and return consistently shaped plot arrays."""
	values = tuple(records)
	if not values:
		raise ValueError("At least one implicit ABBA iteration record is required.")
	if any(
		not isinstance(record, ImplicitIterationRecord) for record in values
	):
		raise TypeError("All values must be ImplicitIterationRecord instances.")
	for previous, current in zip(values, values[1:]):
		if current.step_index <= previous.step_index:
			raise ValueError("Iteration records must be ordered by increasing step.")
	return {
		"steps": np.asarray([record.step_index for record in values], dtype=int),
		"times": np.asarray([record.end_time for record in values]),
		"iterations": np.asarray(
			[record.nonlinear_iterations for record in values], dtype=int
		),
		"residuals": np.asarray(
			[record.nonlinear_residual_norm for record in values]
		),
		"tolerances": np.asarray(
			[record.nonlinear_tolerance for record in values]
		),
		"ratios": np.asarray(
			[record.residual_to_tolerance_ratio for record in values]
		),
		"multipliers": np.asarray(
			[record.projection_multiplier_norm for record in values]
		),
	}


def _positive_for_log(values: np.ndarray) -> np.ndarray:
	"""Replace exact zeros by the smallest positive float for log plotting."""
	return np.maximum(np.asarray(values, dtype=float), np.finfo(float).tiny)


def plot_implicit_iteration_diagnostics(
	records: Sequence[ImplicitIterationRecord],
) -> tuple[Figure, np.ndarray]:
	"""Plot per-step iterations, frequencies, convergence, and multiplier size."""
	data = _record_arrays(records)
	figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

	axes[0, 0].step(
		data["times"],
		data["iterations"],
		where="mid",
		color="tab:blue",
	)
	axes[0, 0].scatter(data["times"], data["iterations"], s=12)
	axes[0, 0].set(
		title="Nonlinear iterations at each accepted step",
		xlabel="$t_{n+1}$",
		ylabel="iterations",
	)
	axes[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))

	iteration_values, frequencies = np.unique(
		data["iterations"], return_counts=True
	)
	axes[0, 1].bar(iteration_values, frequencies, color="tab:blue")
	axes[0, 1].set(
		title="Iteration-count frequency",
		xlabel="iterations",
		ylabel="accepted steps",
	)
	axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
	axes[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))

	axes[1, 0].semilogy(
		data["times"],
		_positive_for_log(data["residuals"]),
		label="final residual",
	)
	axes[1, 0].semilogy(
		data["times"],
		data["tolerances"],
		linestyle="--",
		label="effective tolerance",
	)
	axes[1, 0].set(
		title="Nonlinear convergence at step acceptance",
		xlabel="$t_{n+1}$",
		ylabel="infinity norm",
	)
	axes[1, 0].legend()

	axes[1, 1].semilogy(
		data["times"],
		_positive_for_log(data["multipliers"]),
		color="tab:orange",
	)
	axes[1, 1].set(
		title="Projection multiplier magnitude",
		xlabel="$t_{n+1}$",
		ylabel=r"$\|\mu_n\|_\infty$",
	)

	method = records[0].method_name
	formulation = records[0].formulation_name
	figure.suptitle(f"Implicit nonlinear solve — {method} / {formulation}")
	return figure, axes


def plot_implicit_iteration_comparison(
	records_by_label: Mapping[str, Sequence[ImplicitIterationRecord]],
) -> tuple[Figure, np.ndarray]:
	"""Compare step work and normalized residuals across formulations."""
	if not records_by_label:
		raise ValueError("At least one labeled iteration series is required.")
	figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
	for label, records in records_by_label.items():
		if not isinstance(label, str) or not label:
			raise ValueError("Iteration comparison labels must be non-empty strings.")
		data = _record_arrays(records)
		axes[0].step(
			data["times"],
			data["iterations"],
			where="mid",
			label=label,
		)
		axes[1].semilogy(
			data["times"],
			_positive_for_log(data["ratios"]),
			label=label,
		)
	axis = axes[0]
	axis.set(
		title="Nonlinear iteration count by formulation",
		ylabel="iterations",
	)
	axis.yaxis.set_major_locator(MaxNLocator(integer=True))
	axes[1].axhline(
		1.0,
		color="black",
		linestyle="--",
		label="acceptance limit",
	)
	axes[1].set(
		title="Final residual relative to effective tolerance",
		xlabel="$t_{n+1}$",
		ylabel="residual / tolerance",
	)
	for axis in axes:
		axis.legend()
	return figure, axes


# Method-specific public names share the same record schema and presentation.
plot_implicit_abba_iteration_diagnostics = plot_implicit_iteration_diagnostics
plot_implicit_abba_iteration_comparison = plot_implicit_iteration_comparison
plot_implicit_bm4_iteration_diagnostics = plot_implicit_iteration_diagnostics
plot_implicit_bm4_iteration_comparison = plot_implicit_iteration_comparison


__all__ = [
	"plot_implicit_abba_iteration_comparison",
	"plot_implicit_abba_iteration_diagnostics",
	"plot_implicit_bm4_iteration_comparison",
	"plot_implicit_bm4_iteration_diagnostics",
	"plot_implicit_iteration_comparison",
	"plot_implicit_iteration_diagnostics",
]
