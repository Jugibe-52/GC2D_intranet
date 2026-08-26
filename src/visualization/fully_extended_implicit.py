"""Plots for full-state implicit energy and symplecticity studies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class FullyExtendedRunView(Protocol):
	"""Arrays required by the analytic ``R^8``/``R^4`` plot and audits."""

	@property
	def actual_step(self) -> float: ...

	@property
	def symplecticity_times(self) -> np.ndarray: ...

	@property
	def r8_relative_defects(self) -> np.ndarray: ...

	@property
	def r8_determinant_errors(self) -> np.ndarray: ...

	@property
	def r4_relative_defects(self) -> np.ndarray: ...

	@property
	def r4_determinant_errors(self) -> np.ndarray: ...

	@property
	def dpsi_jacobian_audit_errors(self) -> np.ndarray: ...

	@property
	def dr_jacobian_audit_errors(self) -> np.ndarray: ...

	@property
	def r4_jacobian_audit_errors(self) -> np.ndarray: ...


def plot_fully_extended_symplecticity(
	runs: Sequence[FullyExtendedRunView],
	*,
	method_name: str,
) -> tuple[Figure, np.ndarray]:
	"""Plot analytic form defects and independent centered Jacobian audits."""
	values = tuple(runs)
	if not values:
		raise ValueError("At least one fully extended run is required.")
	figure, axes = plt.subplots(
		3,
		2,
		figsize=(13, 13),
		sharex=True,
		constrained_layout=True,
	)
	floor = float(np.finfo(float).eps)
	for run in values:
		times = np.asarray(run.symplecticity_times, dtype=float)
		series = tuple(
			np.asarray(item, dtype=float)
			for item in (
				run.r8_relative_defects,
				run.r8_determinant_errors,
				run.r4_relative_defects,
				run.r4_determinant_errors,
			)
		)
		audits = tuple(
			np.asarray(item, dtype=float)
			for item in (
				run.dpsi_jacobian_audit_errors,
				run.dr_jacobian_audit_errors,
				run.r4_jacobian_audit_errors,
			)
		)
		if (
			times.ndim != 1
			or times.size == 0
			or any(item.shape != times.shape for item in (*series, *audits))
			or not np.all(np.isfinite(times))
			or any(not np.all(np.isfinite(item)) for item in (*series, *audits))
			or any(np.any(item < 0.0) for item in (*series, *audits))
		):
			raise ValueError("Fully extended defect histories must be aligned.")
		label = rf"$h={run.actual_step:g}$"
		for axis, item in zip(axes.flat[:4], series, strict=True):
			axis.semilogy(times, np.maximum(item, floor), label=label)
		audit_line = axes[2, 0].semilogy(
			times,
			np.maximum(audits[0], floor),
			label=label,
		)[0]
		color = audit_line.get_color()
		axes[2, 1].semilogy(
			times,
			np.maximum(audits[1], floor),
			color=color,
			label=rf"$DR$, {label}",
		)
		axes[2, 1].semilogy(
			times,
			np.maximum(audits[2], floor),
			color=color,
			linestyle="--",
			label=rf"$D\Phi$, {label}",
		)
	axes[0, 0].set(
		title=rf"{method_name}: duplicated splitting on $\mathbb{{R}}^8$",
		ylabel=r"relative $\Omega_8$ defect",
	)
	axes[0, 1].set(title=r"Duplicated volume check", ylabel=r"$|\det D\Psi-1|$")
	axes[1, 0].set(
		title=r"Complete projected method on $\mathbb{R}^4$",
		xlabel="Time",
		ylabel=r"relative $\Omega_4$ defect",
	)
	axes[1, 1].set(
		title="Projected-map volume check",
		xlabel="Time",
		ylabel=r"$|\det D\widehat\Phi-1|$",
	)
	axes[2, 0].set(
		title=r"Analytic $D\Psi$ versus centered differences",
		xlabel="Time",
		ylabel="relative Jacobian error",
	)
	axes[2, 1].set(
		title=r"Analytic $DR$ and projected $D\Phi$ audits",
		xlabel="Time",
		ylabel="relative Jacobian error",
	)
	for axis in axes.flat:
		axis.grid(which="both", alpha=0.25)
		axis.legend()
	return figure, axes


__all__ = ["FullyExtendedRunView", "plot_fully_extended_symplecticity"]
