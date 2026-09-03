"""Plots for physical and generalized GC energy histories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .implicit_comparison import IMPLICIT_METHOD_COLORS


class GeneralizedEnergyRunView(Protocol):
	"""Step-dependent arrays consumed by generalized-energy plots."""

	@property
	def actual_step(self) -> float:
		"""Accepted uniform integration step."""
		...

	@property
	def times(self) -> np.ndarray:
		"""Accepted-node time grid."""
		...

	@property
	def hamiltonian(self) -> np.ndarray:
		"""Physical Hamiltonian values."""
		...

	@property
	def kappa(self) -> np.ndarray:
		"""Normalized time-conjugate momentum values."""
		...

	@property
	def generalized_energy(self) -> np.ndarray:
		"""Generalized-energy values."""
		...

	@property
	def relative_errors(self) -> np.ndarray:
		"""Signed generalized-energy relative errors."""
		...

	@property
	def running_max_relative_errors(self) -> np.ndarray:
		"""Running maximum absolute relative errors."""
		...


class GeneralizedEnergySummaryView(Protocol):
	"""Scalar refinement fields consumed by the convergence plot."""

	@property
	def actual_step(self) -> float:
		"""Accepted uniform integration step."""
		...

	@property
	def max_absolute_error(self) -> float:
		"""Maximum absolute generalized-energy drift."""
		...


class EnergyAccuracySeriesView(Protocol):
	"""Physical-Hamiltonian reference errors consumed by comparison plots."""

	@property
	def errors(self) -> np.ndarray:
		"""Return signed particle-time Hamiltonian errors."""
		...

	@property
	def rms_error(self) -> np.ndarray:
		"""Return particle-RMS Hamiltonian errors over time."""
		...

	@property
	def running_maximum_absolute_error(self) -> np.ndarray:
		"""Return the running worst absolute particle error."""
		...

class ExtendedSymplecticityRunView(Protocol):
	"""Step-dependent arrays consumed by the ``R^6`` defect plot."""

	@property
	def actual_step(self) -> float:
		"""Accepted uniform integration step."""
		...

	@property
	def extended_symplecticity_times(self) -> np.ndarray:
		"""Accepted final times of the extended-map measurements."""
		...

	@property
	def extended_relative_defects(self) -> np.ndarray:
		"""Relative ``D Psi.T Omega D Psi - Omega`` defects."""
		...

	@property
	def extended_determinant_errors(self) -> np.ndarray:
		"""Absolute volume-determinant errors."""
		...


class ReducedExtendedSymplecticityRunView(Protocol):
	"""Arrays consumed by the complete projected ``R^4`` defect plot."""

	@property
	def actual_step(self) -> float:
		"""Accepted uniform integration step."""
		...

	@property
	def reduced_extended_symplecticity_times(self) -> np.ndarray:
		"""Accepted times of the projected physical map measurements."""
		...

	@property
	def reduced_extended_relative_defects(self) -> np.ndarray:
		"""Relative projected ``R^4`` symplecticity defects."""
		...

	@property
	def reduced_extended_determinant_errors(self) -> np.ndarray:
		"""Projected ``R^4`` determinant errors."""
		...

def _validated_runs(
	runs: Sequence[GeneralizedEnergyRunView],
) -> tuple[GeneralizedEnergyRunView, ...]:
	"""Require non-empty, shape-consistent finite energy histories."""
	values = tuple(runs)
	if not values:
		raise ValueError("At least one generalized-energy run is required.")
	for run in values:
		times = np.asarray(run.times, dtype=float)
		arrays = (
			np.asarray(run.hamiltonian, dtype=float),
			np.asarray(run.kappa, dtype=float),
			np.asarray(run.generalized_energy, dtype=float),
			np.asarray(run.relative_errors, dtype=float),
			np.asarray(run.running_max_relative_errors, dtype=float),
		)
		if (
			times.ndim != 1
			or times.size < 2
			or np.any(np.diff(times) <= 0.0)
			or any(array.shape != times.shape for array in arrays)
			or not np.all(np.isfinite(times))
			or any(not np.all(np.isfinite(array)) for array in arrays)
		):
			raise ValueError("Generalized-energy histories must share one finite grid.")
	return values


def plot_generalized_energy_components(
	run: GeneralizedEnergyRunView,
	*,
	method_name: str,
	momentum_symbol: str = r"\kappa",
) -> tuple[Figure, np.ndarray]:
	"""Show physical-energy variation, conjugate compensation, and residual drift."""
	_validated_runs((run,))
	symbol = str(momentum_symbol).strip()
	if not symbol:
		raise ValueError("`momentum_symbol` must not be empty.")
	times = np.asarray(run.times, dtype=float)
	hamiltonian = np.asarray(run.hamiltonian, dtype=float)
	kappa = np.asarray(run.kappa, dtype=float)
	energy = np.asarray(run.generalized_energy, dtype=float)
	figure, axes = plt.subplots(
		2,
		1,
		figsize=(10, 8),
		sharex=True,
		constrained_layout=True,
	)
	axes[0].plot(times, hamiltonian - hamiltonian[0], label=r"$h(t,z)-h(0,z_0)$")
	axes[0].plot(times, kappa, label=rf"${symbol}$")
	axes[0].plot(
		times,
		hamiltonian - hamiltonian[0] + kappa,
		linestyle="--",
		label=rf"$(h-h_0)+{symbol}$",
	)
	axes[0].set(
		title=(
			f"{method_name}: physical-energy variation and conjugate compensation "
			f"($h={run.actual_step:g}$)"
		),
		ylabel="Energy component",
	)
	axes[0].legend()
	axes[1].plot(times, energy - energy[0], color="C3")
	axes[1].axhline(0.0, color="0.45", linestyle="--", linewidth=1.0)
	axes[1].set(
		xlabel="Time",
		ylabel=r"$K_n-K_0$",
		title=rf"Residual generalized-energy drift, $K=h+{symbol}$",
	)
	for axis in axes:
		axis.grid(alpha=0.25)
		axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
	return figure, axes


def plot_generalized_energy_errors(
	runs: Sequence[GeneralizedEnergyRunView],
	*,
	method_name: str,
) -> tuple[Figure, np.ndarray]:
	"""Plot signed relative error and its running envelope for all refinements."""
	values = _validated_runs(runs)
	figure, axes = plt.subplots(
		2,
		1,
		figsize=(10, 9),
		sharex=True,
		constrained_layout=True,
	)
	prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
	positive_envelopes: list[np.ndarray] = []
	peaks: list[float] = []
	for run in values:
		times = np.asarray(run.times, dtype=float)
		relative_error = np.asarray(run.relative_errors, dtype=float)
		envelope = np.asarray(run.running_max_relative_errors, dtype=float)
		prepared.append((times, relative_error, envelope, run.actual_step))
		positive_envelopes.append(envelope[envelope > 0.0])
		peaks.append(float(np.max(np.abs(relative_error))))
	if any(values.size == 0 for values in positive_envelopes):
		raise ValueError("Every energy run must contain a non-zero error sample.")
	positive_floor = min(float(np.min(values)) for values in positive_envelopes) / 2.0
	linear_threshold = min(peaks) / 20.0
	for times, relative_error, envelope, actual_step in prepared:
		label = rf"$h={actual_step:g}$"
		axes[0].plot(times, relative_error, label=label)
		axes[1].semilogy(times, np.maximum(envelope, positive_floor), label=label)
	axes[0].axhline(0.0, color="0.45", linestyle="--", linewidth=1.0)
	axes[0].set_yscale("symlog", linthresh=linear_threshold)
	axes[0].set(
		title=f"{method_name}: signed generalized-energy error",
		ylabel=r"$\varepsilon_K=(K_n-K_0)/|K_0|$",
	)
	axes[1].set(
		title="Growth of the running error envelope",
		xlabel="Time",
		ylabel=r"$\max_{j\leq n}|\varepsilon_K(t_j)|$",
	)
	axes[1].set_ylim(positive_floor / 2.0, max(peaks) * 2.0)
	for axis in axes:
		axis.grid(which="both", alpha=0.25)
		axis.legend()
	return figure, axes


def plot_energy_accuracy_over_time(
	times: np.ndarray,
	series: Mapping[str, EnergyAccuracySeriesView],
	*,
	reference_energy_errors: np.ndarray,
) -> tuple[Figure, np.ndarray]:
	"""Plot Hamiltonian-observable errors against DOP853 and its Radau audit."""
	time_values = np.asarray(times, dtype=float)
	if (
		time_values.ndim != 1
		or time_values.size < 2
		or not np.all(np.isfinite(time_values))
		or np.any(np.diff(time_values) <= 0.0)
	):
		raise ValueError("Energy-accuracy times must be finite and increasing.")
	if not series:
		raise ValueError("At least one energy-accuracy series is required.")
	reference_errors = np.asarray(reference_energy_errors, dtype=float)
	if (
		reference_errors.ndim != 2
		or reference_errors.shape[1] != time_values.size
		or not np.all(np.isfinite(reference_errors))
	):
		raise ValueError("Reference energy errors must share the particle-time grid.")

	reference_rms = np.asarray(
		np.sqrt(np.mean(reference_errors**2, axis=0)),
		dtype=float,
	)
	reference_running = np.asarray(
		np.maximum.accumulate(np.max(np.abs(reference_errors), axis=0)),
		dtype=float,
	)
	prepared: list[tuple[str, np.ndarray, np.ndarray]] = []
	positive_values: list[np.ndarray] = []
	for label, values in series.items():
		errors = np.asarray(values.errors, dtype=float)
		rms = np.asarray(values.rms_error, dtype=float)
		running = np.asarray(values.running_maximum_absolute_error, dtype=float)
		if (
			errors.ndim != 2
			or errors.shape[1] != time_values.size
			or rms.shape != time_values.shape
			or running.shape != time_values.shape
			or not np.all(np.isfinite(errors))
			or np.any(rms < 0.0)
			or np.any(running < 0.0)
		):
			raise ValueError("Every energy series must share one finite time grid.")
		prepared.append((label, rms, running))
		positive_values.extend((rms[rms > 0.0], running[running > 0.0]))
	positive_values.extend(
		(reference_rms[reference_rms > 0.0], reference_running[reference_running > 0.0])
	)
	nonempty = [values for values in positive_values if values.size]
	plot_floor = (
		min(float(np.min(values)) for values in nonempty) / 2.0
		if nonempty
		else float(np.finfo(float).tiny)
	)

	figure, axes = plt.subplots(
		2,
		1,
		figsize=(12, 9),
		sharex=True,
		constrained_layout=True,
	)
	for index, (label, rms, running) in enumerate(prepared):
		color = IMPLICIT_METHOD_COLORS.get(label, f"C{index}")
		axes[0].semilogy(
			time_values,
			np.maximum(rms, plot_floor),
			color=color,
			label=label,
		)
		axes[1].semilogy(
			time_values,
			np.maximum(running, plot_floor),
			color=color,
			label=label,
		)
	for axis, reference_values in zip(
		axes,
		(reference_rms, reference_running),
		strict=True,
	):
		axis.semilogy(
			time_values,
			np.maximum(reference_values, plot_floor),
			color="black",
			linestyle=":",
			linewidth=1.3,
			label="DOP853/Radau energy discrepancy",
		)
		axis.grid(which="both", alpha=0.25)
		axis.legend(fontsize="small", ncol=2)
	axes[0].set(
		title="Particle-RMS Hamiltonian error against DOP853",
		ylabel=r"RMS $|H(t,z_h)-H(t,z_{ref})|$",
	)
	axes[1].set(
		title="Running worst particle Hamiltonian error",
		xlabel="Time",
		ylabel=r"$\max_{j\leq n,\,i}|\Delta H_i(t_j)|$",
	)
	return figure, axes


def plot_generalized_energy_convergence(
	summaries: Sequence[GeneralizedEnergySummaryView],
	*,
	method_name: str,
	expected_order: float,
) -> tuple[Figure, Axes]:
	"""Plot maximum generalized-energy drift against step with an order guide."""
	rows = tuple(summaries)
	if len(rows) < 2:
		raise ValueError("At least two energy summaries are required.")
	order = float(expected_order)
	if not np.isfinite(order) or order <= 0.0:
		raise ValueError("`expected_order` must be positive and finite.")
	steps = np.asarray([row.actual_step for row in rows], dtype=float)
	errors = np.asarray([row.max_absolute_error for row in rows], dtype=float)
	if (
		not np.all(np.isfinite(steps))
		or np.any(steps <= 0.0)
		or np.any(np.diff(steps) >= 0.0)
		or not np.all(np.isfinite(errors))
		or np.any(errors <= 0.0)
	):
		raise ValueError("Energy refinement steps and errors must be positive.")
	reference = errors[0] * (steps / steps[0]) ** order
	figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
	axis.loglog(steps, errors, marker="o", label="Measured max $|K_n-K_0|$")
	axis.loglog(
		steps,
		reference,
		color="black",
		linestyle="--",
		label=rf"$O(h^{{{order:g}}})$ guide",
	)
	axis.invert_xaxis()
	axis.set_xticks(steps, labels=[f"{step:g}" for step in steps])
	axis.set(
		title=f"{method_name}: generalized-energy refinement",
		xlabel="Accepted complete step $h$",
		ylabel=r"$\max_n |K_n-K_0|$",
	)
	axis.grid(which="both", alpha=0.25)
	axis.legend()
	return figure, axis


def plot_time_extended_symplecticity(
	runs: Sequence[ExtendedSymplecticityRunView],
	*,
	method_name: str,
) -> tuple[Figure, np.ndarray]:
	"""Plot ``R^6`` splitting symplecticity and determinant defects over time."""
	values = tuple(runs)
	if not values:
		raise ValueError("At least one extended-symplecticity run is required.")
	figure, axes = plt.subplots(
		2,
		1,
		figsize=(10, 8),
		sharex=True,
		constrained_layout=True,
	)
	floor = float(np.finfo(float).eps)
	for run in values:
		times = np.asarray(run.extended_symplecticity_times, dtype=float)
		defects = np.asarray(run.extended_relative_defects, dtype=float)
		determinants = np.asarray(run.extended_determinant_errors, dtype=float)
		if (
			times.ndim != 1
			or times.size == 0
			or defects.shape != times.shape
			or determinants.shape != times.shape
			or not np.all(np.isfinite(times))
			or not np.all(np.isfinite(defects))
			or not np.all(np.isfinite(determinants))
			or np.any(defects < 0.0)
			or np.any(determinants < 0.0)
		):
			raise ValueError("Extended-symplecticity histories must be finite and aligned.")
		label = rf"$h={run.actual_step:g}$"
		axes[0].semilogy(times, np.maximum(defects, floor), label=label)
		axes[1].semilogy(times, np.maximum(determinants, floor), label=label)
	axes[0].set(
		title=f"{method_name}: symplecticity of the accepted splitting on $R^6$",
		ylabel=r"$\|D\Psi^T\Omega_6D\Psi-\Omega_6\|_F/\|\Omega_6\|_F$",
	)
	axes[1].set(
		title="Necessary volume-preservation check",
		xlabel="Time",
		ylabel=r"$|\det(D\Psi)-1|$",
	)
	for axis in axes:
		axis.grid(which="both", alpha=0.25)
		axis.legend()
	return figure, axes


def plot_reduced_time_extended_symplecticity(
	runs: Sequence[ReducedExtendedSymplecticityRunView],
	*,
	method_name: str,
) -> tuple[Figure, np.ndarray]:
	"""Plot the complete projected method's defect on ``(x, y, t, kappa)``."""
	values = tuple(runs)
	if not values:
		raise ValueError("At least one reduced extended run is required.")
	figure, axes = plt.subplots(
		2,
		1,
		figsize=(10, 8),
		sharex=True,
		constrained_layout=True,
	)
	floor = float(np.finfo(float).eps)
	for run in values:
		times = np.asarray(run.reduced_extended_symplecticity_times, dtype=float)
		defects = np.asarray(run.reduced_extended_relative_defects, dtype=float)
		determinants = np.asarray(
			run.reduced_extended_determinant_errors,
			dtype=float,
		)
		if (
			times.ndim != 1
			or times.size == 0
			or defects.shape != times.shape
			or determinants.shape != times.shape
			or not np.all(np.isfinite(times))
			or not np.all(np.isfinite(defects))
			or not np.all(np.isfinite(determinants))
			or np.any(defects < 0.0)
			or np.any(determinants < 0.0)
		):
			raise ValueError("Reduced extended histories must be finite and aligned.")
		label = rf"$h={run.actual_step:g}$"
		axes[0].semilogy(times, np.maximum(defects, floor), label=label)
		axes[1].semilogy(times, np.maximum(determinants, floor), label=label)
	axes[0].set(
		title=(
			f"{method_name}: complete projected-map symplecticity on $R^4$"
		),
		ylabel=r"$\|D\widehat\Phi^T\Omega_4D\widehat\Phi-\Omega_4\|_F/\|\Omega_4\|_F$",
	)
	axes[1].set(
		title="Projected-map volume check",
		xlabel="Time",
		ylabel=r"$|\det(D\widehat\Phi)-1|$",
	)
	for axis in axes:
		axis.grid(which="both", alpha=0.25)
		axis.legend()
	return figure, axes


__all__ = [
	"EnergyAccuracySeriesView",
	"GeneralizedEnergyRunView",
	"GeneralizedEnergySummaryView",
	"ExtendedSymplecticityRunView",
	"ReducedExtendedSymplecticityRunView",
	"plot_generalized_energy_components",
	"plot_generalized_energy_convergence",
	"plot_generalized_energy_errors",
	"plot_energy_accuracy_over_time",
	"plot_time_extended_symplecticity",
	"plot_reduced_time_extended_symplecticity",
]
