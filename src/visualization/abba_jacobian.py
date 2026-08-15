"""Matplotlib views of local implicit-ABBA Jacobian analyses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from diagnostics.abba_jacobian import ImplicitABBAJacobianSample

if TYPE_CHECKING:
	from studies.abba_implicit_jacobian import (
		ImplicitABBAJacobianParticleStepSeries,
	)


_CLASS_COLORS = {
	"hyperbolic": "tab:red",
	"elliptic": "tab:blue",
	"parabolic": "0.45",
}


def _validated_samples(
	samples: Sequence[ImplicitABBAJacobianSample],
	particle_index: int,
) -> tuple[ImplicitABBAJacobianSample, ...]:
	"""Require non-empty ordered samples containing the selected particle."""
	values = tuple(samples)
	if not values:
		raise ValueError("At least one implicit ABBA Jacobian sample is required.")
	if isinstance(particle_index, (bool, np.bool_)) or not isinstance(
		particle_index, (int, np.integer)
	):
		raise TypeError("`particle_index` must be an integer.")
	index = int(particle_index)
	particle_count = len(values[0].particle_analyses)
	if not 0 <= index < particle_count:
		raise IndexError("`particle_index` is outside the analyzed particle range.")
	for previous, current in zip(values, values[1:]):
		if current.step_index <= previous.step_index:
			raise ValueError("Jacobian samples must be ordered by increasing step.")
	for sample in values:
		if len(sample.particle_analyses) != particle_count:
			raise ValueError("The particle count changed between Jacobian samples.")
	return values


def _particle_arrays(
	samples: tuple[ImplicitABBAJacobianSample, ...],
	particle_index: int,
) -> dict[str, np.ndarray]:
	"""Collect consistently shaped arrays for one particle's time series."""
	analyses = [sample.particle_analyses[particle_index] for sample in samples]
	return {
		"times": np.asarray([sample.end_time for sample in samples]),
		"durations": np.asarray([sample.duration for sample in samples]),
		"jacobians": np.stack([item.jacobian for item in analyses]),
		"traces": np.asarray([item.trace for item in analyses]),
		"determinants": np.asarray([item.determinant for item in analyses]),
		"discriminants": np.asarray([item.discriminant for item in analyses]),
		"discriminant_tolerances": np.asarray(
			[item.discriminant_tolerance for item in analyses]
		),
		"condition_numbers": np.asarray(
			[item.condition_number for item in analyses]
		),
		"spectral_radii": np.asarray([item.spectral_radius for item in analyses]),
		"eigenvalues": np.stack([item.eigenvalues for item in analyses]),
		"eigenline_angles": np.stack(
			[item.eigenvector_line_angles for item in analyses]
		),
		"singular_values": np.stack([item.singular_values for item in analyses]),
		"singular_line_angles": np.stack(
			[item.singular_vector_line_angles for item in analyses]
		),
		"classes": np.asarray([item.spectral_class for item in analyses]),
	}


def _positive_for_log(values: np.ndarray) -> np.ndarray:
	"""Replace exact zeros by the smallest positive float for log plotting."""
	return np.maximum(np.asarray(values, dtype=float), np.finfo(float).tiny)


def _line_angles_for_plot(values: np.ndarray) -> np.ndarray:
	"""Break curves at branch-cut jumps of an unoriented line angle."""
	angles = np.asarray(values, dtype=float).copy()
	finite_pairs = np.isfinite(angles[:-1]) & np.isfinite(angles[1:])
	jumps = finite_pairs & (np.abs(np.diff(angles)) > np.pi / 2.0)
	angles[1:][jumps] = np.nan
	return angles


def _representative_snapshot_indices(
	samples: tuple[ImplicitABBAJacobianSample, ...],
	particle_index: int,
	count: int,
) -> np.ndarray:
	"""Select time-spanning snapshots while representing every spectral class."""
	classes = np.asarray(
		[
			sample.particle_analyses[particle_index].spectral_class
			for sample in samples
		]
	)
	required = {
		int(indices[len(indices) // 2])
		for spectral_class in np.unique(classes)
		if (indices := np.flatnonzero(classes == spectral_class)).size
	}
	baseline = list(np.linspace(0, len(samples) - 1, count, dtype=int))
	selected = set(required)
	for index in baseline:
		if len(selected) >= count:
			break
		selected.add(int(index))
	if len(selected) < count:
		for index in range(len(samples)):
			if len(selected) >= count:
				break
			selected.add(index)
	if len(selected) > count:
		optional = sorted(selected - required)
		keep_optional = max(0, count - len(required))
		if keep_optional:
			positions = np.linspace(0, len(optional) - 1, keep_optional, dtype=int)
			selected = required | {optional[int(position)] for position in positions}
		else:
			selected = set(sorted(required)[:count])
	return np.asarray(sorted(selected), dtype=int)


def plot_implicit_abba_particle_step_series(
	series: ImplicitABBAJacobianParticleStepSeries,
) -> tuple[Figure, np.ndarray]:
	"""Plot endpoint fields and finite-increment magnitude and direction."""
	times = np.asarray(series.end_times, dtype=float)
	electric_fields = np.asarray(
		series.effective_electric_fields_after,
		dtype=float,
	)
	increments = np.asarray(series.state_increments, dtype=float)
	increment_norms = np.asarray(series.state_increment_norms, dtype=float)
	increment_angles = np.asarray(series.state_increment_angles, dtype=float)
	if (
		times.ndim != 1
		or electric_fields.shape != (2, times.size)
		or increments.shape != (2, times.size)
		or increment_norms.shape != times.shape
		or increment_angles.shape != times.shape
		or not np.all(np.isfinite(electric_fields))
		or not np.all(np.isfinite(increments))
		or not np.all(np.isfinite(increment_norms))
		or not np.all(np.isfinite(increment_angles) | np.isnan(increment_angles))
	):
		raise ValueError("Particle step series contains invalid array shapes.")
	figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

	for component, label in enumerate(
		(r"$E_{\mathrm{eff},x}$", r"$E_{\mathrm{eff},y}$")
	):
		axes[0].plot(times, electric_fields[component], label=label)
	axes[0].plot(
		times,
		np.linalg.norm(electric_fields, axis=0),
		color="black",
		linestyle="--",
		label=r"$|E_{\mathrm{eff}}|$",
	)
	axes[0].set(
		title="Effective electric field at step endpoints",
		xlabel="$t_{n+1}$",
		ylabel="electric field",
	)
	axes[0].legend()

	for component, label in enumerate((r"$\Delta x_n$", r"$\Delta y_n$")):
		axes[1].plot(times, increments[component], label=label)
	axes[1].plot(
		times,
		increment_norms,
		color="black",
		linestyle="--",
		label=r"$|\Delta z_n|$",
	)
	axes[1].axhline(0.0, color="0.65", linewidth=0.8)
	axes[1].set(
		title=r"Finite state increment $\Delta z_n=z_{n+1}-z_n$",
		xlabel="$t_{n+1}$",
		ylabel="state increment",
	)
	axes[1].legend()

	finite_directions = np.isfinite(increment_angles)
	axes[2].scatter(
		times[finite_directions],
		increment_angles[finite_directions],
		s=14,
		color="tab:purple",
		label=r"$\theta_{\Delta z}=\operatorname{atan2}(\Delta y,\Delta x)$",
	)
	axes[2].set(
		title=r"Oriented direction of $\Delta z_n$",
		xlabel="$t_{n+1}$",
		ylabel="direction (rad)",
		ylim=(-np.pi, np.pi),
	)
	axes[2].set_yticks(
		[-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi],
		labels=[r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"],
	)
	axes[2].legend()

	figure.suptitle(
		"Implicit ABBA field and finite step variation — "
		f"particle {series.particle_index}"
	)
	return figure, axes


def plot_implicit_abba_jacobian_matrices(
	samples: Sequence[ImplicitABBAJacobianSample],
	*,
	particle_index: int = 0,
) -> tuple[Figure, np.ndarray]:
	"""Plot matrix entries and scalar invariants for one particle over time."""
	values = _validated_samples(samples, particle_index)
	data = _particle_arrays(values, particle_index)
	times = data["times"]
	matrices = data["jacobians"] - np.eye(2)[None, :, :]
	figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
	for row in range(2):
		for column in range(2):
			axes[0, 0].plot(
				times,
				matrices[:, row, column],
				label=rf"$J_{{{row + 1}{column + 1}}}$",
			)
	axes[0, 0].set(
		title="Local Jacobian increments",
		xlabel="$t_{n+1}$",
		ylabel=r"entry of $J_n-I$",
	)
	axes[0, 0].legend(ncol=2)

	axes[0, 1].plot(times, data["traces"], color="tab:blue")
	axes[0, 1].set(
		title="Jacobian trace",
		xlabel="$t_{n+1}$",
		ylabel=r"$\mathrm{tr}(J_n)$",
	)

	axes[0, 2].plot(times, data["determinants"], color="tab:orange")
	axes[0, 2].set(
		title="Jacobian determinant",
		xlabel="$t_{n+1}$",
		ylabel=r"$\det(J_n)$",
	)

	discriminants = data["discriminants"]
	linthresh = max(
		float(np.max(data["discriminant_tolerances"])),
		np.finfo(float).eps,
	)
	axes[1, 0].axhline(0.0, color="0.25", linewidth=1.0)
	for spectral_class, color in _CLASS_COLORS.items():
		mask = data["classes"] == spectral_class
		axes[1, 0].scatter(
			times[mask],
			discriminants[mask],
			s=20,
			color=color,
			label=spectral_class,
		)
	axes[1, 0].set_yscale("symlog", linthresh=linthresh)
	axes[1, 0].set(
		title="Characteristic discriminant",
		xlabel="$t_{n+1}$",
		ylabel=r"$\Delta=(\mathrm{tr}J_n)^2-4\det J_n$",
	)
	axes[1, 0].legend()

	axes[1, 1].semilogy(
		times,
		_positive_for_log(data["condition_numbers"]),
		color="tab:purple",
	)
	axes[1, 1].set(
		title="Jacobian condition number",
		xlabel="$t_{n+1}$",
		ylabel=r"$\kappa_2(J_n)$",
	)

	axes[1, 2].plot(times, data["spectral_radii"], color="tab:green")
	axes[1, 2].set(
		title="Spectral radius",
		xlabel="$t_{n+1}$",
		ylabel=r"$\rho(J_n)=\max_i|\lambda_i|$",
	)
	figure.suptitle(f"Implicit ABBA local Jacobians — particle {particle_index}")
	return figure, axes


def plot_implicit_abba_jacobian_spectrum(
	samples: Sequence[ImplicitABBAJacobianSample],
	*,
	particle_index: int = 0,
) -> tuple[Figure, np.ndarray]:
	"""Plot eigenvalues, spectral rates, and singular values over time."""
	values = _validated_samples(samples, particle_index)
	data = _particle_arrays(values, particle_index)
	times = data["times"]
	durations = np.abs(data["durations"])
	eigenvalues = data["eigenvalues"]
	figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

	colors = np.repeat(times, 2)
	points = eigenvalues.reshape(-1)
	offsets = points - 1.0
	scatter = axes[0, 0].scatter(
		offsets.real,
		offsets.imag,
		c=colors,
		cmap="viridis",
		s=24,
	)
	axes[0, 0].axhline(0.0, color="0.65", linewidth=0.8)
	axes[0, 0].axvline(0.0, color="0.65", linewidth=0.8)
	axes[0, 0].set(
		title="Eigenvalue offsets from the identity",
		xlabel=r"$\operatorname{Re}(\lambda-1)$",
		ylabel=r"$\operatorname{Im}\lambda$",
		aspect="equal",
	)
	figure.colorbar(scatter, ax=axes[0, 0], label="$t_{n+1}$")

	for branch in range(2):
		axes[0, 1].plot(
			times,
			np.abs(eigenvalues[:, branch]),
			label=rf"$|\lambda_{branch + 1}|$",
		)
	axes[0, 1].set(
		title="Eigenvalue moduli",
		xlabel="$t_{n+1}$",
		ylabel=r"$|\lambda|$",
	)
	axes[0, 1].legend()

	for branch in range(2):
		axes[1, 0].plot(
			times,
			np.angle(eigenvalues[:, branch]) / durations,
			label=rf"$\arg(\lambda_{branch + 1})/|h|$",
		)
	axes[1, 0].set(
		title="Discrete spectral angular rates",
		xlabel="$t_{n+1}$",
		ylabel="angle per unit time",
	)
	axes[1, 0].legend()

	singular_values = data["singular_values"]
	for branch, label in enumerate((r"$\log(\sigma_{\max})/|h|$", r"$\log(\sigma_{\min})/|h|$")):
		axes[1, 1].plot(
			times,
			np.log(singular_values[:, branch]) / durations,
			label=label,
		)
	axes[1, 1].axhline(0.0, color="0.5", linestyle="--", linewidth=1.0)
	axes[1, 1].set(
		title="Principal one-step stretching rates",
		xlabel="$t_{n+1}$",
		ylabel="logarithmic rate",
	)
	axes[1, 1].legend()
	figure.suptitle(f"Implicit ABBA local spectrum — particle {particle_index}")
	return figure, axes


def plot_implicit_abba_jacobian_directions(
	samples: Sequence[ImplicitABBAJacobianSample],
	*,
	particle_index: int = 0,
) -> tuple[Figure, np.ndarray]:
	"""Plot available real eigenlines and principal singular directions."""
	values = _validated_samples(samples, particle_index)
	data = _particle_arrays(values, particle_index)
	times = data["times"]
	figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

	eigen_angles = data["eigenline_angles"]
	axes[0].plot(
		times,
		_line_angles_for_plot(eigen_angles[:, 0]),
		".-",
		label="smaller-modulus eigenline",
	)
	axes[0].plot(
		times,
		_line_angles_for_plot(eigen_angles[:, 1]),
		".-",
		label="larger-modulus eigenline",
	)
	axes[0].set(
		title="Real eigendirections (hyperbolic samples only)",
		xlabel="$t_{n+1}$",
		ylabel="line angle [rad]",
		ylim=(-np.pi / 2.0, np.pi / 2.0),
	)
	axes[0].set_yticks(
		[-np.pi / 2.0, -np.pi / 4.0, 0.0, np.pi / 4.0, np.pi / 2.0],
		[r"$-\pi/2$", r"$-\pi/4$", "$0$", r"$\pi/4$", r"$\pi/2$"],
	)
	axes[0].legend()

	singular_angles = data["singular_line_angles"]
	axes[1].plot(
		times,
		_line_angles_for_plot(singular_angles[:, 0]),
		".-",
		label="maximum-stretch line",
	)
	axes[1].plot(
		times,
		_line_angles_for_plot(singular_angles[:, 1]),
		".-",
		label="minimum-stretch line",
	)
	axes[1].set(
		title="Right singular-vector directions",
		xlabel="$t_{n+1}$",
		ylabel="line angle [rad]",
		ylim=(-np.pi / 2.0, np.pi / 2.0),
	)
	axes[1].set_yticks(
		[-np.pi / 2.0, -np.pi / 4.0, 0.0, np.pi / 4.0, np.pi / 2.0],
		[r"$-\pi/2$", r"$-\pi/4$", "$0$", r"$\pi/4$", r"$\pi/2$"],
	)
	axes[1].legend()
	figure.suptitle(f"Implicit ABBA Jacobian directions — particle {particle_index}")
	return figure, axes


def plot_implicit_abba_jacobian_polar_snapshots(
	samples: Sequence[ImplicitABBAJacobianSample],
	*,
	particle_index: int = 0,
	snapshot_count: int = 6,
) -> tuple[Figure, np.ndarray]:
	"""Show real eigenlines and singular directions at selected step snapshots."""
	values = _validated_samples(samples, particle_index)
	if (
		isinstance(snapshot_count, (bool, np.bool_))
		or not isinstance(snapshot_count, (int, np.integer))
		or snapshot_count < 1
	):
		raise ValueError("`snapshot_count` must be a positive integer.")
	count = min(int(snapshot_count), len(values))
	indices = _representative_snapshot_indices(values, particle_index, count)
	figure, axes = plt.subplots(
		1,
		len(indices),
		figsize=(3.3 * len(indices), 3.4),
		subplot_kw={"projection": "polar"},
		constrained_layout=True,
		squeeze=False,
	)
	polar_axes = axes[0]
	for position, sample_index in enumerate(indices):
		sample = values[int(sample_index)]
		analysis = sample.particle_analyses[particle_index]
		axis = polar_axes[position]
		axis.set_thetamin(-90.0)
		axis.set_thetamax(90.0)
		axis.set_ylim(0.0, 1.0)
		axis.set_yticks([])
		axis.set_title(
			f"step {sample.step_index}\n$t={sample.end_time:.3g}$\n"
			f"{analysis.spectral_class}",
			fontsize=10,
		)
		if analysis.eigendirections_defined:
			for branch, (color, label) in enumerate(
				(
					("tab:blue", "smaller-modulus eigenline"),
					("tab:red", "larger-modulus eigenline"),
				)
			):
				angle = analysis.eigenvector_line_angles[branch]
				axis.plot([angle, angle], [0.0, 1.0], color=color, linewidth=2.2, label=label)
		else:
			axis.text(
				0.5,
				0.45,
				"no reliable\nreal eigenline",
				transform=axis.transAxes,
				ha="center",
				va="center",
				fontsize=9,
			)
		if analysis.singular_directions_defined:
			for branch, label in enumerate(("maximum stretch", "minimum stretch")):
				angle = analysis.singular_vector_line_angles[branch]
				axis.plot(
					[angle, angle],
					[0.0, 0.78],
					color="0.25",
					linestyle="--" if branch == 0 else ":",
					linewidth=1.4,
					label=label,
				)
	if len(polar_axes):
		handles, labels = polar_axes[0].get_legend_handles_labels()
		if handles:
			figure.legend(handles, labels, loc="outside lower center", ncol=4)
	figure.suptitle(f"Jacobian direction snapshots — particle {particle_index}")
	return figure, polar_axes


__all__ = [
	"plot_implicit_abba_jacobian_directions",
	"plot_implicit_abba_jacobian_matrices",
	"plot_implicit_abba_jacobian_polar_snapshots",
	"plot_implicit_abba_jacobian_spectrum",
]
