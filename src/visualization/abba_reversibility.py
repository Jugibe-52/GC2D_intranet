"""Plots for forward/backward implicit-ABBA tangent comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from diagnostics import ImplicitABBAReversibilitySample


def _validated_samples(
	samples: Sequence[ImplicitABBAReversibilitySample],
) -> tuple[ImplicitABBAReversibilitySample, ...]:
	"""Return a non-empty, time-ordered sequence of comparison samples."""
	values = tuple(samples)
	if not values:
		raise ValueError("At least one implicit ABBA reversibility sample is required.")
	if any(
		not isinstance(item, ImplicitABBAReversibilitySample) for item in values
	):
		raise TypeError("Every item must be an ImplicitABBAReversibilitySample.")
	if np.any(np.diff([item.end_time for item in values]) <= 0.0):
		raise ValueError("Implicit ABBA reversibility samples must be time ordered.")
	if any(item.method_name != values[0].method_name for item in values):
		raise ValueError("All reversibility samples must use the same ABBA method.")
	return values


def plot_implicit_abba_reversibility_diagnostics(
	samples: Sequence[ImplicitABBAReversibilitySample],
) -> tuple[Figure, np.ndarray]:
	"""Plot matrix reversibility and normalized proposed-increment closure."""
	values = _validated_samples(samples)
	times = np.asarray([item.end_time for item in values])
	method_name = values[0].method_name
	figure, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

	axes[0].semilogy(
		times,
		[item.jacobian_composition_defect_norm for item in values],
		label=r"$\|J^-_{n+1}J^+_n-I\|_F$",
	)
	axes[0].semilogy(
		times,
		[item.backward_state_error_norm for item in values],
		label=r"$\|\Psi_{-h}(\Psi_h(z_n))-z_n\|_2$",
	)
	axes[0].set(
		title=f"{method_name} forward/backward closure",
		ylabel="absolute norm",
	)
	axes[0].legend()

	axes[1].semilogy(
		times,
		[item.normalized_increment_closure for item in values],
		label=(
			r"$\|\Delta^+_n+\Delta^-_{n+1}\|_2/"
			r"\max(\|\Delta^+_n\|_2,\|\Delta^-_{n+1}\|_2)$"
		),
	)
	axes[1].set(
		title="Normalized proposed-increment closure",
		xlabel=r"$t_{n+1}$",
		ylabel="relative defect",
	)
	axes[1].legend()
	return figure, axes


def plot_implicit_abba_transport_components(
	samples: Sequence[ImplicitABBAReversibilitySample],
	*,
	particle_index: int = 0,
) -> tuple[Figure, np.ndarray]:
	"""Compare action and signed-increment components for one planar particle."""
	values = _validated_samples(samples)
	if isinstance(particle_index, (bool, np.bool_)) or not isinstance(
		particle_index,
		(int, np.integer),
	):
		raise TypeError("`particle_index` must be an integer.")
	particle_count = values[0].state_before.size // 2
	index = int(particle_index)
	if not 0 <= index < particle_count:
		raise IndexError("`particle_index` is outside the observed particle range.")
	indices = (index, particle_count + index)
	times = np.asarray([item.end_time for item in values])
	figure, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
	component_labels = ("x", "y")
	for column, (component, label) in enumerate(zip(indices, component_labels, strict=True)):
		axes[column].plot(
			times,
			[item.forward_increment[component] for item in values],
			label=r"$\Delta^+_n$",
		)
		axes[column].plot(
			times,
			[-item.backward_increment[component] for item in values],
			label=r"$-\Delta^-_{n+1}$",
		)
		axes[column].set(
			title=f"Forward versus reversed increment: {label} component",
			xlabel=r"$t_{n+1}$",
			ylabel="component value",
		)
		axes[column].legend()
	return figure, axes


__all__ = [
	"plot_implicit_abba_reversibility_diagnostics",
	"plot_implicit_abba_transport_components",
]
