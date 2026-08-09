"""Reusable animations for single-particle full-cyclotron and GC solutions."""

from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from potential import Potential
from simulation.solution import Solution
from initial_conditions import TrajectoryFC, TrajectoryGC


def _frame_indices(sample_count: int, frames: int | None) -> np.ndarray:
	"""Select strictly increasing saved-state indices for an animation."""
	if frames is None:
		return np.arange(sample_count, dtype=int)
	if (
		isinstance(frames, (bool, np.bool_))
		or not isinstance(frames, (int, np.integer))
		or not 2 <= int(frames) <= sample_count
	):
		raise ValueError("`frames` must be None or an integer from 2 to the sample count.")
	return np.asarray(
		np.unique(np.linspace(0, sample_count - 1, int(frames), dtype=int)),
		dtype=int,
	)


def _field_normalization(fields: np.ndarray) -> mcolors.Normalize:
	"""Return a stable color scale for every frame of a potential animation."""
	minimum = float(np.min(fields))
	maximum = float(np.max(fields))
	if minimum < 0 < maximum:
		return mcolors.TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
	if np.isclose(minimum, maximum):
		delta = abs(minimum) * 0.01 or 1.0
		return mcolors.Normalize(vmin=minimum - delta, vmax=maximum + delta)
	return mcolors.Normalize(vmin=minimum, vmax=maximum)


def _animate_particle_solution(
	potential: Potential,
	solution: Solution,
	*,
	configuration_type: type[TrajectoryFC | TrajectoryGC],
	frames: int | None = 151,
	interval: int = 80,
	cmap: str = "RdBu_r",
	repeat: bool = True,
	**imshow_kwargs: Any,
) -> FuncAnimation:
	"""Animate one particle over a time-dependent periodic potential.

	The animation uses a stable color scale and shows the accumulated position
	path up to the current saved time. It intentionally accepts only one
	particle so its visual semantics remain appropriate for introductory studies.
	"""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(solution, Solution):
		raise TypeError("`solution` must be a Solution instance.")
	if not isinstance(solution.source, configuration_type):
		raise TypeError(
			"`solution` has an incompatible initial-configuration type for this animation."
		)
	if isinstance(interval, (bool, np.bool_)) or int(interval) <= 0:
		raise ValueError("`interval` must be a positive integer.")

	times = np.asarray(solution.t, dtype=float)
	x, y = solution.positions()
	if x.shape != (1, times.size) or y.shape != (1, times.size):
		raise ValueError("Single-particle animation requires exactly one particle.")
	indices = _frame_indices(times.size, frames)
	frame_times = times[indices]
	fields = np.asarray(potential.evaluate(frame_times), dtype=float)
	if fields.shape != (*potential.grid.shape, indices.size):
		raise ValueError("Potential evaluation returned an unexpected animation shape.")

	grid = potential.grid
	figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
	image = axis.imshow(
		fields[:, :, 0].T,
		origin="lower",
		extent=(grid.xmin, grid.xmin + grid.period, grid.ymin, grid.ymin + grid.period),
		aspect="equal",
		cmap=cmap,
		norm=_field_normalization(fields),
		**imshow_kwargs,
	)
	(path,) = axis.plot([], [], color="black", linewidth=1.5, label="Orbit")
	(marker,) = axis.plot(
		[], [], marker="o", color="white", markeredgecolor="black", markersize=7,
		label="Particle",
	)
	axis.set(xlabel="x", ylabel="y")
	axis.legend(loc="upper right")
	figure.colorbar(image, ax=axis, label="Potential")

	def update(frame: int) -> tuple[Any, ...]:
		"""Update field, accumulated orbit, and particle marker for one frame."""
		sample_index = int(indices[frame])
		image.set_data(fields[:, :, frame].T)
		path.set_data(x[0, : sample_index + 1], y[0, : sample_index + 1])
		marker.set_data([x[0, sample_index]], [y[0, sample_index]])
		axis.set_title(f"Single-particle orbit at t = {times[sample_index]:.3f}")
		return image, path, marker

	return FuncAnimation(
		figure,
		update,
		frames=indices.size,
		interval=int(interval),
		blit=False,
		repeat=repeat,
	)


def animate_fc_particle_solution(
	potential: Potential,
	solution: Solution,
	*,
	frames: int | None = 151,
	interval: int = 80,
	cmap: str = "RdBu_r",
	repeat: bool = True,
	**imshow_kwargs: Any,
) -> FuncAnimation:
	"""Animate one full-cyclotron particle over its physical potential."""
	return _animate_particle_solution(
		potential,
		solution,
		configuration_type=TrajectoryFC,
		frames=frames,
		interval=interval,
		cmap=cmap,
		repeat=repeat,
		**imshow_kwargs,
	)


def animate_gc_particle_solution(
	potential: Potential,
	solution: Solution,
	*,
	frames: int | None = 151,
	interval: int = 80,
	cmap: str = "RdBu_r",
	repeat: bool = True,
	**imshow_kwargs: Any,
) -> FuncAnimation:
	"""Animate one guiding-centre particle over its effective potential."""
	return _animate_particle_solution(
		potential,
		solution,
		configuration_type=TrajectoryGC,
		frames=frames,
		interval=interval,
		cmap=cmap,
		repeat=repeat,
		**imshow_kwargs,
	)


__all__ = ["animate_fc_particle_solution", "animate_gc_particle_solution"]
