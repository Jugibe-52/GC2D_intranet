"""Reusable animations for single-particle full-cyclotron and GC solutions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from initial_conditions import FCInitialConfiguration, GCInitialConfiguration
from potential import Potential
from simulation.solution import Solution


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
	configuration_type: type[FCInitialConfiguration | GCInitialConfiguration],
	frames: int | None = 151,
	interval: int = 80,
	cmap: str = "RdBu_r",
	repeat: bool = True,
	show_electric_field: bool = False,
	frame_annotations: Sequence[str] | None = None,
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
	if not isinstance(show_electric_field, (bool, np.bool_)):
		raise TypeError("`show_electric_field` must be a boolean.")

	times = np.asarray(solution.t, dtype=float)
	x, y = solution.positions()
	if x.shape != (1, times.size) or y.shape != (1, times.size):
		raise ValueError("Single-particle animation requires exactly one particle.")
	annotations: tuple[str, ...] | None = None
	if frame_annotations is not None:
		if isinstance(frame_annotations, (str, bytes)):
			raise TypeError("`frame_annotations` must be a sequence of strings.")
		annotations = tuple(frame_annotations)
		if len(annotations) != times.size or not all(
			isinstance(value, str) and value for value in annotations
		):
			raise ValueError(
				"`frame_annotations` must contain one non-empty string per saved time."
			)
	indices = _frame_indices(times.size, frames)
	frame_times = times[indices]
	fields = np.asarray(potential.evaluate(frame_times), dtype=float)
	if fields.shape != (*potential.grid.shape, indices.size):
		raise ValueError("Potential evaluation returned an unexpected animation shape.")

	grid = potential.grid
	quiver = None
	electric_fields: tuple[tuple[np.ndarray, np.ndarray], ...] = ()
	quiver_x: np.ndarray | None = None
	quiver_y: np.ndarray | None = None
	quiver_scale: float | None = None
	if show_electric_field:
		# Roughly twenty arrows per axis keep the field legible over the potential.
		stride = max(1, int(np.ceil(max(grid.shape) / 20)))
		quiver_x, quiver_y = np.meshgrid(
			grid.x[::stride],
			grid.y[::stride],
			indexing="ij",
		)
		electric_fields = tuple(
			potential.electric_field(time, quiver_x, quiver_y)
			for time in frame_times
		)
		max_magnitude = max(
			float(np.max(np.hypot(field_x, field_y)))
			for field_x, field_y in electric_fields
		)
		arrow_length = 0.75 * stride * min(grid.dx, grid.dy)
		quiver_scale = (
			None if np.isclose(max_magnitude, 0.0) else max_magnitude / arrow_length
		)

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
	if show_electric_field:
		assert quiver_x is not None and quiver_y is not None
		quiver = axis.quiver(
			quiver_x,
			quiver_y,
			*electric_fields[0],
			color="black",
			angles="xy",
			scale_units="xy",
			scale=quiver_scale,
			width=0.003,
			alpha=0.75,
		)
	annotation = None
	if annotations is not None:
		annotation = axis.text(
			0.02,
			0.02,
			"",
			transform=axis.transAxes,
			verticalalignment="bottom",
			bbox={"facecolor": "white", "edgecolor": "0.35", "alpha": 0.85},
		)
	axis.set(xlabel="x", ylabel="y")
	axis.legend(loc="upper right")
	figure.colorbar(image, ax=axis, label="Potential")

	def update(frame: int) -> tuple[Any, ...]:
		"""Update field, accumulated orbit, and particle marker for one frame."""
		sample_index = int(indices[frame])
		image.set_data(fields[:, :, frame].T)
		if quiver is not None:
			quiver.set_UVC(*electric_fields[frame])
		path.set_data(x[0, : sample_index + 1], y[0, : sample_index + 1])
		marker.set_data([x[0, sample_index]], [y[0, sample_index]])
		if annotation is not None:
			assert annotations is not None
			annotation.set_text(annotations[sample_index])
		axis.set_title(f"Single-particle orbit at t = {times[sample_index]:.3f}")
		artists: list[Any] = [image]
		if quiver is not None:
			artists.append(quiver)
		artists.extend((path, marker))
		if annotation is not None:
			artists.append(annotation)
		return tuple(artists)

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
	show_electric_field: bool = False,
	frame_annotations: Sequence[str] | None = None,
	**imshow_kwargs: Any,
) -> FuncAnimation:
	"""Animate one full-cyclotron particle over its physical potential."""
	return _animate_particle_solution(
		potential,
		solution,
		configuration_type=FCInitialConfiguration,
		frames=frames,
		interval=interval,
		cmap=cmap,
		repeat=repeat,
		show_electric_field=show_electric_field,
		frame_annotations=frame_annotations,
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
	show_electric_field: bool = False,
	frame_annotations: Sequence[str] | None = None,
	**imshow_kwargs: Any,
) -> FuncAnimation:
	"""Animate one guiding-centre particle over its effective potential."""
	return _animate_particle_solution(
		potential,
		solution,
		configuration_type=GCInitialConfiguration,
		frames=frames,
		interval=interval,
		cmap=cmap,
		repeat=repeat,
		show_electric_field=show_electric_field,
		frame_annotations=frame_annotations,
		**imshow_kwargs,
	)


__all__ = ["animate_fc_particle_solution", "animate_gc_particle_solution"]
