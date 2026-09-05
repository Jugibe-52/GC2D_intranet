"""Optional Matplotlib presentation for periodic electrostatic potentials."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from potential import Potential


def _colour_scale(field: np.ndarray) -> mcolors.Normalize:
	"""Choose a stable normalization centered on zero when appropriate."""
	vmin = float(np.nanmin(field))
	vmax = float(np.nanmax(field))
	if vmin < 0 < vmax:
		return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
	if np.isclose(vmin, vmax):
		delta = abs(vmin) * 0.01 or 1.0
		return mcolors.Normalize(vmin=vmin - delta, vmax=vmax + delta)
	return mcolors.Normalize(vmin=vmin, vmax=vmax)


def plot_potential(
	potential: Potential,
	*,
	t: float = 0.0,
	contours: int | Sequence[float] | None = 12,
	cmap: str = "RdBu_r",
	show: bool = True,
	**pcolormesh_kwargs: Any,
) -> tuple[Figure, Axes]:
	"""Plot one potential field without coupling presentation to the model."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	field = potential.evaluate(t)
	figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
	mesh = axis.pcolormesh(
		potential.grid.x,
		potential.grid.y,
		field.T,
		shading="auto",
		cmap=cmap,
		norm=_colour_scale(field),
		**pcolormesh_kwargs,
	)
	if contours is not None:
		axis.contour(
			potential.grid.x,
			potential.grid.y,
			field.T,
			levels=contours,
			colors="black",
			linewidths=0.45,
			alpha=0.55,
		)
	figure.colorbar(mesh, ax=axis, label=r"$\phi$")
	axis.set(
		xlabel="x",
		ylabel="y",
		title=rf"Potential, $t={t:.3f}$",
		aspect="equal",
	)
	if show:
		plt.show()
	return figure, axis


def animate_potential(
	potential: Potential,
	*,
	t_max: float = 1.0,
	frames: int | None = None,
	interval: int = 200,
	cmap: str = "RdBu_r",
	repeat: bool = True,
	**pcolormesh_kwargs: Any,
) -> FuncAnimation:
	"""Animate normalized cycles with 10 frames per cycle at 5 fps."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	t_max = float(t_max)
	if not np.isfinite(t_max) or t_max <= 0:
		raise ValueError("`t_max` must be positive and finite.")
	if frames is None:
		# A normalized cycle has length one. Ceil retains at least 10 saved
		# frames per cycle when the requested horizon is not an integer.
		frame_count = max(2, int(np.ceil(10.0 * t_max)))
	elif not isinstance(frames, int) or isinstance(frames, bool) or frames < 2:
		raise ValueError("`frames` must be None or an integer of at least 2.")
	else:
		frame_count = frames
	times = np.linspace(0.0, t_max, frame_count, endpoint=False)
	fields = [potential.evaluate(time) for time in times]
	stride = max(1, int(np.ceil(max(potential.grid.shape) / 20)))
	quiver_x, quiver_y = np.meshgrid(
		potential.grid.x[::stride],
		potential.grid.y[::stride],
		indexing="ij",
	)
	electric_fields = [
		potential.electric_field(time, quiver_x, quiver_y) for time in times
	]
	vmin = min(float(np.min(field)) for field in fields)
	vmax = max(float(np.max(field)) for field in fields)
	norm = (
		mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
		if vmin < 0 < vmax
		else _colour_scale(np.asarray([vmin, vmax]))
	)
	figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
	mesh = axis.pcolormesh(
		potential.grid.x,
		potential.grid.y,
		fields[0].T,
		shading="auto",
		cmap=cmap,
		norm=norm,
		**pcolormesh_kwargs,
	)
	figure.colorbar(mesh, ax=axis, label=r"$\phi$")
	quiver = axis.quiver(
		quiver_x,
		quiver_y,
		*electric_fields[0],
		color="black",
		width=0.003,
	)
	axis.set(xlabel="x", ylabel="y", aspect="equal")

	def update(index: int) -> tuple[Any, ...]:
		mesh.set_array(fields[index].T)
		quiver.set_UVC(*electric_fields[index])
		axis.set_title(rf"Potential, $t={times[index]:.3f}$")
		return mesh, quiver, axis.title

	update(0)
	animation = FuncAnimation(
		figure,
		update,
		frames=frame_count,
		interval=interval,
		blit=False,
		repeat=repeat,
	)
	plt.close(figure)
	return animation


__all__ = ["animate_potential", "plot_potential"]
