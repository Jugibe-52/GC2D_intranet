"""Initial-state builders for potential trajectory simulations."""

from typing import Literal

import numpy as np

from classes.potential import PotentialHamsys
from classes.potential.potential import Array


def make_initial_conditions(
	potential_hamsys: PotentialHamsys,
	n_traj: int,
	*,
	method: Literal["random", "fixed"] = "fixed",
	x: Array | None = None,
	y: Array | None = None,
	rng: np.random.Generator | None = None,
) -> Array:
	"""Build an initial state compatible with ``potential_hamsys``.

	``random`` samples positions uniformly from the supplied axes. ``fixed``
	builds the largest square mesh with no more than ``n_traj`` points. Full
	cyclotron trajectories additionally receive unit perpendicular velocities
	with random phase.
	"""
	if not isinstance(potential_hamsys, PotentialHamsys):
		raise TypeError("`potential_hamsys` must be a PotentialHamsys instance.")
	if not isinstance(n_traj, (int, np.integer)) or isinstance(n_traj, bool) or n_traj < 1:
		raise ValueError("`n_traj` must be a positive integer.")
	x_axis = potential_hamsys.grid.x if x is None else np.asarray(x)
	y_axis = potential_hamsys.grid.y if y is None else np.asarray(y)
	if x_axis.ndim != 1 or y_axis.ndim != 1 or x_axis.size < 2 or y_axis.size < 2:
		raise ValueError("`x` and `y` must be one-dimensional axes with at least two values.")

	random = np.random.default_rng() if rng is None else rng
	if method == "random":
		x0 = random.uniform(x_axis[0], x_axis[-1], n_traj)
		y0 = random.uniform(y_axis[0], y_axis[-1], n_traj)
	elif method == "fixed":
		points_per_axis = int(np.sqrt(n_traj))
		x0 = np.linspace(x_axis[0], x_axis[-1], points_per_axis, endpoint=False)
		y0 = np.linspace(y_axis[0], y_axis[-1], points_per_axis, endpoint=False)
		x0, y0 = np.meshgrid(x0, y0, indexing="ij")
		x0, y0 = x0.ravel(), y0.ravel()
	else:
		raise ValueError("`method` must be either 'random' or 'fixed'.")

	positions = np.concatenate((x0, y0))
	if potential_hamsys.kind == "gc":
		return positions
	if potential_hamsys.kind == "fo":
		gyro_angle = random.uniform(0.0, 2 * np.pi, x0.size)
		return np.concatenate((positions, np.cos(gyro_angle), np.sin(gyro_angle)))
	raise ValueError(f"Unsupported potential system type: {potential_hamsys.kind!r}.")
