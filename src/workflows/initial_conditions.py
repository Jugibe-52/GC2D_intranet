"""Compatibility helpers for constructing System initial states."""

from __future__ import annotations

from typing import Literal

import numpy as np

from classes import System
from contracts import Array


def make_initial_conditions(
	system: System,
	n_traj: int,
	*,
	method: Literal["random", "fixed"] = "fixed",
	x: Array | None = None,
	y: Array | None = None,
	rng: np.random.Generator | None = None,
) -> Array:
	"""Delegate initial-state construction to the independent Trajectory entity."""
	if not isinstance(system, System):
		raise TypeError("system must be a System instance.")
	if (
		not isinstance(n_traj, (int, np.integer))
		or isinstance(n_traj, (bool, np.bool_))
		or n_traj < 1
	):
		raise ValueError("n_traj must be a positive integer.")
	if (x is None) != (y is None):
		raise ValueError("x and y must be provided together.")
	if x is None:
		return system.initial_state(
			n_trajectories=int(n_traj),
			initialization=method,
			rng=rng,
		)
	x_axis = np.asarray(x)
	y_axis = np.asarray(y)
	if (
		x_axis.ndim != 1
		or y_axis.ndim != 1
		or x_axis.size < 2
		or y_axis.size < 2
	):
		raise ValueError(
			"x and y must be one-dimensional axes with at least two values."
		)
	return system.trajectory.initial_state(
		(float(x_axis[0]), float(x_axis[-1])),
		(float(y_axis[0]), float(y_axis[-1])),
		n_trajectories=int(n_traj),
		initialization=method,
		rng=rng,
	)


__all__ = ["make_initial_conditions"]
