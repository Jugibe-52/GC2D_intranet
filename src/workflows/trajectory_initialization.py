"""Workflow-level initialization of trajectory-owned particle states."""

from __future__ import annotations

from typing import Literal

import numpy as np

from classes import Grid, Trajectory, TrajectoryGC
from contracts import Array


def _spatial_bounds(grid: Grid) -> tuple[tuple[float, float], tuple[float, float]]:
	if not isinstance(grid, Grid):
		raise TypeError("`grid` must be a Grid instance.")
	if grid.period is None:
		return (grid.xmin, grid.xmax), (grid.ymin, grid.ymax)
	return (
		(grid.xmin, grid.xmin + grid.period),
		(grid.ymin, grid.ymin + grid.period),
	)


def _require_uninitialized(trajectory: Trajectory, replace: bool) -> None:
	if not isinstance(trajectory, Trajectory):
		raise TypeError("`trajectory` must be a Trajectory instance.")
	if trajectory.state is not None and not replace:
		raise ValueError(
			"The trajectory already has an initial state; pass `replace=True` "
			"to initialize it again."
		)


def initialize_trajectory(
	trajectory: Trajectory,
	grid: Grid,
	*,
	n_trajectories: int | None = None,
	initialization: Literal["random", "fixed", "selected"] | None = None,
	rng: np.random.Generator | np.random.RandomState | None = None,
	replace: bool = False,
) -> Array:
	"""Generate and assign a state before composing a ``System``.

	The workflow receives only the two independent inputs required for
	initialization: a trajectory and the potential grid that defines its domain.
	"""
	_require_uninitialized(trajectory, replace)
	x_bounds, y_bounds = _spatial_bounds(grid)
	state = trajectory.initial_state(
		x_bounds,
		y_bounds,
		n_trajectories=n_trajectories,
		initialization=initialization,
		rng=rng,
	)
	trajectory.set_initial_state(state)
	return state


def initialize_guiding_center_square(
	trajectory: TrajectoryGC,
	grid: Grid,
	*,
	side: float = 1.0,
	lower_left: tuple[float, float] | None = None,
	points_per_side: int = 1,
	replace: bool = False,
) -> Array:
	"""Assign a counter-clockwise square boundary without requiring a System.

	The ordering matches the polygon convention consumed later by
	``SystemResearch.guiding_center_polygon_area``.
	"""
	if not isinstance(trajectory, TrajectoryGC):
		raise TypeError("Square initialization requires a TrajectoryGC instance.")
	_require_uninitialized(trajectory, replace)
	if not isinstance(grid, Grid):
		raise TypeError("`grid` must be a Grid instance.")
	try:
		side = float(side)
	except (TypeError, ValueError) as exc:
		raise ValueError("`side` must be a positive finite number.") from exc
	if not np.isfinite(side) or side <= 0:
		raise ValueError("`side` must be a positive finite number.")
	if (
		isinstance(points_per_side, (bool, np.bool_))
		or not isinstance(points_per_side, (int, np.integer))
		or points_per_side < 1
	):
		raise ValueError("`points_per_side` must be a positive integer.")
	if lower_left is None:
		x0 = (grid.xmin + grid.xmax) / 2
		y0 = (grid.ymin + grid.ymax) / 2
	else:
		if len(lower_left) != 2:
			raise ValueError("`lower_left` must contain exactly two coordinates.")
		x0, y0 = map(float, lower_left)
		if not np.isfinite(x0) or not np.isfinite(y0):
			raise ValueError("`lower_left` must contain finite coordinates.")

	edge = np.linspace(0.0, side, int(points_per_side), endpoint=False)
	x_boundary = np.concatenate((
		x0 + edge,
		np.full_like(edge, x0 + side),
		x0 + side - edge,
		np.full_like(edge, x0),
	))
	y_boundary = np.concatenate((
		np.full_like(edge, y0),
		y0 + edge,
		np.full_like(edge, y0 + side),
		y0 + side - edge,
	))
	state: Array = np.asarray(np.concatenate((x_boundary, y_boundary)), dtype=float)
	trajectory.set_initial_state(state)
	return state


__all__ = ["initialize_guiding_center_square", "initialize_trajectory"]
