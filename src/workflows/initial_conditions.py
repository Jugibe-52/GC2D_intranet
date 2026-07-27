"""Semantic initial-condition factories for common notebook experiments."""

from __future__ import annotations

import numpy as np

from classes import Area, Potential, TrajectoryGC


def domain_center(potential: Potential) -> tuple[float, float]:
	"""Return the geometric center of the potential's periodic base cell."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	grid = potential.grid
	return grid.xmin + grid.period / 2, grid.ymin + grid.period / 2


def centered_circle(
	potential: Potential,
	*,
	radius: float,
	points: int = 128,
	rho: float = 0.0,
) -> Area:
	"""Create a circular GC boundary centered in the periodic base cell."""
	return Area.circle(
		center=domain_center(potential),
		radius=radius,
		points=points,
		rho=rho,
	)


def centered_square(
	potential: Potential,
	*,
	side: float,
	points_per_side: int = 1,
	rho: float = 0.0,
) -> Area:
	"""Create a square GC boundary centered in the periodic base cell."""
	return Area.square(
		center=domain_center(potential),
		side=side,
		points_per_side=points_per_side,
		rho=rho,
	)


def centered_gc_trajectory(
	potential: Potential,
	*,
	rho: float = 0.0,
) -> TrajectoryGC:
	"""Create one GC initial condition at the center of the periodic base cell."""
	center_x, center_y = domain_center(potential)
	return TrajectoryGC.from_components(
		x=np.asarray([center_x]),
		y=np.asarray([center_y]),
		rho=rho,
	)


__all__ = [
	"centered_circle",
	"centered_gc_trajectory",
	"centered_square",
	"domain_center",
]
