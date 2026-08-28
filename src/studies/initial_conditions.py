"""Semantic initial-condition factories for common notebook experiments."""

from __future__ import annotations

import numpy as np

from initial_conditions import (
	Area,
	GCInitialConfiguration,
	TrajectoryGC,
)
from potential import Potential


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


def centered_gc_configuration(potential: Potential) -> GCInitialConfiguration:
	"""Create one canonical GC initial state at the periodic-cell center."""
	center_x, center_y = domain_center(potential)
	return GCInitialConfiguration.from_components(
		x=np.asarray([center_x]),
		y=np.asarray([center_y]),
	)


def random_gc_configuration(
	potential: Potential,
	*,
	particle_count: int,
	seed: int,
	x_bounds: tuple[float, float] | None = None,
	y_bounds: tuple[float, float] | None = None,
) -> GCInitialConfiguration:
	"""Sample reproducible independent GC positions inside the periodic cell.

	Bounds default to the complete base cell and use NumPy's half-open uniform
	interval. Explicit bounds are useful for keeping a short visual comparison
	away from periodic edges while remaining fully reproducible.
	"""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if (
		isinstance(particle_count, (bool, np.bool_))
		or not isinstance(particle_count, (int, np.integer))
		or particle_count < 1
	):
		raise ValueError("`particle_count` must be a positive integer.")
	if (
		isinstance(seed, (bool, np.bool_))
		or not isinstance(seed, (int, np.integer))
		or seed < 0
	):
		raise ValueError("`seed` must be a non-negative integer.")
	grid = potential.grid

	def normalized_bounds(
		bounds: tuple[float, float] | None,
		cell_bounds: tuple[float, float],
		name: str,
	) -> tuple[float, float]:
		"""Validate one sampling interval inside the periodic base cell."""
		candidate = np.asarray(cell_bounds if bounds is None else bounds, dtype=float)
		if (
			candidate.shape != (2,)
			or not np.all(np.isfinite(candidate))
			or candidate[0] >= candidate[1]
			or candidate[0] < cell_bounds[0]
			or candidate[1] > cell_bounds[1]
		):
			raise ValueError(
				f"`{name}` must be a finite increasing interval inside the base cell."
			)
		return float(candidate[0]), float(candidate[1])

	x_interval = normalized_bounds(
		x_bounds,
		(grid.xmin, grid.xmin + grid.period),
		"x_bounds",
	)
	y_interval = normalized_bounds(
		y_bounds,
		(grid.ymin, grid.ymin + grid.period),
		"y_bounds",
	)
	random = np.random.default_rng(int(seed))
	return GCInitialConfiguration.from_components(
		x=random.uniform(*x_interval, size=int(particle_count)),
		y=random.uniform(*y_interval, size=int(particle_count)),
	)


def latin_hypercube_gc_configuration(
	potential: Potential,
	*,
	particle_count: int,
	seed: int,
	domain_margin_fraction: float = 0.0,
) -> GCInitialConfiguration:
	"""Place a reproducible heterogeneous sample across the grid extent.

	Each spatial axis is divided into ``particle_count`` equally sized strata.
	Independent jitter and permutations put exactly one guiding center in every
	x and y stratum without forcing the two-dimensional points onto a regular
	lattice. ``domain_margin_fraction`` removes the same fraction from both ends
	of each sampled axis, which is useful for non-periodic spline boundaries.
	"""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if (
		isinstance(particle_count, (bool, np.bool_))
		or not isinstance(particle_count, (int, np.integer))
		or particle_count < 1
	):
		raise ValueError("`particle_count` must be a positive integer.")
	if (
		isinstance(seed, (bool, np.bool_))
		or not isinstance(seed, (int, np.integer))
		or seed < 0
	):
		raise ValueError("`seed` must be a non-negative integer.")
	margin = float(domain_margin_fraction)
	if not np.isfinite(margin) or margin < 0.0 or margin >= 0.5:
		raise ValueError("`domain_margin_fraction` must lie in [0, 0.5).")

	grid = potential.grid
	domain_min = np.asarray((grid.xmin, grid.ymin), dtype=float)
	domain_max = np.asarray((grid.xmax, grid.ymax), dtype=float)
	domain_span = domain_max - domain_min
	sampling_min = domain_min + margin * domain_span
	sampling_max = domain_max - margin * domain_span
	random = np.random.default_rng(int(seed))
	count = int(particle_count)
	unit_x = (np.arange(count) + random.random(count)) / count
	unit_y = (np.arange(count) + random.random(count)) / count
	random.shuffle(unit_x)
	random.shuffle(unit_y)
	return GCInitialConfiguration.from_components(
		x=sampling_min[0] + unit_x * (sampling_max[0] - sampling_min[0]),
		y=sampling_min[1] + unit_y * (sampling_max[1] - sampling_min[1]),
	)


__all__ = [
	"centered_circle",
	"centered_gc_configuration",
	"centered_gc_trajectory",
	"centered_square",
	"domain_center",
	"latin_hypercube_gc_configuration",
	"random_gc_configuration",
]
