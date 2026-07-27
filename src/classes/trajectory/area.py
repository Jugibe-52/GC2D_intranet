"""Closed guiding-centre boundaries and their enclosed signed area."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .gc import TrajectoryGC


class Area(TrajectoryGC):
	"""A counter-clockwise square or circular GC boundary.

	The ``N`` particles of this trajectory represent ordered boundary vertices,
	not samples of the interior.  Their state has the regular GC layout
	``[x_1, ..., x_N, y_1, ..., y_N]``, so an ``Area`` can be passed directly to
	a guiding-centre initial-value problem. The vertices omit a repeated closing
	point because
	polygon operations close the last vertex back to the first implicitly.
	"""

	def __init__(
		self,
		state: np.ndarray,
		*,
		shape: Literal["square", "circle"],
		rho: float = 0.0,
	) -> None:
		"""Create a closed boundary from at least three GC vertex positions.

		``shape`` records which constructor produced the initial contour; the
		integrated vertices may subsequently deform away from that geometry.
		"""
		if shape not in ("square", "circle"):
			raise ValueError("`shape` must be 'square' or 'circle'.")
		self.shape = shape
		super().__init__(state, rho=rho)
		x, _ = self.positions(self._required_state())
		if x.size < 3:
			raise ValueError("An area boundary requires at least three points.")

	@classmethod
	def from_components(
		cls,
		*,
		x: np.ndarray,
		y: np.ndarray,
		shape: Literal["square", "circle"] = "square",
		rho: float = 0.0,
	) -> Area:
		"""Create a boundary from named coordinates and an optional shape label."""
		state = cls.pack_components(x, y)
		return cls(state, shape=shape, rho=rho)

	@classmethod
	def square(
		cls,
		*,
		center: tuple[float, float],
		side: float,
		points_per_side: int = 1,
		rho: float = 0.0,
	) -> Area:
		"""Sample a square counter-clockwise with equal density on every edge.

		``center`` locates the geometric centre, ``side`` is its full side
		length, and ``points_per_side`` controls boundary resolution.  The final
		state contains ``4 * points_per_side`` vertices.
		"""
		center_x, center_y = cls._validate_center(center)
		side = cls._positive_finite(side, "side")
		if (
			isinstance(points_per_side, (bool, np.bool_))
			or not isinstance(points_per_side, (int, np.integer))
			or points_per_side < 1
		):
			raise ValueError("`points_per_side` must be a positive integer.")

		# These four bounds describe the geometry; ``edge`` below is a local
		# coordinate measured from the first corner of each oriented side.
		half_side = side / 2
		left = center_x - half_side
		right = center_x + half_side
		bottom = center_y - half_side
		top = center_y + half_side
		# Omitting each edge endpoint prevents duplicate corners: the following
		# edge supplies that corner, and polygon closure supplies the final one.
		edge = np.linspace(0.0, side, int(points_per_side), endpoint=False)
		x = np.concatenate(
			(
				left + edge,
				np.full_like(edge, right),
				right - edge,
				np.full_like(edge, left),
			)
		)
		y = np.concatenate(
			(
				np.full_like(edge, bottom),
				bottom + edge,
				np.full_like(edge, top),
				top - edge,
			)
		)
		return cls(np.concatenate((x, y)), shape="square", rho=rho)

	@classmethod
	def circle(
		cls,
		*,
		center: tuple[float, float],
		radius: float,
		points: int = 128,
		rho: float = 0.0,
	) -> Area:
		"""Sample a counter-clockwise circle without repeating its first point.

		``center`` and ``radius`` define the initial geometry, while ``points``
		is the total number of boundary vertices rather than a density per arc.
		"""
		center_x, center_y = cls._validate_center(center)
		radius = cls._positive_finite(radius, "radius")
		if (
			isinstance(points, (bool, np.bool_))
			or not isinstance(points, (int, np.integer))
			or points < 3
		):
			raise ValueError("`points` must be an integer of at least 3.")

		# The shoelace formula closes the polygon, so sampling 2π again would add
		# a redundant copy of the vertex at angle zero.
		angle = np.linspace(0.0, 2 * np.pi, int(points), endpoint=False)
		x = center_x + radius * np.cos(angle)
		y = center_y + radius * np.sin(angle)
		return cls(np.concatenate((x, y)), shape="circle", rho=rho)

	def calculate_area(
		self,
		state: np.ndarray | None = None,
		*,
		period: float | None = None,
	) -> np.ndarray:
		"""Calculate the signed polygon area for one state or a time series.

		When ``period`` is provided, consecutive vertices are unwrapped with the
		minimum-image convention before applying the shoelace formula; ``period``
		must therefore use the same coordinate normalization as ``x`` and ``y``.
		A single state returns a scalar array.  An input with shape
		``(2 * N, *sample_axes)`` returns an area array of shape ``sample_axes``.
		Counter-clockwise contours have positive area.
		"""
		value = self._required_state() if state is None else np.asarray(state, dtype=float)
		if not np.all(np.isfinite(value)):
			raise ValueError("The area state must contain only finite values.")
		# Here axis zero enumerates successive boundary vertices; any trailing
		# axes are independent contour snapshots, commonly integration times.
		x, y = self.positions(value)
		if x.shape[0] < 3:
			raise ValueError("An area boundary requires at least three points.")
		# Periodic unwrapping edits coordinates in place.  Copies protect both the
		# stored initial condition and integration results supplied by callers.
		x = np.asarray(x, dtype=float).copy()
		y = np.asarray(y, dtype=float).copy()

		if period is not None:
			period = self._positive_finite(period, "period")
			# Reconstruct a locally continuous polygon independently at every time
			# sample, choosing the nearest periodic image of each next vertex.
			for vertex in range(1, x.shape[0]):
				delta_x = x[vertex] - x[vertex - 1]
				delta_y = y[vertex] - y[vertex - 1]
				delta_x -= period * np.round(delta_x / period)
				delta_y -= period * np.round(delta_y / period)
				x[vertex] = x[vertex - 1] + delta_x
				y[vertex] = y[vertex - 1] + delta_y

		# The rolled arrays pair every vertex with its successor and implicitly
		# include the closing edge from the final point to the first.
		return np.asarray(
			0.5
			* np.sum(
				x * np.roll(y, -1, axis=0) - y * np.roll(x, -1, axis=0),
				axis=0,
			)
		)

	def _required_state(self) -> np.ndarray:
		"""Return the boundary state guaranteed by the required constructor input."""
		state = self.state
		assert state is not None
		return state

	@staticmethod
	def _validate_center(center: tuple[float, float]) -> tuple[float, float]:
		"""Normalize a finite two-dimensional centre."""
		try:
			center_x, center_y = center
		except (TypeError, ValueError) as exc:
			raise ValueError("`center` must contain exactly two coordinates.") from exc
		values = np.asarray((center_x, center_y), dtype=float)
		if not np.all(np.isfinite(values)):
			raise ValueError("`center` must contain finite coordinates.")
		return float(values[0]), float(values[1])

	@staticmethod
	def _positive_finite(value: float, name: str) -> float:
		"""Normalize a scalar geometric parameter and enforce positivity."""
		try:
			result = float(value)
		except (TypeError, ValueError) as exc:
			raise ValueError(f"`{name}` must be positive and finite.") from exc
		if not np.isfinite(result) or result <= 0:
			raise ValueError(f"`{name}` must be positive and finite.")
		return result


__all__ = ["Area"]
