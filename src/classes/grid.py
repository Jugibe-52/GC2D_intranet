"""Spatial grid model shared by potential-based simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts import Array


@dataclass(frozen=True, slots=True, eq=False)
class Grid:
	"""Validated two-dimensional Cartesian grid.

	The coordinate axes are copied and made read-only so the dimensions, spacing,
	and bounds cannot become inconsistent after construction. ``period`` applies
	to both axes; ``None`` selects clipped, non-periodic boundaries.
	"""

	x: Array
	y: Array
	period: float | None = None

	def __post_init__(self) -> None:
		x = self._validated_axis(self.x, "x")
		y = self._validated_axis(self.y, "y")
		if self.period is not None:
			period = float(self.period)
			if not np.isfinite(period) or period <= 0:
				raise ValueError("`period` must be a positive finite number.")
			if not np.isclose(x.size * (x[1] - x[0]), period):
				raise ValueError("`period` must equal `nx * dx` for a periodic x-axis.")
			if not np.isclose(y.size * (y[1] - y[0]), period):
				raise ValueError("`period` must equal `ny * dy` for a periodic y-axis.")
			object.__setattr__(self, "period", period)
		x.setflags(write=False)
		y.setflags(write=False)
		object.__setattr__(self, "x", x)
		object.__setattr__(self, "y", y)

	@staticmethod
	def _validated_axis(values: Array, name: str) -> Array:
		axis = np.asarray(values, dtype=float).copy()
		if axis.ndim != 1:
			raise ValueError(f"`{name}` must be 1-dimensional.")
		if axis.size < 2:
			raise ValueError(f"`{name}` must contain at least two points.")
		if not np.all(np.isfinite(axis)):
			raise ValueError(f"`{name}` must contain only finite values.")
		steps = np.diff(axis)
		if np.any(steps <= 0):
			raise ValueError(f"Values in `{name}` must be strictly increasing.")
		if not np.allclose(steps, steps[0]):
			raise ValueError(f"Values in `{name}` must be uniformly spaced.")
		return axis

	@property
	def shape(self) -> tuple[int, int]:
		return self.nx, self.ny

	@property
	def nx(self) -> int:
		return int(self.x.size)

	@property
	def ny(self) -> int:
		return int(self.y.size)

	@property
	def dx(self) -> float:
		return float(self.x[1] - self.x[0])

	@property
	def dy(self) -> float:
		return float(self.y[1] - self.y[0])

	@property
	def xmin(self) -> float:
		return float(self.x[0])

	@property
	def xmax(self) -> float:
		return float(self.x[-1])

	@property
	def ymin(self) -> float:
		return float(self.y[0])

	@property
	def ymax(self) -> float:
		return float(self.y[-1])

	def resized(self, nx: int | None = None, ny: int | None = None) -> Grid:
		"""Return a grid with new axis sizes and the same spatial domain."""
		target_nx = self.nx if nx is None else nx
		target_ny = self.ny if ny is None else ny
		if target_nx < 2 or target_ny < 2:
			raise ValueError("`nx` and `ny` must be at least 2.")
		if self.period is None:
			x = np.linspace(self.xmin, self.xmax, target_nx)
			y = np.linspace(self.ymin, self.ymax, target_ny)
		else:
			x = np.linspace(self.xmin, self.xmin + self.period, target_nx, endpoint=False)
			y = np.linspace(self.ymin, self.ymin + self.period, target_ny, endpoint=False)
		return Grid(x, y, period=self.period)

	def wrap_or_clip(self, x: Array, y: Array) -> tuple[Array, Array]:
		"""Apply the grid boundary policy to paired evaluation coordinates."""
		if self.period is None:
			x = np.clip(x, self.xmin, self.xmax)
			y = np.clip(y, self.ymin, self.ymax)
		else:
			x = ((np.asarray(x) - self.xmin) % self.period) + self.xmin
			y = ((np.asarray(y) - self.ymin) % self.period) + self.ymin
		return np.asarray(x), np.asarray(y)
