"""Spatial grid model shared by potential-based simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts import Array


@dataclass(frozen=True, slots=True, eq=False)
class Grid:
	"""Validated regular two-dimensional Cartesian grid.

	The grid stores only the origin, spacing, and number of points of each axis.
	Coordinate arrays are generated on demand. ``period`` applies to both axes;
	``None`` selects clipped, non-periodic boundaries.
	"""

	x0: float
	y0: float
	dx: float
	dy: float
	nx: int
	ny: int
	period: float | None = None

	def __post_init__(self) -> None:
		for name in ("x0", "y0", "dx", "dy"):
			value = float(getattr(self, name))
			if not np.isfinite(value):
				raise ValueError(f"`{name}` must be finite.")
			object.__setattr__(self, name, value)
		if self.dx <= 0 or self.dy <= 0:
			raise ValueError("`dx` and `dy` must be positive.")

		for name in ("nx", "ny"):
			value = getattr(self, name)
			if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
				raise TypeError(f"`{name}` must be an integer.")
			if value < 2:
				raise ValueError(f"`{name}` must be at least 2.")
			object.__setattr__(self, name, int(value))

		if self.period is not None:
			period = float(self.period)
			if not np.isfinite(period) or period <= 0:
				raise ValueError("`period` must be a positive finite number.")
			if not np.isclose(self.nx * self.dx, period):
				raise ValueError("`period` must equal `nx * dx` for a periodic x-axis.")
			if not np.isclose(self.ny * self.dy, period):
				raise ValueError("`period` must equal `ny * dy` for a periodic y-axis.")
			object.__setattr__(self, "period", period)

	@classmethod
	def from_axes(cls, x: Array, y: Array, *, period: float | None = None) -> Grid:
		"""Build a parametric grid from regular coordinate arrays.

		This is the input adapter for data formats such as HDF5 that store complete
		coordinate axes. The axes are validated, then reduced to origin, spacing,
		and size.
		"""
		x0, dx, nx = cls._axis_parameters(x, "x")
		y0, dy, ny = cls._axis_parameters(y, "y")
		return cls(x0=x0, y0=y0, dx=dx, dy=dy, nx=nx, ny=ny, period=period)

	@classmethod
	def from_bounds(
		cls,
		x_start: float,
		x_stop: float,
		y_start: float,
		y_stop: float,
		nx: int,
		ny: int,
		*,
		periodic: bool = False,
	) -> Grid:
		"""Build a grid from spatial bounds and the number of points."""
		if nx < 2 or ny < 2:
			raise ValueError("`nx` and `ny` must be at least 2.")
		x_span = float(x_stop) - float(x_start)
		y_span = float(y_stop) - float(y_start)
		if x_span <= 0 or y_span <= 0:
			raise ValueError("Grid stop coordinates must be greater than start coordinates.")
		if periodic:
			if not np.isclose(x_span, y_span):
				raise ValueError("Periodic x and y bounds must define the same period.")
			return cls(x_start, y_start, x_span / nx, y_span / ny, nx, ny, period=x_span)
		return cls(x_start, y_start, x_span / (nx - 1), y_span / (ny - 1), nx, ny)

	@staticmethod
	def _axis_parameters(values: Array, name: str) -> tuple[float, float, int]:
		axis = np.asarray(values, dtype=float)
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
		return float(axis[0]), float(steps[0]), int(axis.size)

	@staticmethod
	def _axis(origin: float, spacing: float, size: int) -> Array:
		axis = origin + spacing * np.arange(size, dtype=float)
		axis.setflags(write=False)
		return axis

	@property
	def x(self) -> Array:
		return self._axis(self.x0, self.dx, self.nx)

	@property
	def y(self) -> Array:
		return self._axis(self.y0, self.dy, self.ny)

	@property
	def shape(self) -> tuple[int, int]:
		return self.nx, self.ny

	@property
	def xmin(self) -> float:
		return self.x0

	@property
	def xmax(self) -> float:
		return self.x0 + (self.nx - 1) * self.dx

	@property
	def ymin(self) -> float:
		return self.y0

	@property
	def ymax(self) -> float:
		return self.y0 + (self.ny - 1) * self.dy

	def resized(self, nx: int | None = None, ny: int | None = None) -> Grid:
		"""Return a grid with new axis sizes and the same spatial domain."""
		target_nx = self.nx if nx is None else nx
		target_ny = self.ny if ny is None else ny
		if target_nx < 2 or target_ny < 2:
			raise ValueError("`nx` and `ny` must be at least 2.")
		if target_nx == self.nx and target_ny == self.ny:
			return self
		if self.period is not None:
			return Grid(
				self.x0,
				self.y0,
				self.period / target_nx,
				self.period / target_ny,
				target_nx,
				target_ny,
				period=self.period,
			)
		return Grid(
			self.x0,
			self.y0,
			(self.xmax - self.xmin) / (target_nx - 1),
			(self.ymax - self.ymin) / (target_ny - 1),
			target_nx,
			target_ny,
		)

	def wrap_or_clip(self, x: Array, y: Array) -> tuple[Array, Array]:
		"""Apply the grid boundary policy to paired evaluation coordinates."""
		if self.period is None:
			x = np.clip(x, self.xmin, self.xmax)
			y = np.clip(y, self.ymin, self.ymax)
		else:
			x = ((np.asarray(x) - self.xmin) % self.period) + self.xmin
			y = ((np.asarray(y) - self.ymin) % self.period) + self.ymin
		return np.asarray(x), np.asarray(y)
