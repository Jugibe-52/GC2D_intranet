"""Regular spatial grid used by :mod:`classes.potential`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Grid:
	"""A validated two-dimensional grid.

	Periodic grids omit the duplicated upper endpoint.  That convention makes
	the FFT and the interpolation use exactly the same samples.
	"""

	x0: float
	y0: float
	dx: float
	dy: float
	nx: int
	ny: int
	period: float

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

		period = float(self.period)
		if not np.isfinite(period) or period <= 0:
			raise ValueError("`period` must be positive and finite.")
		if not np.isclose(self.nx * self.dx, period):
			raise ValueError("A periodic grid requires `nx * dx == period`.")
		if not np.isclose(self.ny * self.dy, period):
			raise ValueError("A periodic grid requires `ny * dy == period`.")
		object.__setattr__(self, "period", period)

	@classmethod
	def periodic(cls, nx: int, ny: int, period: float = 2 * np.pi) -> Grid:
		"""Create a square periodic domain starting at the origin."""
		for size, name in ((nx, "nx"), (ny, "ny")):
			if (
				isinstance(size, (bool, np.bool_))
				or not isinstance(size, (int, np.integer))
				or size < 2
			):
				raise ValueError(f"`{name}` must be an integer of at least 2.")
		period = float(period)
		return cls(0.0, 0.0, period / nx, period / ny, nx, ny, period)

	@staticmethod
	def _axis(origin: float, spacing: float, size: int) -> np.ndarray:
		axis = origin + spacing * np.arange(size, dtype=float)
		axis.setflags(write=False)
		return axis

	@property
	def x(self) -> np.ndarray:
		return self._axis(self.x0, self.dx, self.nx)

	@property
	def y(self) -> np.ndarray:
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

	def normalize(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Wrap coordinates into the periodic domain."""
		x_array = np.asarray(x)
		y_array = np.asarray(y)
		return (
			((x_array - self.xmin) % self.period) + self.xmin,
			((y_array - self.ymin) % self.period) + self.ymin,
		)


__all__ = ["Grid"]
