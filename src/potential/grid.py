"""Regular periodic grids used to sample electrostatic potentials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Grid:
	"""A validated two-dimensional grid with a shared period on both axes.

	Periodic grids omit the duplicated upper endpoint.  That convention makes
	the FFT and the interpolation use exactly the same independent samples.

	``x0`` and ``y0`` locate the lower corner of the fundamental cell; ``dx``
	and ``dy`` are its sampling intervals.  ``nx`` and ``ny`` count independent
	samples along x and y, so every grid-shaped array follows the convention
	``(nx, ny)``.  ``period`` is expressed in the same coordinate units and is
	shared by both directions.
	"""

	x0: float
	y0: float
	dx: float
	dy: float
	nx: int
	ny: int
	period: float

	def __post_init__(self) -> None:
		"""Validate and normalize the values stored by the frozen dataclass."""
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
		# A single period is used to wrap both coordinates, so each sampled axis
		# must span that period once its omitted endpoint is restored.
		if not np.isclose(self.nx * self.dx, period):
			raise ValueError("A periodic grid requires `nx * dx == period`.")
		if not np.isclose(self.ny * self.dy, period):
			raise ValueError("A periodic grid requires `ny * dy == period`.")
		object.__setattr__(self, "period", period)

	@classmethod
	def periodic(cls, nx: int, ny: int, period: float = 2 * np.pi) -> Grid:
		"""Create a periodic domain starting at the origin.

		The returned axes contain ``nx`` and ``ny`` samples in ``[0, period)``;
		the endpoint is deliberately omitted because it represents the origin.
		"""
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
		"""Return one coordinate axis with shape ``(size,)`` and no endpoint."""
		axis = origin + spacing * np.arange(size, dtype=float)
		# Freezing the dataclass does not make returned arrays immutable by itself;
		# read-only axes keep downstream code from altering sampled coordinates.
		axis.setflags(write=False)
		return axis

	@property
	def x(self) -> np.ndarray:
		"""Sample coordinates on the x axis, with shape ``(nx,)``."""
		return self._axis(self.x0, self.dx, self.nx)

	@property
	def y(self) -> np.ndarray:
		"""Sample coordinates on the y axis, with shape ``(ny,)``."""
		return self._axis(self.y0, self.dy, self.ny)

	@property
	def shape(self) -> tuple[int, int]:
		"""Shape ``(nx, ny)`` expected for fields sampled on this grid."""
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
		"""Wrap paired coordinates into the half-open fundamental cell.

		The returned arrays preserve the respective shapes of ``x`` and ``y``;
		the operation changes their representative modulo ``period``, not their
		physical location on the periodic domain.
		"""
		x_array = np.asarray(x)
		y_array = np.asarray(y)
		return (
			((x_array - self.xmin) % self.period) + self.xmin,
			((y_array - self.ymin) % self.period) + self.ymin,
		)


__all__ = ["Grid"]
