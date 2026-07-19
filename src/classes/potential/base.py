"""Common contract for physical potential representations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from contracts import Array

from .grid import Grid


class Potential(ABC):
	"""A physical scalar potential independent from any trajectory model.

	Concrete potentials decide how the field is represented (for example by a
	grid of spline coefficients or by an analytic Fourier series).  They expose
	the same evaluation contract so a :class:`classes.system.System` can combine
	them with either a guiding-centre or a full-cyclotron trajectory.
	"""

	grid: Grid

	@abstractmethod
	def field_at_time(
		self,
		t: float | Array,
		x: Array | None = None,
		y: Array | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> Array:
		"""Evaluate the field or one of its derivatives."""

	def evaluate(
		self,
		t: float | Array,
		x: Array | None = None,
		y: Array | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> Array:
		"""Readable alias for :meth:`field_at_time`."""
		return self.field_at_time(t, x, y, dx=dx, dy=dy, dt=dt)

	def electric_field(
		self,
		t: float | Array,
		x: Array | None = None,
		y: Array | None = None,
	) -> tuple[Array, Array]:
		"""Return the electric field :math:`-\nabla\phi`."""
		if x is None and y is None:
			x, y = np.meshgrid(self.grid.x, self.grid.y, indexing="ij")
		elif x is None or y is None:
			raise ValueError("`x` and `y` must be provided together.")
		return (
			-self.field_at_time(t, x, y, dx=1),
			-self.field_at_time(t, x, y, dy=1),
		)

	@abstractmethod
	def gyroaveraged(self, rho: float) -> Potential:
		"""Return the effective potential averaged at Larmor radius ``rho``."""

	@abstractmethod
	def copy(self) -> Potential:
		"""Return an independent potential with identical physical contents."""
