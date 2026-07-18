# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Common behaviour for trajectories over interpolated potentials."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from pyhamsys import HamSys

from contracts import TrajectoryKind, TrajectoryParams
from ..grid import Grid
from ..potential import Array, Potential, PotentialFields, PotentialInterpolators


class Trajectory(HamSys, ABC):
	r"""Base class shared by guiding-centre and full-cyclotron trajectories.

	The class owns the physical potential :math:`\phi`, builds the effective
	gyroaveraged potential :math:`\psi`, and provides operations which do not
	depend on the representation of the trajectory state.  Concrete subclasses
	contain the equations of motion.
	"""

	kind: TrajectoryKind
	_state_dimension: int

	def __init__(self, potential: Potential, params: TrajectoryParams, *, ndof: float) -> None:
		if params.get("type") != self.kind:
			raise ValueError(
				f"{self.__class__.__name__} requires trajectory type {self.kind!r}, "
				f"got {params.get('type')!r}."
			)
		super().__init__(ndof=ndof)
		self.traj = params.copy()
		self.rho = float(params.get("rho", 0.0))
		self.eta = float(params.get("eta", 0.0))
		if not np.isfinite(self.rho) or self.rho < 0:
			raise ValueError("`rho` must be a finite, non-negative number.")
		if not np.isfinite(self.eta):
			raise ValueError("`eta` must be finite.")

		# Keep phi and psi independent: psi may be gyroaveraged while phi must
		# remain the physical potential used by full-cyclotron dynamics.
		self.physical_potential = Potential(
			grid=potential.grid,
			fields=potential.fields.copy(),
			k=potential.kinterp,
		)
		if min(potential.kinterp * potential.grid.dx, potential.kinterp * potential.grid.dy) < self.rho:
			raise ValueError(
				f"Interpolation order {potential.kinterp} is too low for rho = {self.rho}. "
				"Increase k or decrease rho."
			)
		self.effective_potential = (
			self.physical_potential.gyroaveraged(self.rho)
			if self.rho != 0
			else Potential(
				grid=potential.grid,
				fields=potential.fields.copy(),
				k=potential.kinterp,
			)
		)

	def __str__(self) -> str:
		return f"2D {self.__class__.__name__} for turbulent potentials"

	@property
	def grid(self) -> Grid:
		r"""Spatial grid shared by :math:`\phi` and :math:`\psi`."""
		return self.effective_potential.grid

	@property
	def fields(self) -> PotentialFields:
		r"""Fields of the effective potential :math:`\psi`."""
		return self.effective_potential.fields

	@property
	def interpolators(self) -> PotentialInterpolators:
		r"""Interpolators of the effective potential :math:`\psi`."""
		return self.effective_potential.interpolators

	@property
	def kinterp(self) -> int:
		r"""Spline order of the effective potential :math:`\psi`."""
		return self.effective_potential.kinterp

	def _split_state(self, state: Array) -> tuple[Array, ...]:
		"""Split a state into equally sized model-specific components."""
		state_array = np.asarray(state)
		if state_array.shape[0] % self._state_dimension != 0:
			raise ValueError(
				f"The first state dimension must be divisible by {self._state_dimension} "
				f"for {self.__class__.__name__}."
			)
		return tuple(np.split(state_array, self._state_dimension, axis=0))

	def get_positions(self, state: Array) -> tuple[Array, Array]:
		"""Return the x and y blocks stored in ``state``."""
		x, y, *_ = self._split_state(state)
		return x, y

	@abstractmethod
	def get_velocities(self, state: Array) -> tuple[Array, Array] | None:
		"""Return velocity blocks, or ``None`` if the model has none."""

	def psi(
		self,
		t: float | Array,
		x: Array | None = None,
		y: Array | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> Array:
		"""Evaluate the effective, possibly gyroaveraged potential."""
		return self.effective_potential.field_at_time(t, x, y, dx=dx, dy=dy, dt=dt)

	def phi(
		self,
		t: float | Array,
		x: Array | None = None,
		y: Array | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> Array:
		"""Evaluate the physical, ungyroaveraged potential."""
		return self.physical_potential.field_at_time(t, x, y, dx=dx, dy=dy, dt=dt)

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
		r"""Evaluate :math:`\psi`; retained as a convenience alias."""
		return self.psi(t, x, y, dx=dx, dy=dy, dt=dt)

	def electric_field(
		self,
		t: float | Array,
		x: Array | None = None,
		y: Array | None = None,
		*,
		effective: bool = True,
	) -> tuple[Array, Array]:
		r"""Return :math:`-\nabla\psi` or the physical field :math:`-\nabla\phi`."""
		if x is None and y is None:
			x, y = np.meshgrid(self.grid.x, self.grid.y, indexing="ij")
		elif x is None or y is None:
			raise ValueError("`x` and `y` must be provided together.")
		potential = self.psi if effective else self.phi
		return -potential(t, x, y, dx=1), -potential(t, x, y, dy=1)

	@abstractmethod
	def hamiltonian(self, t: float | Array, state: Array) -> Array:  # type: ignore[override]
		"""Evaluate the model Hamiltonian."""

	@abstractmethod
	def y_dot(  # type: ignore[override]
		self,
		t: float,
		state: Array,
		output: Literal["full", "reduced"] = "full",
	) -> Array:
		"""Evaluate the equations of motion."""

	@abstractmethod
	def k_dot(self, t: float, state: Array) -> float | Array:  # type: ignore[override]
		"""Evaluate the derivative of the extended momentum."""
