"""Guiding-centre system composed from a potential and trajectory."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from contracts import Array
from classes.potential import GridPotential, Potential
from classes.trajectory import TrajectoryGC

from .system import System

if TYPE_CHECKING:
	from .solution import Solution


class SystemGC(System):
	"""Guiding-centre dynamics over an effective gyroaveraged potential."""

	trajectory: TrajectoryGC

	def __init__(self, potential: Potential, trajectory: TrajectoryGC) -> None:
		if not isinstance(trajectory, TrajectoryGC):
			raise TypeError("SystemGC requires a TrajectoryGC instance.")
		super().__init__(potential, trajectory)

	def _build_effective_potential(self) -> Potential:
		if isinstance(self.potential, GridPotential):
			resolution = min(
				self.potential.kinterp * self.potential.grid.dx,
				self.potential.kinterp * self.potential.grid.dy,
			)
			if resolution < self.trajectory.rho:
				raise ValueError(
					f"Interpolation order {self.potential.kinterp} is too low for "
					f"rho={self.trajectory.rho}. Increase k or decrease rho."
				)
		return self.potential.gyroaveraged(self.trajectory.rho)

	def vector_field(self, t: float, state: Array) -> Array:
		x, y = self.get_positions(state)
		ex, ey = self.electric_field(t, x, y)
		return np.concatenate((ey, -ex))

	def hamiltonian(self, t: float | Array, state: Array) -> Array:
		x, y = self.get_positions(state)
		return np.asarray(self.psi(t, x, y))

	def extended_momentum_derivative(self, t: float, state: Array) -> Array:
		x, y = self.get_positions(state)
		return np.asarray(-self.psi(t, x, y, dt=1))

	def _integrate(
		self,
		state: Array,
		*,
		t_span: tuple[float, float],
		step: float,
		t_eval: Array | None,
		save_step: float | None,
		n_save_step: int | None,
		method: str,
		check_energy: bool,
		command: Callable[[float, Array], None] | None,
		progress: bool,
	) -> Solution:
		"""Integrate GC through the non-separable extended-phase-space path."""
		from .gc_solver import solve_extended

		return solve_extended(
			self,
			t_span,
			state,
			step,
			t_eval,
			method,
			command=command,
			check_energy=check_energy,
			save_step=save_step,
			n_save_step=n_save_step,
			progress=progress,
		)


__all__ = ["SystemGC"]
