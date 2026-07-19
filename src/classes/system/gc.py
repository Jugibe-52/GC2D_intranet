"""Guiding-centre system."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.animation import FuncAnimation

from classes.potential import Potential
from classes.trajectory import Area, TrajectoryGC

from ._integration import solve_gc
from ._visualization import animate_gc_area
from .solution import Solution
from .system import System


class SystemGC(System):
	"""Guiding-centre dynamics over a gyroaveraged potential."""

	trajectory: TrajectoryGC

	def __init__(self, potential: Potential, trajectory: TrajectoryGC) -> None:
		if not isinstance(trajectory, TrajectoryGC):
			raise TypeError("SystemGC requires a TrajectoryGC instance.")
		super().__init__(potential, trajectory)
		self.effective_potential = potential.gyroaverage(trajectory.rho)

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		components = self.trajectory.split(state)
		ex, ey = self.effective_potential.electric_field(
			t,
			components.x,
			components.y,
		)
		return self.trajectory.pack(ey, -ex)

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		components = self.trajectory.split(state)
		return self.effective_potential.evaluate(t, components.x, components.y)

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		components = self.trajectory.split(state)
		return -self.effective_potential.evaluate(
			t,
			components.x,
			components.y,
			dt=1,
		)

	def animate_area(
		self,
		solution: Solution,
		*,
		frames: int | None = 120,
		interval: int = 50,
		cmap: str = "RdBu_r",
		repeat: bool = True,
		**pcolormesh_kwargs: Any,
	) -> FuncAnimation:
		"""Animate an area over the effective potential and its relative error."""
		if not isinstance(self.trajectory, Area):
			raise TypeError("`animate_area` requires an Area trajectory.")
		if not isinstance(solution, Solution):
			raise TypeError("`solution` must be a Solution instance.")
		return animate_gc_area(
			self.effective_potential,
			self.trajectory,
			solution,
			frames=frames,
			interval=interval,
			cmap=cmap,
			repeat=repeat,
			pcolormesh_kwargs=pcolormesh_kwargs,
		)

	def _integrate(
		self,
		state: np.ndarray,
		*,
		step: float,
		t_span: tuple[float, float],
		n_save_step: int,
		check_energy: bool,
		progress: bool,
	) -> Solution:
		return solve_gc(
			self,
			state,
			step=step,
			t_span=t_span,
			n_save_step=n_save_step,
			check_energy=check_energy,
			progress=progress,
		)


__all__ = ["SystemGC"]
