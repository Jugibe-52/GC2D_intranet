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
	"""Guiding-centre dynamics over a gyroaveraged potential.

	The trajectory state is component-major ``[x, y]``. Its ``rho`` determines
	the fixed-radius gyroaverage stored as ``effective_potential``; all GC forces,
	energies, and visualizations use that effective field rather than the raw one.
	"""

	trajectory: TrajectoryGC

	def __init__(self, potential: Potential, trajectory: TrajectoryGC) -> None:
		"""Construct the GC system and precompute its effective potential."""
		if not isinstance(trajectory, TrajectoryGC):
			raise TypeError("SystemGC requires a TrajectoryGC instance.")
		super().__init__(potential, trajectory)
		# Gyroaveraging depends only on the fixed Larmor radius, so doing it once
		# avoids rebuilding the effective field during every vector evaluation.
		self.effective_potential = potential.gyroaverage(trajectory.rho)

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Evaluate the GC drift at all positions in a flat ``[x, y]`` state.

		``ex`` and ``ey`` have one value per represented particle, and the packed
		result preserves exactly the input state's component-major shape.
		"""
		components = self.trajectory.split(state)
		ex, ey = self.effective_potential.electric_field(
			t,
			components.x,
			components.y,
		)
		# The GC Poisson structure rotates the electric field clockwise.
		return self.trajectory.pack(ey, -ex)

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate the gyroaveraged Hamiltonian per particle and saved time.

		A scalar ``t`` normally accompanies a flat state; a time vector broadcasts
		against a solution whose trailing axis stores those same saved times.
		"""
		components = self.trajectory.split(state)
		return self.effective_potential.evaluate(t, components.x, components.y)

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate the per-particle derivative of momentum conjugate to time."""
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
		"""Animate an Area over the effective field and its relative error.

		The relative error is measured against the initial signed area and the
		requested frames are sampled uniformly from the saved solution. ``frames``
		limits displayed samples without reintegration, and ``interval`` is the
		delay between displayed frames in milliseconds.
		"""
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
		"""Delegate GC-specific state expansion and BM4 flows to the integrator."""
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
