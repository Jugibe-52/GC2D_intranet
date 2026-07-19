"""Guiding-centre system."""

from __future__ import annotations

import numpy as np

from classes.potential import Potential
from classes.trajectory import TrajectoryGC

from ._integration import solve_gc
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
		x, y = self.trajectory.positions(state)
		ex, ey = self.effective_potential.electric_field(t, x, y)
		return np.concatenate((ey, -ex))

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		x, y = self.trajectory.positions(state)
		return self.effective_potential.evaluate(t, x, y)

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		x, y = self.trajectory.positions(state)
		return -self.effective_potential.evaluate(t, x, y, dt=1)

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
