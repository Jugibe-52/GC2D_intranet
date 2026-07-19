"""Full-cyclotron system."""

from __future__ import annotations

import numpy as np

from classes.potential import Potential
from classes.trajectory import TrajectoryFC

from ._integration import solve_fc
from .solution import Solution
from .system import System


class SystemFC(System):
	"""Full-cyclotron dynamics over the physical potential."""

	trajectory: TrajectoryFC

	def __init__(self, potential: Potential, trajectory: TrajectoryFC) -> None:
		if not isinstance(trajectory, TrajectoryFC):
			raise TypeError("SystemFC requires a TrajectoryFC instance.")
		super().__init__(potential, trajectory)

	def electric_acceleration(
		self,
		t: float,
		x: np.ndarray,
		y: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Return the electric contribution to ``dvx/dt`` and ``dvy/dt``."""
		ex, ey = self.potential.electric_field(t, x, y)
		return (
			self.trajectory.electric_scale * ex,
			self.trajectory.electric_scale * ey,
		)

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		components = self.trajectory.split(state)
		acceleration_x, acceleration_y = self.electric_acceleration(
			t,
			components.x,
			components.y,
		)
		return self.trajectory.pack(
			components.vx * self.trajectory.velocity_scale,
			components.vy * self.trajectory.velocity_scale,
			acceleration_x + components.vy * self.trajectory.larmor_frequency,
			acceleration_y - components.vx * self.trajectory.larmor_frequency,
		)

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		components = self.trajectory.split(state)
		kinetic_scale = self.trajectory.rho / (4 * abs(self.trajectory.eta))
		return np.asarray(
			kinetic_scale * (components.vx**2 + components.vy**2)
			+ self.trajectory.electric_scale
			* self.potential.evaluate(t, components.x, components.y)
		)

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		components = self.trajectory.split(state)
		return np.asarray(
			-self.trajectory.electric_scale
			* self.potential.evaluate(t, components.x, components.y, dt=1)
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
		return solve_fc(
			self,
			state,
			step=step,
			t_span=t_span,
			n_save_step=n_save_step,
			check_energy=check_energy,
			progress=progress,
		)


__all__ = ["SystemFC"]
