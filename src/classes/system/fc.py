"""Full-cyclotron system."""

from __future__ import annotations

import numpy as np

from classes.potential import Potential
from classes.trajectory import TrajectoryFC

from ._integration import solve_fc
from .solution import Solution
from .system import System


def _real_imaginary(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	return np.asarray(value.real), np.asarray(value.imag)


class SystemFC(System):
	"""Full-cyclotron dynamics over the physical potential."""

	trajectory: TrajectoryFC

	def __init__(self, potential: Potential, trajectory: TrajectoryFC) -> None:
		if not isinstance(trajectory, TrajectoryFC):
			raise TypeError("SystemFC requires a TrajectoryFC instance.")
		super().__init__(potential, trajectory)

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		x, y = self.trajectory.positions(state)
		vx, vy = self.trajectory.velocities(state)
		ex, ey = self.potential.electric_field(t, x, y)
		return np.concatenate(
			(
				vx * self.trajectory.velocity_scale,
				vy * self.trajectory.velocity_scale,
				ex * self.trajectory.electric_scale
				+ vy * self.trajectory.larmor_frequency,
				ey * self.trajectory.electric_scale
				- vx * self.trajectory.larmor_frequency,
			)
		)

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		x, y = self.trajectory.positions(state)
		vx, vy = self.trajectory.velocities(state)
		kinetic_scale = self.trajectory.rho / (4 * abs(self.trajectory.eta))
		return np.asarray(
			kinetic_scale * (vx**2 + vy**2)
			+ self.trajectory.electric_scale * self.potential.evaluate(t, x, y)
		)

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		x, y = self.trajectory.positions(state)
		return np.asarray(
			-self.trajectory.electric_scale * self.potential.evaluate(t, x, y, dt=1)
		)

	def _reduced_vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		value = np.asarray(state)
		if value.ndim == 0 or value.shape[0] % 2:
			raise ValueError("A reduced FC state requires equally sized x and y blocks.")
		x, y = np.split(value, 2, axis=0)
		ex, ey = self.potential.electric_field(t, x, y)
		return np.concatenate((ey, -ex))

	@staticmethod
	def _split_flow_state(
		state: np.ndarray,
		check_energy: bool,
	) -> tuple[np.ndarray, ...]:
		component_count = 5 if check_energy else 4
		value = np.asarray(state)
		if value.ndim == 0 or value.shape[0] % component_count:
			raise ValueError(
				f"The FC flow state requires {component_count} equally sized blocks."
			)
		return tuple(np.split(value, component_count, axis=0))

	def _flow(
		self,
		h: float,
		t: float,
		state: np.ndarray,
		*,
		check_energy: bool,
	) -> np.ndarray:
		components = self._split_flow_state(state, check_energy)
		x, y, vx, vy = components[:4]
		momentum = components[4].copy() if check_energy else None
		rotation = np.exp(-1j * h * self.trajectory.larmor_frequency)
		x, y = _real_imaginary(
			x
			+ 1j * y
			+ 1j
			* self.trajectory.rho
			* np.sign(self.trajectory.eta)
			* (rotation - 1)
			* (vx + 1j * vy)
		)
		vx, vy = _real_imaginary(rotation * (vx + 1j * vy))
		force_x, force_y = np.split(
			self._reduced_vector_field(t, np.concatenate((x, y))),
			2,
		)
		vx, vy = _real_imaginary(
			vx
			+ 1j * vy
			+ h
			* 1j
			* (force_x + 1j * force_y)
			* self.trajectory.electric_scale
		)
		physical_state = (x, y, vx, vy)
		if momentum is None:
			return np.concatenate(physical_state)
		momentum += h * self.extended_momentum_derivative(
			t,
			np.concatenate(physical_state),
		)
		return np.concatenate((*physical_state, momentum))

	def _adjoint_flow(
		self,
		h: float,
		t: float,
		state: np.ndarray,
		*,
		check_energy: bool,
	) -> np.ndarray:
		components = self._split_flow_state(state, check_energy)
		x, y, vx, vy = components[:4]
		momentum = components[4].copy() if check_energy else None
		force_x, force_y = np.split(
			self._reduced_vector_field(t, np.concatenate((x, y))),
			2,
		)
		vx, vy = _real_imaginary(
			vx
			+ 1j * vy
			+ h
			* 1j
			* (force_x + 1j * force_y)
			* self.trajectory.electric_scale
		)
		if momentum is not None:
			momentum += h * self.extended_momentum_derivative(
				t,
				np.concatenate((x, y, vx, vy)),
			)
		rotation = np.exp(-1j * h * self.trajectory.larmor_frequency)
		x, y = _real_imaginary(
			x
			+ 1j * y
			+ 1j
			* self.trajectory.rho
			* np.sign(self.trajectory.eta)
			* (rotation - 1)
			* (vx + 1j * vy)
		)
		vx, vy = _real_imaginary(rotation * (vx + 1j * vy))
		physical_state = (x, y, vx, vy)
		return np.concatenate(
			physical_state if momentum is None else (*physical_state, momentum)
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
