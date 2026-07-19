"""Full-cyclotron system composed from a potential and trajectory."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from contracts import Array
from classes.potential import Potential
from classes.trajectory import TrajectoryFC

from .system import System

if TYPE_CHECKING:
	from .solution import Solution


def _real_imag(value: Array) -> tuple[Array, Array]:
	return np.asarray(value.real), np.asarray(value.imag)


class SystemFC(System):
	"""Full-cyclotron dynamics over the physical, ungyroaveraged potential."""

	trajectory: TrajectoryFC

	def __init__(self, potential: Potential, trajectory: TrajectoryFC) -> None:
		if not isinstance(trajectory, TrajectoryFC):
			raise TypeError("SystemFC requires a TrajectoryFC instance.")
		super().__init__(potential, trajectory)

	@property
	def velocity_scale(self) -> float:
		return self.trajectory.velocity_scale

	@property
	def electric_scale(self) -> float:
		return self.trajectory.electric_scale

	@property
	def larmor_frequency(self) -> float:
		return self.trajectory.larmor_frequency

	def _build_effective_potential(self) -> Potential:
		return self.potential.copy()

	def reduced_vector_field(self, t: float, state: Array) -> Array:
		state_array = np.asarray(state)
		if state_array.ndim == 0 or state_array.shape[0] % 2 != 0:
			raise ValueError("A reduced FC state must contain equally sized x and y blocks.")
		x, y = np.split(state_array, 2, axis=0)
		ex, ey = self.electric_field(t, x, y, effective=False)
		return np.concatenate((ey, -ex))

	def vector_field(self, t: float, state: Array) -> Array:
		x, y = self.get_positions(state)
		vx, vy = self.trajectory.get_velocities(state)
		ex, ey = self.electric_field(t, x, y, effective=False)
		return np.concatenate((
			vx * self.velocity_scale,
			vy * self.velocity_scale,
			ex * self.electric_scale + vy * self.larmor_frequency,
			ey * self.electric_scale - vx * self.larmor_frequency,
		))

	def hamiltonian(self, t: float | Array, state: Array) -> Array:
		x, y = self.get_positions(state)
		vx, vy = self.trajectory.get_velocities(state)
		return np.asarray(
			self.trajectory.rho / (4 * abs(self.trajectory.eta)) * (vx**2 + vy**2)
			+ self.phi(t, x, y) * self.electric_scale
		)

	def extended_momentum_derivative(self, t: float, state: Array) -> Array:
		x, y = self.get_positions(state)
		return np.asarray(-self.electric_scale * self.phi(t, x, y, dt=1))

	def _split_flow_state(self, state: Array, check_energy: bool) -> tuple[Array, ...]:
		components = 5 if check_energy else 4
		state_array = np.asarray(state)
		if state_array.ndim == 0 or state_array.shape[0] % components != 0:
			raise ValueError(
				f"The FC flow state must contain {components} equally sized blocks."
			)
		return tuple(np.split(state_array, components, axis=0))

	def flow(self, h: float, t: float, state: Array, *, check_energy: bool = False) -> Array:
		"""Apply the explicit full-cyclotron splitting flow."""
		components = self._split_flow_state(state, check_energy)
		x, y, vx, vy = components[:4]
		k = components[4].copy() if check_energy else None
		rotation = np.exp(-1j * h * self.larmor_frequency)
		x, y = _real_imag(
			x + 1j * y
			+ 1j * self.trajectory.rho * np.sign(self.trajectory.eta)
			* (rotation - 1) * (vx + 1j * vy)
		)
		vx, vy = _real_imag(rotation * (vx + 1j * vy))
		force_x, force_y = np.split(
			self.reduced_vector_field(t, np.concatenate((x, y))),
			2,
		)
		vx, vy = _real_imag(
			vx + 1j * vy
			+ h * 1j * (force_x + 1j * force_y) * self.electric_scale
		)
		result = (x, y, vx, vy)
		if k is None:
			return np.concatenate(result)
		k += h * self.extended_momentum_derivative(t, np.concatenate(result))
		return np.concatenate((*result, k))

	def adjoint_flow(
		self,
		h: float,
		t: float,
		state: Array,
		*,
		check_energy: bool = False,
	) -> Array:
		"""Apply the adjoint full-cyclotron splitting flow."""
		components = self._split_flow_state(state, check_energy)
		x, y, vx, vy = components[:4]
		k = components[4].copy() if check_energy else None
		force_x, force_y = np.split(
			self.reduced_vector_field(t, np.concatenate((x, y))),
			2,
		)
		vx, vy = _real_imag(
			vx + 1j * vy
			+ h * 1j * (force_x + 1j * force_y) * self.electric_scale
		)
		if k is not None:
			k += h * self.extended_momentum_derivative(t, np.concatenate((x, y, vx, vy)))
		rotation = np.exp(-1j * h * self.larmor_frequency)
		x, y = _real_imag(
			x + 1j * y
			+ 1j * self.trajectory.rho * np.sign(self.trajectory.eta)
			* (rotation - 1) * (vx + 1j * vy)
		)
		vx, vy = _real_imag(rotation * (vx + 1j * vy))
		result = (x, y, vx, vy)
		return np.concatenate(result if k is None else (*result, k))

	def chi(self, h: float, t: float, state: Array) -> Array:
		return self.flow(h, t, state)

	def chi_star(self, h: float, t: float, state: Array) -> Array:
		return self.adjoint_flow(h, t, state)

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
		"""Integrate FC by composing its explicit flow and adjoint flow."""
		from .fc_solver import solve_symplectic

		del progress  # The explicit FC path has no progress renderer.
		n_trajectories = state.shape[0] // self.trajectory.state_dimension
		integrator_state = state
		if check_energy:
			integrator_state = np.concatenate((state, np.zeros(n_trajectories)))
		solution = solve_symplectic(
			lambda h, t, value: self.flow(h, t, value, check_energy=check_energy),
			lambda h, t, value: self.adjoint_flow(
				h,
				t,
				value,
				check_energy=check_energy,
			),
			t_span,
			integrator_state,
			step,
			t_eval,
			method,
			command,
			save_step=save_step,
			n_save_step=n_save_step,
		)
		if check_energy:
			components = np.split(solution.y, 5, axis=0)
			solution.y = np.concatenate(components[:4], axis=0)
			solution.k = components[4]
			solution.err = self.compute_energy(solution)
		return solution


__all__ = ["SystemFC"]
