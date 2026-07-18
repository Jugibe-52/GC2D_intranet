# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Full-cyclotron trajectory dynamics."""

from typing import Literal

import numpy as np

from contracts import TrajectoryParams
from ..potential import Array, Potential, real_imag
from .trajectory import Trajectory


class TrajectoryFC(Trajectory):
	"""Full-cyclotron dynamics with state ``[x, y, vx, vy]``.

	Configuration files retain the historical value ``type='fo'`` (full orbit),
	while the class name uses ``FC`` to match the full-cyclotron model name.
	"""

	kind = "fo"
	_state_dimension = 4

	def __init__(self, potential: Potential, params: TrajectoryParams) -> None:
		super().__init__(potential, params, ndof=2.5)
		if self.rho == 0 or self.eta == 0:
			raise ValueError("TrajectoryFC requires non-zero `rho` and `eta`.")
		self.velocity_scale = self.rho / (2 * abs(self.eta))
		self.electric_scale = np.sign(self.eta) / self.rho
		self.larmor_frequency = 1 / (2 * self.eta)

		# Historical public names used by integrations and notebooks.
		self.v_fo = self.velocity_scale
		self.phi_fo = self.electric_scale
		self.omlar = self.larmor_frequency

	def get_velocities(self, state: Array) -> tuple[Array, Array]:
		_, _, vx, vy = self._split_state(state)
		return vx, vy

	def hamiltonian(self, t: float | Array, state: Array) -> Array:  # type: ignore[override]
		x, y = self.get_positions(state)
		vx, vy = self.get_velocities(state)
		return np.asarray(
			self.rho / (4 * abs(self.eta)) * (vx**2 + vy**2)
			+ self.phi(t, x, y) * np.sign(self.eta) / self.rho
		)

	def y_dot(  # type: ignore[override]
		self,
		t: float,
		state: Array,
		output: Literal["full", "reduced"] = "full",
	) -> Array:
		if output == "reduced":
			state_array = np.asarray(state)
			if state_array.shape[0] % 2 != 0:
				raise ValueError("A reduced full-cyclotron state must contain equally sized x and y blocks.")
			x, y = np.split(state_array, 2, axis=0)
			ex, ey = self.electric_field(t, x, y, effective=False)
			return np.concatenate((ey, -ex))
		if output != "full":
			raise ValueError("`output` must be either 'full' or 'reduced'.")
		x, y = self.get_positions(state)
		ex, ey = self.electric_field(t, x, y, effective=False)
		vx, vy = self.get_velocities(state)
		return np.concatenate((
			vx * self.velocity_scale,
			vy * self.velocity_scale,
			ex * self.electric_scale + vy * self.larmor_frequency,
			ey * self.electric_scale - vx * self.larmor_frequency,
		))

	def k_dot(self, t: float, state: Array) -> float:  # type: ignore[override]
		x, y = self.get_positions(state)
		return float(self.electric_scale * np.sum(self.phi(t, x, y, dt=1)))

	def chi(self, h: float, t: float, state: Array) -> Array:
		"""Apply the explicit full-cyclotron splitting flow."""
		x, y, vx, vy = self._split_state(state)
		rotation = np.exp(-1j * h * self.larmor_frequency)
		x, y = real_imag(
			x + 1j * y
			+ 1j * self.rho * np.sign(self.eta) * (rotation - 1) * (vx + 1j * vy)
		)
		vx, vy = real_imag(rotation * (vx + 1j * vy))
		force_x, force_y = np.split(self.y_dot(t, np.concatenate((x, y)), output="reduced"), 2)
		vx, vy = real_imag(
			vx + 1j * vy + h * 1j * (force_x + 1j * force_y) * self.electric_scale
		)
		return np.concatenate((x, y, vx, vy))

	def chi_star(self, h: float, t: float, state: Array) -> Array:
		"""Apply the adjoint full-cyclotron splitting flow."""
		x, y, vx, vy = self._split_state(state)
		force_x, force_y = np.split(self.y_dot(t, np.concatenate((x, y)), output="reduced"), 2)
		vx, vy = real_imag(
			vx + 1j * vy + h * 1j * (force_x + 1j * force_y) * self.electric_scale
		)
		rotation = np.exp(-1j * h * self.larmor_frequency)
		x, y = real_imag(
			x + 1j * y
			+ 1j * self.rho * np.sign(self.eta) * (rotation - 1) * (vx + 1j * vy)
		)
		vx, vy = real_imag(rotation * (vx + 1j * vy))
		return np.concatenate((x, y, vx, vy))

	def fo2gc(self, state: Array) -> tuple[Array, Array]:
		"""Convert a full-cyclotron state to guiding-centre positions."""
		x, y, vx, vy = self._split_state(state)
		velocity = vy + 1j * vx
		angle = np.pi + np.angle(velocity)
		radius = self.rho * np.abs(velocity)
		return x - radius * np.cos(angle), y + radius * np.sin(angle)
