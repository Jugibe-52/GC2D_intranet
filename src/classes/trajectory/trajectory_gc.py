# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Guiding-centre trajectory dynamics."""

from typing import Literal

import numpy as np

from contracts import TrajectoryParams
from ..potential import Array, Potential
from .trajectory import Trajectory


class TrajectoryGC(Trajectory):
	"""Guiding-centre dynamics with state ``[x, y]``."""

	kind = "gc"
	_state_dimension = 2

	def __init__(self, potential: Potential, params: TrajectoryParams) -> None:
		super().__init__(potential, params, ndof=1.5)

	def _initial_state(self, x: Array, y: Array) -> Array:
		return np.concatenate((x, y))

	def get_velocities(self, state: Array) -> None:
		"""Guiding-centre states do not contain independent velocities."""
		self._split_state(state)
		return None

	def hamiltonian(self, t: float | Array, state: Array) -> Array:  # type: ignore[override]
		x, y = self.get_positions(state)
		return np.asarray(self.psi(t, x, y))

	def y_dot(  # type: ignore[override]
		self,
		t: float,
		state: Array,
		output: Literal["full", "reduced"] = "full",
	) -> Array:
		x, y = self.get_positions(state)
		ex, ey = self.electric_field(t, x, y)
		return np.concatenate((ey, -ex))

	def y_dot_lyap(self, t: float, state: Array) -> Array:
		x, y, *jacobian_parts = np.split(state, 6)
		position = np.concatenate((x, y))
		jacobian = np.asarray(jacobian_parts).reshape((2, 2, -1))

		d2psi_dx2 = self.psi(t, x, y, dx=2)
		d2psi_dxdy = self.psi(t, x, y, dx=1, dy=1)
		d2psi_dy2 = self.psi(t, x, y, dy=2)
		linearization = np.zeros_like(jacobian)
		linearization[0, 0], linearization[0, 1] = -d2psi_dxdy, -d2psi_dy2
		linearization[1, 0], linearization[1, 1] = d2psi_dx2, d2psi_dxdy
		jacobian_dot = np.einsum("ijm,jkm->ikm", linearization, jacobian)
		return np.concatenate((self.y_dot(t, position), jacobian_dot.reshape(-1)))

	def k_dot(self, t: float, state: Array) -> float:  # type: ignore[override]
		x, y = self.get_positions(state)
		return -float(np.sum(self.psi(t, x, y, dt=1)))
