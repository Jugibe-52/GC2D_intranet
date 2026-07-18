# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Guiding-centre trajectory dynamics."""

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
	) -> Array:
		x, y = self.get_positions(state)
		ex, ey = self.electric_field(t, x, y)
		return np.concatenate((ey, -ex))

	def k_dot(self, t: float, state: Array) -> float:  # type: ignore[override]
		x, y = self.get_positions(state)
		return -float(np.sum(self.psi(t, x, y, dt=1)))
