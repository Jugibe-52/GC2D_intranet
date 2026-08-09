"""Guiding-centre physical dynamics independent of initial conditions."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from potential import Potential

from ._layout import pack_components, split_components


class GuidingCenterDynamics:
	"""Guiding-centre equations over a fixed gyroaveraged potential."""

	state_dimension: ClassVar[int] = 2

	def __init__(self, potential: Potential, *, rho: float = 0.0) -> None:
		"""Create physical GC dynamics for one normalized Larmor radius."""
		if not isinstance(potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		rho = float(rho)
		if not np.isfinite(rho) or rho < 0:
			raise ValueError("`rho` must be finite and non-negative.")
		self.potential = potential
		self.rho = rho
		self.effective_potential = potential.gyroaverage(rho)

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Evaluate GC drift while preserving the packed physical layout."""
		x, y = split_components(state, component_count=self.state_dimension)
		ex, ey = self.effective_potential.electric_field(
			t,
			x,
			y,
		)
		return pack_components(ey, -ex)

	def particle_vector_field_jacobians(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return one exact two-by-two spatial Jacobian per GC particle.

		For ``N`` particles the packed state is ``[x_1, ..., x_N, y_1, ...,
		y_N]`` and the result has shape ``(N, 2, 2)``. Particles are uncoupled
		by the field evaluation, so this batched representation avoids assembling
		a sparse ``(2N, 2N)`` matrix. With ``f = (-phi_y, phi_x)``, the rows are
		``(-phi_xy, -phi_yy)`` and ``(phi_xx, phi_xy)``.
		"""
		x, y = split_components(state, component_count=self.state_dimension)
		potential = self.effective_potential
		if potential.interpolation_order < 3:
			raise ValueError(
				"Exact GC vector-field Jacobians require interpolation_order >= 3."
			)
		phi_xx = potential.evaluate(t, x, y, dx=2)
		phi_xy = potential.evaluate(t, x, y, dx=1, dy=1)
		phi_yy = potential.evaluate(t, x, y, dy=2)
		return np.stack(
			(
				np.stack((-phi_xy, -phi_yy), axis=-1),
				np.stack((phi_xx, phi_xy), axis=-1),
			),
			axis=-2,
		)

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate gyroaveraged Hamiltonian values."""
		x, y = split_components(state, component_count=self.state_dimension)
		return self.effective_potential.evaluate(t, x, y)

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate the time-conjugate momentum derivative."""
		x, y = split_components(state, component_count=self.state_dimension)
		return -self.effective_potential.evaluate(
			t,
			x,
			y,
			dt=1,
		)


__all__ = ["GuidingCenterDynamics"]
