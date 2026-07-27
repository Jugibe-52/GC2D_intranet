"""Guiding-centre physical dynamics independent of initial conditions."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from classes.potential import Potential

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
