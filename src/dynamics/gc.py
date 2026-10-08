"""Guiding-centre physical dynamics independent of initial conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from potential.potential import Potential

from contracts.arrays import array_namespace
from contracts.state_layout import GCStateLayout
from ._equations import gc_velocity


@dataclass(frozen=True, init=False, eq=False)
class GuidingCenterDynamics:
	"""GC equations over a fixed gyroaveraged potential, for NumPy or JAX arrays."""

	state_dimension: ClassVar[int] = 2

	potential: Potential
	rho: float
	effective_potential: Potential

	def __init__(self, potential: Potential, *, rho: float = 0.0) -> None:
		"""Create physical GC dynamics for one normalized Larmor radius."""
		if not isinstance(potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		rho = float(rho)
		if not np.isfinite(rho) or rho < 0:
			raise ValueError("`rho` must be finite and non-negative.")
		object.__setattr__(self, "potential", potential)
		object.__setattr__(self, "rho", rho)
		object.__setattr__(self, "effective_potential", potential.gyroaverage(rho))

	def vector_field(self, t: Any, state: Any) -> Any:
		"""Evaluate GC drift at one time or broadcast times over a packed history."""
		x, y = GCStateLayout().split(state)
		ex, ey = self.effective_potential.electric_field(
			t,
			x,
			y,
		)
		return GCStateLayout.pack_components(*gc_velocity(ex, ey))

	def particle_vector_field_jacobians(
		self,
		t: Any,
		state: Any,
	) -> Any:
		"""Return one exact two-by-two spatial Jacobian per GC particle.

		For ``N`` particles the packed state is ``[x_1, ..., x_N, y_1, ...,
		y_N]`` and the result has shape ``(N, 2, 2)``. Particles are uncoupled
		by the field evaluation, so this batched representation avoids assembling
		a sparse ``(2N, 2N)`` matrix. With ``f = (-phi_y, phi_x)``, the rows are
		``(-phi_xy, -phi_yy)`` and ``(phi_xx, phi_xy)``.
		"""
		x, y = GCStateLayout().split(state)
		potential = self.effective_potential
		if potential.interpolation_order < 3:
			raise ValueError(
				"Exact GC vector-field Jacobians require interpolation_order >= 3."
			)
		phi_xx = potential.evaluate(t, x, y, dx=2)
		phi_xy = potential.evaluate(t, x, y, dx=1, dy=1)
		phi_yy = potential.evaluate(t, x, y, dy=2)
		xp = array_namespace(t, state)
		return xp.stack(
			(
				xp.stack((-phi_xy, -phi_yy), axis=-1),
				xp.stack((phi_xx, phi_xy), axis=-1),
			),
			axis=-2,
		)

	def hamiltonian(
		self,
		t: Any,
		state: Any,
	) -> Any:
		"""Evaluate gyroaveraged Hamiltonian values."""
		x, y = GCStateLayout().split(state)
		return self.effective_potential.evaluate(t, x, y)

	def extended_momentum_derivative(
		self,
		t: Any,
		state: Any,
	) -> Any:
		"""Evaluate the time-conjugate momentum derivative."""
		x, y = GCStateLayout().split(state)
		return -self.effective_potential.evaluate(
			t,
			x,
			y,
			dt=1,
		)


__all__ = ["GuidingCenterDynamics"]
