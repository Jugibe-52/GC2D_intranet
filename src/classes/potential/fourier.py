"""Analytic turbulent potential represented by a truncated Fourier series."""

from __future__ import annotations

import numpy as np
from scipy.special import jv

from contracts import Array

from .base import Potential
from .grid import Grid


class FourierPotential(Potential):
	r"""Synthetic potential evaluated directly in Fourier space.

	The representation preserves the project's historical analytic Fourier
	convention:

	.. math::

	   \phi(t,x,y) = \operatorname{Im}\sum_{n,m} c_{nm}
	   \exp\left(i(nx + my - t)\right).

	Spatial and temporal derivatives are evaluated analytically; no spline or
	grid resampling participates in the dynamics.
	"""

	def __init__(
		self,
		amplitude: float,
		modes: int,
		*,
		seed: int = 27,
		grid_size: int = 64,
		coefficients: Array | None = None,
	) -> None:
		amplitude = float(amplitude)
		if not np.isfinite(amplitude):
			raise ValueError("`amplitude` must be finite.")
		if isinstance(modes, (bool, np.bool_)) or not isinstance(modes, (int, np.integer)):
			raise TypeError("`modes` must be an integer.")
		if modes < 1:
			raise ValueError("`modes` must be positive.")
		if isinstance(grid_size, (bool, np.bool_)) or not isinstance(grid_size, (int, np.integer)):
			raise TypeError("`grid_size` must be an integer.")
		if grid_size < 2:
			raise ValueError("`grid_size` must be at least 2.")

		self.amplitude = amplitude
		self.modes = int(modes)
		self.seed = int(seed)
		self.grid = Grid.from_bounds(
			0.0,
			2 * np.pi,
			0.0,
			2 * np.pi,
			int(grid_size),
			int(grid_size),
			periodic=True,
		)
		self.wave_numbers = np.asarray(
			np.meshgrid(
				np.arange(self.modes + 1),
				np.arange(self.modes + 1),
				indexing="ij",
			)
		)
		if coefficients is None:
			# RandomState intentionally preserves the sequence used by the legacy
			# Fourier model for a given seed.
			random = np.random.RandomState(self.seed)
			phases = 2 * np.pi * random.random_sample((self.modes, self.modes))
			coefficient_array = np.zeros(
				(self.modes + 1, self.modes + 1),
				dtype=np.complex128,
			)
			n, m = self.wave_numbers
			coefficient_array[1:, 1:] = (
				self.amplitude
				/ (n[1:, 1:] ** 2 + m[1:, 1:] ** 2) ** 1.5
				* np.exp(1j * phases)
			)
			coefficient_array[np.hypot(n, m) > self.modes] = 0
		else:
			coefficient_array = np.asarray(coefficients, dtype=np.complex128)
			expected = (self.modes + 1, self.modes + 1)
			if coefficient_array.shape != expected:
				raise ValueError(
					f"`coefficients` has shape {coefficient_array.shape}, expected {expected}."
				)
			coefficient_array = coefficient_array.copy()
		self.coefficients = coefficient_array

	@property
	def phic(self) -> Array:
		"""Fourier coefficients, exposed under the conventional mathematical name."""
		return self.coefficients

	def __str__(self) -> str:
		active = int(np.count_nonzero(self.coefficients))
		return (
			f"FourierPotential(amplitude={self.amplitude:g}, modes={self.modes}, "
			f"active_modes={active}, seed={self.seed})"
		)

	@staticmethod
	def _derivative_order(value: int, name: str) -> int:
		if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
			raise TypeError(f"`{name}` must be an integer.")
		if value < 0:
			raise ValueError(f"`{name}` must be non-negative.")
		return int(value)

	def field_at_time(
		self,
		t: float | Array,
		x: Array | None = None,
		y: Array | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> Array:
		"""Evaluate the analytic series or one of its exact derivatives."""
		dx = self._derivative_order(dx, "dx")
		dy = self._derivative_order(dy, "dy")
		dt = self._derivative_order(dt, "dt")
		if (x is None) != (y is None):
			raise ValueError("`x` and `y` must be provided together.")
		if x is None:
			x, y = np.meshgrid(self.grid.x, self.grid.y, indexing="ij")
		assert y is not None
		x_array, y_array = np.broadcast_arrays(np.asarray(x), np.asarray(y))
		coordinates = np.asarray((x_array, y_array))
		spatial_phase = np.einsum(
			"aij,a...->ij...",
			self.wave_numbers,
			coordinates,
		)
		phase = spatial_phase - np.asarray(t)
		n, m = self.wave_numbers
		factor = (1j * n) ** dx * (1j * m) ** dy * (-1j) ** dt
		extra_axes = (1,) * (phase.ndim - 2)
		weighted = (self.coefficients * factor).reshape(self.coefficients.shape + extra_axes)
		value = np.sum(weighted * np.exp(1j * phase), axis=(0, 1)).imag
		return np.asarray(value)

	def gyroaveraged(self, rho: float) -> FourierPotential:
		rho = float(rho)
		if not np.isfinite(rho) or rho < 0:
			raise ValueError("`rho` must be a finite, non-negative number.")
		n, m = self.wave_numbers
		factor = jv(0, rho * np.hypot(n, m))
		return FourierPotential(
			self.amplitude,
			self.modes,
			seed=self.seed,
			grid_size=self.grid.nx,
			coefficients=self.coefficients * factor,
		)

	def copy(self) -> FourierPotential:
		return FourierPotential(
			self.amplitude,
			self.modes,
			seed=self.seed,
			grid_size=self.grid.nx,
			coefficients=self.coefficients,
		)


__all__ = ["FourierPotential"]
