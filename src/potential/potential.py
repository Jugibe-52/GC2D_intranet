# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Periodic electrostatic potentials represented by a mean and harmonic modes.

Every potential uses the same runtime convention, independently of whether its
fields were generated artificially or loaded from measured data:

``Phi(t, x, y) = Phi_0(x, y) + 2 Re sum_j[C_j(x, y) exp(i 2*pi*f_j*t)]``.

Keeping the harmonic time dependence separate makes spatial interpolation and
gyroaveraging independent of the evaluation time.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Any, Self, overload

import numpy as np
from numpy.fft import fft2, fftfreq, ifft2
from scipy.special import jv

from contracts.execution_options import ExecutionOptions

from .grid import Grid
from .gc2d_h5 import (
	DEFAULT_CHARACTERISTIC_LENGTH,
	SpatialNormalization,
	_load_gc2d_h5_data,
)
from .prepared import PreparedPotential
from ._evaluation import PotentialEvaluator
from .scipy_evaluator import ScipyPotentialEvaluator

_DEFAULT_EXECUTION = ExecutionOptions()


def _validated_random_parameters(A: float, M: int, seed: int) -> tuple[float, int, int]:
	"""Check random-spectrum controls, retaining the generator's seed policy."""
	A = float(A)
	if not np.isfinite(A) or A < 0:
		raise ValueError("`A` must be a finite, non-negative number.")
	if (
		isinstance(M, (bool, np.bool_))
		or not isinstance(M, (int, np.integer))
		or M < 1
	):
		raise ValueError("`M` must be a positive integer.")
	if isinstance(seed, (bool, np.bool_)) or not isinstance(
		seed, (int, np.integer)
	):
		raise TypeError("`seed` must be an integer.")

	return A, M, seed


def _random_positive_frequency_mode(
	grid: Grid,
	*,
	amplitude: float,
	maximum_wave_number: int,
	seed: int,
) -> np.ndarray:
	"""Generate the canonical positive-frequency mode for a random potential."""
	wave_x, wave_y = np.meshgrid(
		np.arange(maximum_wave_number + 1),
		np.arange(maximum_wave_number + 1),
		indexing="ij",
	)
	spectrum = np.zeros(
		(maximum_wave_number + 1, maximum_wave_number + 1),
		dtype=np.complex128,
	)
	phases = 2.0 * np.pi * np.random.default_rng(seed).random(
		(maximum_wave_number, maximum_wave_number)
	)
	spectrum[1:, 1:] = (
		amplitude
		/ (wave_x[1:, 1:] ** 2 + wave_y[1:, 1:] ** 2) ** 1.5
		* np.exp(1j * phases)
	)
	# A circular cutoff makes ``maximum_wave_number`` limit |k| without
	# privileging diagonal spatial modes.
	spectrum[np.hypot(wave_x, wave_y) > maximum_wave_number] = 0

	mode_x, mode_y = np.indices(spectrum.shape)
	x_mesh, y_mesh = np.meshgrid(grid.x, grid.y, indexing="ij")
	phase = np.exp(
		1j
		* (
			mode_x[:, :, None, None] * x_mesh[None, None, :, :]
			+ mode_y[:, :, None, None] * y_mesh[None, None, :, :]
		)
	)
	legacy_coefficient = np.asarray(np.einsum("nm,nm...->...", spectrum, phase))
	# Re(C exp(-i*t)) equals 2*Re((conj(C)/2) exp(+i*2*pi*f*t)) for
	# f=1/(2*pi), preserving the established artificial field exactly.
	return np.asarray(0.5 * np.conjugate(legacy_coefficient))


class Potential:
	"""Mean plus positive-frequency modes on a regular periodic grid.

	``mean`` has shape ``grid.shape == (nx, ny)``. ``modes`` has
	shape ``(mode_count, nx, ny)``, and ``frequencies`` stores one positive cycle
	rate per mode. A missing mean is normalized to a zero array and missing modes
	to an empty collection, giving every instance the same internal structure.
	"""

	def __init__(
		self,
		grid: Grid,
		mean: np.ndarray | None = None,
		modes: np.ndarray | None = None,
		frequencies: np.ndarray | None = None,
		*,
		metadata: object | None = None,
		interpolation_order: int = 3,
	) -> None:
		"""Store sampled fields and prepare their periodic interpolants.

		``interpolation_order`` is the polynomial degree used independently on both
		spatial axes.  It affects off-grid evaluations but not the stored samples.
		"""
		self._prepared = PreparedPotential.build(grid, mean, modes, frequencies, interpolation_order)
		self.metadata = metadata
		self._evaluators: dict[ExecutionOptions, PotentialEvaluator] = {}

	@property
	def prepared(self) -> PreparedPotential:
		"""Shared immutable data from which all evaluators are prepared."""
		return self._prepared

	@property
	def grid(self) -> Grid:
		"""Periodic spatial grid shared by every evaluator."""
		return self._prepared.grid

	@property
	def interpolation_order(self) -> int:
		"""Polynomial degree of the prepared spatial splines."""
		return self._prepared.interpolation_order

	@property
	def mean(self) -> np.ndarray:
		"""Read-only mean samples with shape (nx, ny)."""
		return self._prepared.mean

	@property
	def modes(self) -> np.ndarray:
		"""Read-only complex samples with shape (mode_count, nx, ny)."""
		return self._prepared.modes

	@property
	def frequencies(self) -> np.ndarray:
		"""Read-only positive harmonic frequencies in cycles per normalized time."""
		return self._prepared.frequencies

	def _evaluator(self, execution: ExecutionOptions | None) -> PotentialEvaluator:
		"""Resolve a per-call choice and reuse the matching prepared evaluator."""
		choice = _DEFAULT_EXECUTION if execution is None else execution
		if not isinstance(choice, ExecutionOptions):
			raise TypeError("`execution` must be an ExecutionOptions instance or None.")
		if choice not in self._evaluators:
			if choice.backend == "scipy":
				evaluator: PotentialEvaluator = ScipyPotentialEvaluator(self.prepared)
			else:
				from .jax_evaluator import JaxPotentialEvaluator
				evaluator = JaxPotentialEvaluator(self.prepared, device=choice.device, device_index=choice.device_index)
			self._evaluators[choice] = evaluator
		return self._evaluators[choice]

	def __getstate__(self) -> dict[str, Any]:
		"""Serialize physical data only; device buffers and compiled functions are local."""
		return dict(grid=self.grid, mean=self.mean, modes=self.modes, frequencies=self.frequencies,
		            metadata=self.metadata, interpolation_order=self.interpolation_order)

	def __setstate__(self, state: dict[str, Any]) -> None:
		"""Rebuild CPU splines and an empty device cache in the receiving process."""
		Potential.__init__(self, **state)

	@classmethod
	def random(
		cls,
		*,
		A: float,
		M: int,
		nx: int,
		ny: int,
		seed: int = 27,
		interpolation_order: int = 3,
	) -> Potential:
		"""Create the reproducible periodic potential used in the notebooks.

		``A`` controls the spectral amplitude and ``M`` the maximum radial
		spatial wave number.  Each retained mode receives a reproducible random
		phase and an amplitude that decays as ``|k|**-3``.  ``nx`` and ``ny`` are
		the numbers of physical-space samples; they determine the returned
		mode shape, not the number of generated spatial wave numbers.
		"""
		A, M, seed = _validated_random_parameters(A, M, seed)

		grid = Grid.periodic(nx, ny)
		mode = _random_positive_frequency_mode(
			grid,
			amplitude=A,
			maximum_wave_number=int(M),
			seed=int(seed),
		)
		return cls(
			grid,
			modes=mode[np.newaxis, ...],
			frequencies=np.asarray([1.0 / (2.0 * np.pi)]),
			interpolation_order=interpolation_order,
		)

	@classmethod
	def from_gc2d_h5(
		cls,
		filename: str | PathLike[str],
		*,
		B: float = 1.5,
		characteristic_length: float = DEFAULT_CHARACTERISTIC_LENGTH,
		characteristic_frequency: float | None = None,
		indx: int | Sequence[int] | np.ndarray | None = (0, 1),
		nx: int | None = None,
		ny: int | None = None,
		denoising: bool = False,
		sigma: float = 1.0,
		interpolation_order: int = 3,
		spatial_normalization: SpatialNormalization = "characteristic_length",
	) -> Self:
		"""Construct a potential from measured GC2D HDF5 fields.

		Options and normalization follow :func:`~potential.load_gc2d_h5_potential`.
		By default, select the mean and dominant positive-frequency mode with
		``B=1.5`` and characteristic length ``0.06``. The HDF5 adapter handles
		selection, normalization, optional filtering and resampling; this class
		prepares the common runtime representation and retains source metadata.
		"""
		data = _load_gc2d_h5_data(
			filename,
			B=B,
			characteristic_length=characteristic_length,
			characteristic_frequency=characteristic_frequency,
			indx=indx,
			nx=nx,
			ny=ny,
			denoising=denoising,
			sigma=sigma,
			interpolation_order=interpolation_order,
			spatial_normalization=spatial_normalization,
		)
		return cls(
			data.grid,
			mean=data.mean,
			modes=data.modes,
			frequencies=data.frequencies,
			metadata=data.metadata,
			interpolation_order=interpolation_order,
		)

	@overload
	def evaluate(self, t: Any, x: Any, y: Any, *, dx: int = 0, dy: int = 0, dt: int = 0,
	             execution: None = None) -> np.ndarray: ...

	@overload
	def evaluate(self, t: Any, x: Any, y: Any, *, dx: int = 0, dy: int = 0, dt: int = 0,
	             execution: ExecutionOptions) -> Any: ...

	def evaluate(self, t: Any, x: Any, y: Any, *, dx: int = 0, dy: int = 0, dt: int = 0,
	             execution: ExecutionOptions | None = None) -> Any:
		"""Evaluate paired points or derivatives using one explicit execution choice.

		Coordinates must have the same shape; time broadcasts against that shape.
		Omitting execution always selects SciPy/CPU and returns a NumPy array.
		JAX returns a device array. A choice never changes later default calls.
		"""
		return self._evaluator(execution).evaluate(t, x, y, dx=dx, dy=dy, dt=dt)

	@overload
	def evaluate_grid(self, t: Any, *, dt: int = 0, execution: None = None) -> np.ndarray: ...

	@overload
	def evaluate_grid(self, t: Any, *, dt: int = 0, execution: ExecutionOptions) -> Any: ...

	def evaluate_grid(self, t: Any, *, dt: int = 0, execution: ExecutionOptions | None = None) -> Any:
		"""Return grid values with spatial axes preceding any axes contributed by time."""
		return self._evaluator(execution).evaluate_grid(t, dt=dt)

	@overload
	def electric_field(self, t: Any, x: Any = None, y: Any = None,
	                   *, execution: None = None) -> tuple[np.ndarray, np.ndarray]: ...

	@overload
	def electric_field(self, t: Any, x: Any = None, y: Any = None,
	                   *, execution: ExecutionOptions) -> tuple[Any, Any]: ...

	def electric_field(self, t: Any, x: Any = None, y: Any = None,
	                   *, execution: ExecutionOptions | None = None) -> tuple[Any, Any]:
		"""Return (-phi_x, -phi_y) at paired points, or on the grid if both are omitted."""
		return self._evaluator(execution).electric_field(t, x, y)

	def gyroaverage(self, rho: float) -> Potential:
		"""Return the Larmor-circle average of every field at radius ``rho``.

		A circular average multiplies each Fourier mode by ``J_0(rho |k|)``.
		This spectral form performs the average exactly for the sampled modes.
		``rho`` uses the same length scale as the grid coordinates, making
		``rho |k|`` dimensionless.
		"""
		radius = float(rho)
		if not np.isfinite(radius) or radius < 0:
			raise ValueError("`rho` must be finite and non-negative.")
		if radius == 0:
			return self
		# ``fftfreq`` returns cycles per unit length; multiplying its norm by
		# ``2*pi`` below converts it to the angular wave number used by J_0.
		kx = fftfreq(self.grid.nx, d=self.grid.dx)
		ky = fftfreq(self.grid.ny, d=self.grid.dy)
		kx_mesh, ky_mesh = np.meshgrid(kx, ky, indexing="ij")
		# ``factor`` has shape ``(nx, ny)`` and attenuates each discrete spatial
		# Fourier coefficient without mixing modes.
		wave_number_norm = np.sqrt(kx_mesh**2 + ky_mesh**2)
		factor = jv(0, 2 * np.pi * radius * wave_number_norm)
		mean = np.asarray(ifft2(fft2(self.mean) * factor).real)
		modes = (
			self.modes.copy()
			if self.modes.shape[0] == 0
			else np.asarray(
				[ifft2(fft2(field) * factor) for field in self.modes],
				dtype=np.complex128,
			)
		)
		return Potential(
			self.grid,
			mean=mean,
			modes=modes,
			frequencies=self.frequencies,
			metadata=self.metadata,
			interpolation_order=self.interpolation_order,
		)


__all__ = ["Potential"]
