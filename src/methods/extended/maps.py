# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Method-owned doubled GC maps and harmonic coupling on supplied workspaces."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from typing import cast
from contracts.arrays import BackendArray
from types import ModuleType
from contracts.arrays import array_namespace, checked_finite
from contracts.dynamics.coordinates import StateLayout
from dynamics.coordinates import GCCoordinates
from contracts.methods.maps import Projection
from contracts.methods.functions import JacobianStructure, VectorField
from formulations.state import DoubleFormulation

def spatial_shear(f: VectorField, time: float, target: BackendArray,
                  source: BackendArray, duration: float) -> BackendArray:
	"""Update one spatial copy using the field at the other copy.

	Both arrays use the same component-major physical layout. The method supplies
	the signed duration and actual evaluation time; no auxiliary participates.
	"""
	xp: ModuleType = array_namespace(source)
	derivative: BackendArray = xp.asarray(f(time, source), dtype=float)
	derivative = checked_finite(derivative, "The vector field changed shape or became non-finite.", ValueError)
	if derivative.shape != source.shape:
		raise ValueError("The vector field changed shape or became non-finite.")
	return cast(BackendArray, xp.asarray(target + duration * derivative))


_COUPLING_BASE: 'np.ndarray' = np.asarray(
	[[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
	dtype=float,
)
_COUPLING_COS: 'np.ndarray' = np.asarray(
	[[1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1]],
	dtype=float,
)
_COUPLING_SIN: 'np.ndarray' = np.asarray(
	[[0, -1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0]],
	dtype=float,
)


def _coupling_matrix(duration: float, frequency: float) -> BackendArray:
	"""Return exact harmonic mixing for both duplicated GC copies."""
	xp: ModuleType = array_namespace(duration)
	return cast(BackendArray, xp.asarray(
		(
			_COUPLING_BASE
			+ xp.cos(2 * frequency * duration) * _COUPLING_COS
			+ xp.sin(2 * frequency * duration) * _COUPLING_SIN
		)
		/ 2
	))


def gc_coupling_matrix(duration: float, frequency: float) -> BackendArray:
	"""Return a safe copy of the exact doubled-GC coupling matrix."""
	xp: ModuleType = array_namespace(duration)
	step: float = duration
	rate: float = frequency
	step = checked_finite(step, "`duration` must be finite.")
	if not np.isfinite(rate) or rate < 0.0:
		raise ValueError("`frequency` must be finite and non-negative.")
	return _coupling_matrix(step, rate).copy()


@dataclass(frozen=True, slots=True)
class HarmonicCoupling:
	"""Independent C flow for packed doubled planar states of shape (4*N,).

	The component order is (u_x, u_y, v_x, v_y), each block of length N.
	The copy mean is fixed; their difference rotates by 2*frequency*duration.
	Frequency has inverse runtime-time units. Time and passive momentum are
	managed by the caller and are not part of this spatial array.
	"""

	frequency: float

	def __post_init__(self) -> None:
		"""Validate the reusable coupling rate independently of any method."""
		rate: float = float(self.frequency)
		if not np.isfinite(rate) or rate < 0:
			raise ValueError("`frequency` must be finite and non-negative.")
		object.__setattr__(self, "frequency", rate)

	def apply(self, state: BackendArray, duration: float) -> BackendArray:
		"""Return the exact signed C flow without modifying the input array."""
		xp: ModuleType = array_namespace(state)
		value: BackendArray = xp.asarray(state, dtype=float)
		value = checked_finite(value, "C requires a finite doubled planar vector of length 4*N.", ValueError)
		if value.ndim != 1 or value.size == 0 or value.size % 4:
			raise ValueError("C requires a finite doubled planar vector of length 4*N.")
		step: float = duration
		step = checked_finite(step, "`duration` must be finite.")
		return cast(BackendArray, xp.asarray((_coupling_matrix(step, self.frequency) @ value.reshape(4, -1)).reshape(-1)))


@dataclass(frozen=True, slots=True)
class GCDoubledMaps:
	"""Spatial direct/adjoint maps using explicit functions and two physical copies."""

	f: VectorField
	physical_size: int
	coordinates: StateLayout = field(kw_only=True)
	jacobian: VectorField | None = field(default=None, kw_only=True)
	jacobian_structure: JacobianStructure = field(default='particle', kw_only=True)
	coupling_frequency: float | None = field(default=np.pi / 8, kw_only=True)
	supports_stage_projection: bool = field(default=False, kw_only=True)
	state_formulation: DoubleFormulation = field(init=False)
	coupling: HarmonicCoupling | None = field(init=False)
	particle_count: int = field(init=False)

	def __post_init__(self) -> None:
		"""Validate physical roles and bind spatial work without retaining a problem."""
		if not callable(self.f) or (self.jacobian is not None and not callable(self.jacobian)):
			raise TypeError('Supply a field function and an optional Jacobian function.')
		layout: StateLayout = self.coordinates.snapshot()
		if type(layout) is not GCCoordinates:
			raise TypeError('GC physical maps require GCCoordinates position coordinates.')
		if self.jacobian_structure not in ('dense', 'particle'):
			raise ValueError('Jacobian structure must be dense or particle.')
		representation: DoubleFormulation = DoubleFormulation(self.physical_size)
		if self.physical_size % layout.state_dimension:
			raise ValueError('GC maps require complete physical particle blocks.')
		coupling: HarmonicCoupling | None = None if self.coupling_frequency is None else HarmonicCoupling(self.coupling_frequency)
		object.__setattr__(self, 'state_formulation', representation)
		object.__setattr__(self, 'coupling', coupling)
		object.__setattr__(self, 'coupling_frequency', None if coupling is None else coupling.frequency)
		object.__setattr__(self, 'particle_count', self.physical_size // layout.state_dimension)

	def direct_map(self, duration: float, t: float, state: BackendArray) -> BackendArray:
		"""Apply B, A and optional C, retaining the established arithmetic order."""
		updated: BackendArray = self.apply_spatial('B', duration, t, state)
		updated = self.apply_spatial('A', duration, t, updated)
		return updated if self.coupling is None else self.apply_spatial('C', duration, t, updated)

	def adjoint_map(self, duration: float, t: float, state: BackendArray) -> BackendArray:
		"""Apply optional C, A and B in the adjoint order."""
		current: BackendArray = state if self.coupling is None else self.apply_spatial('C', duration, t, state)
		current = self.apply_spatial('A', duration, t, current)
		return self.apply_spatial('B', duration, t, current)

	def apply_spatial(
		self, operation: Literal['A', 'B', 'C'], duration: float, t: float, state: BackendArray,
	) -> BackendArray:
		"""Apply one signed spatial flow; the caller owns time translations."""
		xp: ModuleType = array_namespace(state)
		first: BackendArray
		second: BackendArray
		first, second = self.state_formulation.split_copies(state)
		if operation == 'C':
			if self.coupling is None:
				raise TypeError('The uncoupled GC maps have no coupling flow.')
			return self.coupling.apply(self.state_formulation.join_copies(first, second), duration)
		if operation == 'A':
			first = spatial_shear(self.f, t, first, second, duration)
		elif operation == 'B':
			second = spatial_shear(self.f, t, second, first, duration)
		else:
			raise ValueError('A spatial operation must be A, B or C.')
		return cast(BackendArray, xp.asarray(self.state_formulation.join_copies(first, second)))

	def project_internal_state(self, state: BackendArray) -> BackendArray:
		"""Embed the mean of both physical copies back onto the diagonal."""
		xp: ModuleType = array_namespace(state)
		if not self.supports_stage_projection:
			raise TypeError('These prepared maps do not permit stage projection.')
		mean: BackendArray = self.project(state)[0]
		return cast(BackendArray, xp.asarray(self.state_formulation.join_copies(mean, mean)))

	def project(self, internal_history: BackendArray) -> Projection:
		"""Recover the physical mean without auxiliary state or diagnostics."""
		xp: ModuleType = array_namespace(internal_history)
		first: BackendArray
		second: BackendArray
		first, second = self.state_formulation.split_copies(internal_history)
		return xp.asarray((first + second) / 2), {}


__all__ = ['GCDoubledMaps', 'HarmonicCoupling', 'gc_coupling_matrix']
