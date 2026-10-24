# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Method-owned doubled GC maps and harmonic coupling on supplied workspaces."""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, TypeAlias
import numpy as np
from contracts.dynamics.protocols import DynamicalSystem, HamiltonianSystem
from contracts.dynamics.coordinates import StateLayout
from contracts.methods.maps import Projection
from dynamics.coordinates import describe_coordinates
from dynamics.gc import GuidingCenterDynamics
from formulations.state import DoubleFormulation
from methods._momentum import _updated_momentum

# Each point retains (evaluation time, signed shear duration, packed physical
# state). Spatial maps allocate their updates, so these references stay valid.
_EnergyQuadraturePoint: TypeAlias = tuple[float, float, np.ndarray]


def spatial_shear(dynamics: DynamicalSystem, time: float, target: np.ndarray,
                  source: np.ndarray, duration: float) -> np.ndarray:
	"""Update one spatial copy using the field at the other copy.

	Both arrays use the same component-major physical layout. The method supplies
	the signed duration and actual evaluation time; no auxiliary participates.
	"""
	derivative: np.ndarray = np.asarray(dynamics.vector_field(time, source), dtype=float)
	if derivative.shape != source.shape or not np.all(np.isfinite(derivative)):
		raise ValueError("The vector field changed shape or became non-finite.")
	return np.asarray(target + duration * derivative)


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


@lru_cache(maxsize=256)
def _coupling_matrix(duration: float, frequency: float) -> np.ndarray:
	"""Return exact harmonic mixing for both duplicated GC copies."""
	return np.asarray(
		(
			_COUPLING_BASE
			+ np.cos(2 * frequency * duration) * _COUPLING_COS
			+ np.sin(2 * frequency * duration) * _COUPLING_SIN
		)
		/ 2
	)


def gc_coupling_matrix(duration: float, frequency: float) -> np.ndarray:
	"""Return a safe copy of the exact doubled-GC coupling matrix."""
	step: float = float(duration)
	rate: float = float(frequency)
	if not np.isfinite(step):
		raise ValueError("`duration` must be finite.")
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

	def apply(self, state: np.ndarray, duration: float) -> np.ndarray:
		"""Return the exact signed C flow without modifying the input array."""
		value: np.ndarray = np.asarray(state, dtype=float)
		if value.ndim != 1 or value.size == 0 or value.size % 4 or not np.all(np.isfinite(value)):
			raise ValueError("C requires a finite doubled planar vector of length 4*N.")
		step: float = float(duration)
		if not np.isfinite(step):
			raise ValueError("`duration` must be finite.")
		return np.asarray((_coupling_matrix(step, self.frequency) @ value.reshape(4, -1)).reshape(-1))


@dataclass(frozen=True, slots=True, init=False)
class GCDoubledMaps:
	"""GC direct/adjoint operations with no retained initial state.

	Work arrays contain two supplied physical copies. Optional trailing momentum
	is the SUM of both map contributions; project divides that sum by two.
	Accepted physical kappa remains separate in the integration snapshot.
	"""

	dynamics: DynamicalSystem
	state_formulation: DoubleFormulation
	coupling_frequency: float | None
	coupling: HarmonicCoupling | None
	physical_size: int
	particle_count: int
	track_energy: bool
	supports_stage_projection: bool

	def __init__(self, dynamics: DynamicalSystem, physical_size: int,
	             coupling_frequency: float | None = np.pi / 8, *,
	             coordinates: StateLayout | None = None, track_energy: bool = False,
	             supports_stage_projection: bool = False) -> None:
		"""Bind numerical operations and sizes, never a run's initial values."""
		declaration: StateLayout | None = coordinates if coordinates is not None else getattr(dynamics, 'coordinates', None)
		declared: StateLayout | None = getattr(dynamics, 'coordinates', None)
		if coordinates is not None and declared is not None and describe_coordinates(coordinates) != describe_coordinates(declared):
			raise ValueError('Explicit coordinates disagree with dynamics.')
		if declaration is not None and declaration.state_dimension != dynamics.state_dimension:
			raise ValueError('Coordinate dimension disagrees with dynamics.')
		if declaration is None or describe_coordinates(declaration) != GuidingCenterDynamics.coordinates:
			raise TypeError('GC doubled maps require declared (x, y) position coordinates.')
		representation: DoubleFormulation = DoubleFormulation(physical_size)
		if physical_size % declaration.state_dimension:
			raise ValueError('GC maps require complete physical particle blocks.')
		if track_energy and not isinstance(dynamics, HamiltonianSystem):
			raise TypeError('Energy tracking requires HamiltonianSystem.')
		coupling: HarmonicCoupling | None = None if coupling_frequency is None else HarmonicCoupling(coupling_frequency)
		object.__setattr__(self, 'dynamics', dynamics)
		object.__setattr__(self, 'state_formulation', representation)
		object.__setattr__(self, 'coupling', coupling)
		object.__setattr__(self, 'coupling_frequency', None if coupling is None else coupling.frequency)
		object.__setattr__(self, 'physical_size', representation.physical_size)
		object.__setattr__(self, 'particle_count', physical_size // declaration.state_dimension)
		object.__setattr__(self, 'track_energy', bool(track_energy))
		object.__setattr__(self, 'supports_stage_projection', supports_stage_projection)

	@property
	def dynamics_name(self) -> str:
		return type(self.dynamics).__name__

	def _unpack(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
		expected: int = self.state_formulation.spatial_size + (self.particle_count if self.track_energy else 0)
		if value.ndim == 0 or value.shape[0] != expected:
			raise ValueError('The GC map changed the workspace shape.')
		first: np.ndarray
		second: np.ndarray
		first, second = self.state_formulation.split_copies(value[:2 * self.physical_size])
		return first, second, value[2 * self.physical_size:] if self.track_energy else None

	def _pack(self, first: np.ndarray, second: np.ndarray, momentum: np.ndarray | None) -> np.ndarray:
		spatial: np.ndarray = self.state_formulation.join_copies(first, second)
		return spatial if momentum is None else np.concatenate((spatial, momentum), axis=0)

	def direct_map(
		self,
		duration: float,
		t: float,
		state: np.ndarray,
		*,
		energy_points: list[_EnergyQuadraturePoint] | None = None,
	) -> np.ndarray:
		"""Update both copies, optionally retaining energy quadrature states."""
		updated: np.ndarray = self.apply_spatial('B', duration, t, state, energy_points=energy_points)
		updated = self.apply_spatial('A', duration, t, updated, energy_points=energy_points)
		if self.coupling_frequency is None:
			return updated
		return self.apply_spatial('C', duration, t, updated)

	def adjoint_map(
		self,
		duration: float,
		t: float,
		state: np.ndarray,
		*,
		energy_points: list[_EnergyQuadraturePoint] | None = None,
	) -> np.ndarray:
		"""Couple and update both copies, optionally retaining quadrature states."""
		current: np.ndarray = state
		if self.coupling_frequency is not None:
			current = self.apply_spatial('C', duration, t, current)
		current = self.apply_spatial('A', duration, t, current, energy_points=energy_points)
		return self.apply_spatial('B', duration, t, current, energy_points=energy_points)

	def apply_spatial(
		self, operation: Literal['A', 'B', 'C'], duration: float, t: float,
		state: np.ndarray, *, energy_points: list[_EnergyQuadraturePoint] | None = None,
	) -> np.ndarray:
		"""Apply one exact spatial flow at fixed time, with no hidden half-step.

		A updates the first copy from the second, B updates the second from the
		first, and C rotates their difference. The caller owns time translation.
		"""
		first: np.ndarray
		second: np.ndarray
		momentum: np.ndarray | None
		first, second, momentum = self._unpack(state)
		if operation == 'C':
			if self.coupling is None:
				raise TypeError('The uncoupled GC maps have no coupling flow.')
			coupled: np.ndarray = self.coupling.apply(self.state_formulation.join_copies(first, second), duration)
			first, second = self.state_formulation.split_copies(coupled)
			return self._pack(first, second, momentum)
		if operation == 'A':
			source = second
			first = spatial_shear(self.dynamics, t, first, source, duration)
		elif operation == 'B':
			source = first
			second = spatial_shear(self.dynamics, t, second, source, duration)
		else:
			raise ValueError('A spatial operation must be A, B or C.')
		momentum = _updated_momentum(self.dynamics, momentum, duration, t, source)
		if energy_points is not None:
			energy_points.append((t, duration, source))
		return self._pack(first, second, momentum)

	def project_internal_state(self, state: np.ndarray) -> np.ndarray:
		"""Embed the mean of both physical copies back onto the diagonal."""
		if not self.supports_stage_projection:
			raise TypeError(
				"This prepared formulation does not permit stage projection."
			)
		first: np.ndarray
		second: np.ndarray
		momentum: np.ndarray | None
		first, second, momentum = self._unpack(state)
		mean: np.ndarray = np.asarray((first + second) / 2)
		return self._pack(mean, mean, momentum)

	def project(self, internal_history: np.ndarray) -> Projection:
		"""Average both copies and expose projected extended momentum."""
		first: np.ndarray
		second: np.ndarray
		momentum: np.ndarray | None
		first, second, momentum = self._unpack(internal_history)
		states: np.ndarray = np.asarray((first + second) / 2)
		diagnostics: dict[str, np.ndarray | float | int | str | bool] = {}
		if momentum is not None:
			diagnostics["extended_momentum"] = momentum / 2
		return states, diagnostics


__all__ = ['GCDoubledMaps', 'HarmonicCoupling', 'gc_coupling_matrix']
