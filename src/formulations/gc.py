# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Doubled guiding-centre formulation for direct/adjoint compositions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, TYPE_CHECKING, TypeAlias

import numpy as np

from dynamics.protocols import DynamicalSystem, HamiltonianSystem
from contracts.configuration import StateLayout
from formulations.layout import GCStateLayout


if TYPE_CHECKING:
	from contracts.problem import InitialValueProblem
	from typing import Any


from formulations.base import _updated_momentum
from formulations.base import (
	PreparedDirectAdjointFormulation,
	PreparedStageProjectedFormulation,
	Projection,
)


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


@dataclass(frozen=True, slots=True)
class _GCExtendedState:
	"""Two physical copies and optional time-conjugate momentum."""

	first: np.ndarray
	second: np.ndarray
	momentum: np.ndarray | None = None

	def pack(self) -> np.ndarray:
		parts = (self.first, self.second)
		return np.concatenate(parts if self.momentum is None else (*parts, self.momentum))


@dataclass(frozen=True, slots=True, init=False)
class GCDoubledMaps:
	"""Spatial GC maps constructed directly for one problem.

	The map vector holds two spatial copies and, optionally, a summed energy
	accumulator. Its time is supplied separately by the composition. Accepted
	R4/R6 states and normalized kappa are owned by DoubledFormulation.
	"""

	dynamics: DynamicalSystem
	configuration: np.ndarray
	layout: StateLayout
	coupling_frequency: float | None
	coupling: HarmonicCoupling | None
	physical_size: int
	particle_count: int
	track_energy: bool
	supports_stage_projection: bool
	dynamics_name: str
	initial_internal_state: np.ndarray

	def __init__(self, problem: InitialValueProblem, coupling_frequency: float | None = np.pi / 8,
	             *, track_energy: bool = False, supports_stage_projection: bool = False) -> None:
		"""Bind validated spatial maps without a configuration/preparation chain."""
		layout = problem.layout
		if layout.state_dimension != 2:
			raise TypeError("GC doubled maps require a planar component-major layout.")
		if coupling_frequency is not None:
			frequency: float = float(coupling_frequency)
			if not np.isfinite(frequency) or frequency < 0:
				raise ValueError("`coupling_frequency` must be finite and non-negative.")
			coupling_frequency = frequency
		if track_energy and not isinstance(problem.dynamics, HamiltonianSystem):
			raise TypeError("Energy tracking requires HamiltonianSystem.")
		physical = problem.initial_state
		count = problem.particle_count
		initial: np.ndarray = _GCExtendedState(physical, physical, np.zeros(count) if track_energy else None).pack()
		initial.setflags(write=False)
		object.__setattr__(self, "dynamics", problem.dynamics)
		object.__setattr__(self, "configuration", problem.initial_state)
		object.__setattr__(self, "layout", layout)
		object.__setattr__(self, "coupling_frequency", coupling_frequency)
		object.__setattr__(self, "coupling", None if coupling_frequency is None else HarmonicCoupling(coupling_frequency))
		object.__setattr__(self, "physical_size", physical.size)
		object.__setattr__(self, "particle_count", count)
		object.__setattr__(self, "track_energy", bool(track_energy))
		object.__setattr__(self, "supports_stage_projection", supports_stage_projection)
		object.__setattr__(self, "dynamics_name", type(problem.dynamics).__name__)
		object.__setattr__(self, "initial_internal_state", initial)

	def _unpack(self, value: np.ndarray) -> _GCExtendedState:
		expected = 2 * self.physical_size + (
			self.particle_count if self.track_energy else 0
		)
		if value.ndim == 0 or value.shape[0] != expected:
			raise ValueError("The GC extended map changed the internal state shape.")
		momentum = value[2 * self.physical_size :] if self.track_energy else None
		return _GCExtendedState(
			first=value[: self.physical_size],
			second=value[self.physical_size : 2 * self.physical_size],
			momentum=momentum,
		)

	def _couple(
		self,
		duration: float,
		state: _GCExtendedState,
	) -> _GCExtendedState:
		coupling = self.coupling
		if coupling is None:
			raise TypeError("The uncoupled GC formulation has no coupling flow.")
		first: tuple[np.ndarray, ...] = self.layout.split(state.first)
		second: tuple[np.ndarray, ...] = self.layout.split(state.second)
		blocks: np.ndarray = np.stack((*first, *second), axis=0)
		coupled: np.ndarray = coupling.apply(blocks.reshape(-1), duration).reshape(blocks.shape)
		return _GCExtendedState(
			first=GCStateLayout().from_blocks(coupled[:2]),
			second=GCStateLayout().from_blocks(coupled[2:]),
			momentum=state.momentum,
		)

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
		current: _GCExtendedState = self._unpack(state)
		if operation == 'C':
			return self._couple(duration, current).pack()
		first: np.ndarray
		second: np.ndarray
		first, second = current.first, current.second
		if operation == 'A':
			source = second
			first = spatial_shear(self.dynamics, t, first, source, duration)
		elif operation == 'B':
			source = first
			second = spatial_shear(self.dynamics, t, second, source, duration)
		else:
			raise ValueError('A spatial operation must be A, B or C.')
		momentum: np.ndarray | None = _updated_momentum(self.dynamics, current.momentum, duration, t, source)
		if energy_points is not None:
			energy_points.append((t, duration, source))
		return _GCExtendedState(first, second, momentum).pack()

	def project_internal_state(self, state: np.ndarray) -> np.ndarray:
		"""Embed the mean of both physical copies back onto the diagonal."""
		if not self.supports_stage_projection:
			raise TypeError(
				"This prepared formulation does not permit stage projection."
			)
		current: _GCExtendedState = self._unpack(state)
		mean: np.ndarray = np.asarray((current.first + current.second) / 2)
		return _GCExtendedState(
			first=mean,
			second=mean,
			momentum=current.momentum,
		).pack()

	def project(self, internal_history: np.ndarray) -> Projection:
		"""Average both copies and expose projected extended momentum."""
		final_state: _GCExtendedState = self._unpack(internal_history)
		states: np.ndarray = np.asarray((final_state.first + final_state.second) / 2)
		diagnostics: dict[str, np.ndarray | float | int | str | bool] = {}
		if final_state.momentum is not None:
			diagnostics["extended_momentum"] = final_state.momentum / 2
		return states, diagnostics


@dataclass(frozen=True, slots=True)
class GCExtendedFormulation:
	"""Reusable doubled-state GC formulation configuration."""

	coupling_frequency: float = np.pi / 8

	def __post_init__(self) -> None:
		frequency: float = float(self.coupling_frequency)
		if not np.isfinite(frequency) or frequency < 0:
			raise ValueError("`coupling_frequency` must be finite and non-negative.")
		object.__setattr__(self, "coupling_frequency", frequency)

	def prepare(
		self,
		problem: InitialValueProblem,
		*,
		track_energy: bool,
	) -> PreparedDirectAdjointFormulation:
		"""Bind immutable GC maps to one compatible problem."""
		return GCDoubledMaps(
			problem,
			track_energy=track_energy,
			coupling_frequency=self.coupling_frequency,
			supports_stage_projection=False,
		)


@dataclass(frozen=True, slots=True)
class GCStageProjectedFormulation:
	"""Uncoupled doubled GC maps projected after every composition stage."""

	def prepare(
		self,
		problem: InitialValueProblem,
		*,
		track_energy: bool,
	) -> PreparedStageProjectedFormulation:
		"""Bind uncoupled triangular GC maps to one compatible problem."""
		return GCDoubledMaps(
			problem,
			track_energy=track_energy,
			coupling_frequency=None,
			supports_stage_projection=True,
		)


__all__ = [
	"HarmonicCoupling",
	"GCExtendedFormulation",
	"GCDoubledMaps",
	"GCStageProjectedFormulation",
	"gc_coupling_matrix",
]
