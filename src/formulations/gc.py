# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Doubled guiding-centre formulation for direct/adjoint compositions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TypeAlias

import numpy as np

from dynamics import DynamicalSystem, HamiltonianSystem
from initial_conditions import GCInitialConfiguration

from contracts.problem import InitialValueProblem
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
	derivative = np.asarray(dynamics.vector_field(time, source), dtype=float)
	if derivative.shape != source.shape or not np.all(np.isfinite(derivative)):
		raise ValueError("The vector field changed shape or became non-finite.")
	return np.asarray(target + duration * derivative)


_COUPLING_BASE = np.asarray(
	[[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
	dtype=float,
)
_COUPLING_COS = np.asarray(
	[[1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1]],
	dtype=float,
)
_COUPLING_SIN = np.asarray(
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
	step = float(duration)
	rate = float(frequency)
	if not np.isfinite(step):
		raise ValueError("`duration` must be finite.")
	if not np.isfinite(rate) or rate < 0.0:
		raise ValueError("`frequency` must be finite and non-negative.")
	return _coupling_matrix(step, rate).copy()


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
	configuration: GCInitialConfiguration
	coupling_frequency: float | None
	physical_size: int
	particle_count: int
	track_energy: bool
	supports_stage_projection: bool
	dynamics_name: str
	initial_internal_state: np.ndarray

	def __init__(self, problem: InitialValueProblem, coupling_frequency: float | None = np.pi / 8,
	             *, track_energy: bool = False, supports_stage_projection: bool = False) -> None:
		"""Bind validated spatial maps without a configuration/preparation chain."""
		configuration = problem.initial_configuration
		if not isinstance(configuration, GCInitialConfiguration):
			raise TypeError("GC doubled maps require a GC configuration.")
		if coupling_frequency is not None:
			frequency = float(coupling_frequency)
			if not np.isfinite(frequency) or frequency < 0:
				raise ValueError("`coupling_frequency` must be finite and non-negative.")
			coupling_frequency = frequency
		if track_energy and not isinstance(problem.dynamics, HamiltonianSystem):
			raise TypeError("Energy tracking requires HamiltonianSystem.")
		physical = problem.initial_state
		count = problem.particle_count
		initial = _GCExtendedState(physical, physical, np.zeros(count) if track_energy else None).pack()
		initial.setflags(write=False)
		object.__setattr__(self, "dynamics", problem.dynamics)
		object.__setattr__(self, "configuration", configuration)
		object.__setattr__(self, "coupling_frequency", coupling_frequency)
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
		frequency = self.coupling_frequency
		if frequency is None:
			raise TypeError("The uncoupled GC formulation has no coupling flow.")
		first = self.configuration.layout.split(state.first)
		second = self.configuration.layout.split(state.second)
		blocks = np.stack((*first, *second), axis=0)
		coupled = np.asarray(
			np.einsum(
				"ij,j...->i...",
				_coupling_matrix(duration, frequency),
				blocks,
			)
		)
		return _GCExtendedState(
			first=self.configuration.layout.from_blocks(coupled[:2]),
			second=self.configuration.layout.from_blocks(coupled[2:]),
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
		current = self._unpack(state)
		second = spatial_shear(self.dynamics, t, current.second, current.first, duration)
		momentum = _updated_momentum(
			self.dynamics,
			current.momentum,
			duration,
			t,
			current.first,
		)
		first = spatial_shear(self.dynamics, t, current.first, second, duration)
		momentum = _updated_momentum(self.dynamics, momentum, duration, t, second)
		if energy_points is not None:
			energy_points.extend(((t, duration, current.first), (t, duration, second)))
		updated = _GCExtendedState(first, second, momentum)
		if self.coupling_frequency is None:
			return updated.pack()
		return self._couple(duration, updated).pack()

	def adjoint_map(
		self,
		duration: float,
		t: float,
		state: np.ndarray,
		*,
		energy_points: list[_EnergyQuadraturePoint] | None = None,
	) -> np.ndarray:
		"""Couple and update both copies, optionally retaining quadrature states."""
		current = self._unpack(state)
		if self.coupling_frequency is not None:
			current = self._couple(duration, current)
		first = spatial_shear(self.dynamics, t, current.first, current.second, duration)
		momentum = _updated_momentum(
			self.dynamics,
			current.momentum,
			duration,
			t,
			current.second,
		)
		second = spatial_shear(self.dynamics, t, current.second, first, duration)
		momentum = _updated_momentum(self.dynamics, momentum, duration, t, first)
		if energy_points is not None:
			energy_points.extend(((t, duration, current.second), (t, duration, first)))
		return _GCExtendedState(first, second, momentum).pack()

	def project_internal_state(self, state: np.ndarray) -> np.ndarray:
		"""Embed the mean of both physical copies back onto the diagonal."""
		if not self.supports_stage_projection:
			raise TypeError(
				"This prepared formulation does not permit stage projection."
			)
		current = self._unpack(state)
		mean = np.asarray((current.first + current.second) / 2)
		return _GCExtendedState(
			first=mean,
			second=mean,
			momentum=current.momentum,
		).pack()

	def project(self, internal_history: np.ndarray) -> Projection:
		"""Average both copies and expose projected extended momentum."""
		final_state = self._unpack(internal_history)
		first = self.configuration.layout.split(final_state.first)
		second = self.configuration.layout.split(final_state.second)
		states = self.configuration.layout.pack_components(
			(first.x + second.x) / 2,
			(first.y + second.y) / 2,
		)
		diagnostics: dict[str, np.ndarray | float | int | str | bool] = {}
		if final_state.momentum is not None:
			diagnostics["extended_momentum"] = final_state.momentum / 2
		return states, diagnostics




@dataclass(frozen=True, slots=True)
class GCExtendedFormulation:
	"""Reusable doubled-state GC formulation configuration."""

	coupling_frequency: float = np.pi / 8

	def __post_init__(self) -> None:
		frequency = float(self.coupling_frequency)
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
	"GCExtendedFormulation",
	"GCDoubledMaps",
	"GCStageProjectedFormulation",
	"gc_coupling_matrix",
]
