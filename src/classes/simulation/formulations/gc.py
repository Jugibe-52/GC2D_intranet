# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Doubled guiding-centre formulation for direct/adjoint compositions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from classes.dynamics import DynamicalSystem, ExtendedHamiltonianSystem
from classes.trajectory import TrajectoryGC

from ..problem import InitialValueProblem
from .base import PreparedDirectAdjointFormulation, Projection


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


@dataclass(frozen=True, slots=True)
class _GCExtendedState:
	"""Two physical copies and optional time-conjugate momentum."""

	first: np.ndarray
	second: np.ndarray
	momentum: np.ndarray | None = None

	def pack(self) -> np.ndarray:
		parts = (self.first, self.second)
		return np.concatenate(parts if self.momentum is None else (*parts, self.momentum))


@dataclass(frozen=True, slots=True)
class _PreparedGC:
	"""Immutable GC maps bound to one problem and diagnostic choice."""

	dynamics: DynamicalSystem
	configuration: TrajectoryGC
	coupling_frequency: float
	physical_size: int
	particle_count: int
	track_energy: bool
	observer_label: str
	initial_internal_state: np.ndarray

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

	def _vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		derivative = np.asarray(self.dynamics.vector_field(t, state))
		if derivative.shape != state.shape:
			raise ValueError("The GC vector field changed the physical state shape.")
		return derivative

	def _updated_momentum(
		self,
		momentum: np.ndarray | None,
		duration: float,
		t: float,
		state: np.ndarray,
	) -> np.ndarray | None:
		if momentum is None:
			return None
		assert isinstance(self.dynamics, ExtendedHamiltonianSystem)
		derivative = np.asarray(
			self.dynamics.extended_momentum_derivative(t, state)
		)
		if derivative.shape != momentum.shape:
			raise ValueError("The extended-momentum derivative changed its shape.")
		return np.asarray(momentum + duration * derivative)

	def _couple(
		self,
		duration: float,
		state: _GCExtendedState,
	) -> _GCExtendedState:
		first = self.configuration.split(state.first)
		second = self.configuration.split(state.second)
		blocks = np.stack((*first, *second), axis=0)
		coupled = np.asarray(
			np.einsum(
				"ij,j...->i...",
				_coupling_matrix(duration, self.coupling_frequency),
				blocks,
			)
		)
		return _GCExtendedState(
			first=self.configuration.from_blocks(coupled[:2]),
			second=self.configuration.from_blocks(coupled[2:]),
			momentum=state.momentum,
		)

	def direct_map(
		self,
		duration: float,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Update second then first copy and apply exact coupling."""
		current = self._unpack(state)
		second = current.second + duration * self._vector_field(t, current.first)
		momentum = self._updated_momentum(
			current.momentum,
			duration,
			t,
			current.first,
		)
		first = current.first + duration * self._vector_field(t, second)
		momentum = self._updated_momentum(momentum, duration, t, second)
		return self._couple(
			duration,
			_GCExtendedState(first, second, momentum),
		).pack()

	def adjoint_map(
		self,
		duration: float,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Apply coupling then update first and second copies."""
		current = self._couple(duration, self._unpack(state))
		first = current.first + duration * self._vector_field(t, current.second)
		momentum = self._updated_momentum(
			current.momentum,
			duration,
			t,
			current.second,
		)
		second = current.second + duration * self._vector_field(t, first)
		momentum = self._updated_momentum(momentum, duration, t, first)
		return _GCExtendedState(first, second, momentum).pack()

	def project(self, internal_history: np.ndarray) -> Projection:
		"""Average both copies and expose projected extended momentum."""
		final_state = self._unpack(internal_history)
		first = self.configuration.split(final_state.first)
		second = self.configuration.split(final_state.second)
		states = self.configuration.pack_components(
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
		configuration = problem.initial_configuration
		if not isinstance(configuration, TrajectoryGC):
			raise TypeError("GCExtendedFormulation requires a GC configuration.")
		if not isinstance(problem.dynamics, DynamicalSystem):
			raise TypeError("GC formulation requires DynamicalSystem.")
		if track_energy and not isinstance(
			problem.dynamics,
			ExtendedHamiltonianSystem,
		):
			raise TypeError("Energy tracking requires ExtendedHamiltonianSystem.")
		physical = problem.initial_state
		particle_count = configuration.particle_count(physical)
		extended = _GCExtendedState(
			first=physical,
			second=physical,
			momentum=np.zeros(particle_count) if track_energy else None,
		)
		initial_internal_state = extended.pack()
		initial_internal_state.setflags(write=False)
		return _PreparedGC(
			dynamics=problem.dynamics,
			configuration=configuration,
			coupling_frequency=self.coupling_frequency,
			physical_size=physical.size,
			particle_count=particle_count,
			track_energy=bool(track_energy),
			observer_label=str(
				getattr(problem.dynamics, "observer_name", "GuidingCenterDynamics")
			),
			initial_internal_state=initial_internal_state,
		)


__all__ = ["GCExtendedFormulation"]
