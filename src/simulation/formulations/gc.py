# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Doubled guiding-centre formulation for direct/adjoint compositions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from dynamics import DynamicalSystem, ExtendedHamiltonianSystem
from initial_conditions import GCInitialConfiguration

from ..problem import InitialValueProblem
from .base import (
	PreparedDirectAdjointFormulation,
	PreparedStageProjectedFormulation,
	Projection,
)


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


@dataclass(frozen=True, slots=True)
class _PreparedGC:
	"""Immutable GC maps bound to one problem and diagnostic choice."""

	dynamics: DynamicalSystem
	configuration: GCInitialConfiguration
	coupling_frequency: float | None
	physical_size: int
	particle_count: int
	track_energy: bool
	supports_stage_projection: bool
	dynamics_name: str
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
	) -> np.ndarray:
		"""Update second then first copy and optionally apply exact coupling."""
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
		updated = _GCExtendedState(first, second, momentum)
		if self.coupling_frequency is None:
			return updated.pack()
		return self._couple(duration, updated).pack()

	def adjoint_map(
		self,
		duration: float,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Optionally apply coupling, then update first and second copies."""
		current = self._unpack(state)
		if self.coupling_frequency is not None:
			current = self._couple(duration, current)
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


def _prepare_gc(
	problem: InitialValueProblem,
	*,
	track_energy: bool,
	coupling_frequency: float | None,
	supports_stage_projection: bool,
	formulation_name: str,
) -> _PreparedGC:
	"""Build one immutable doubled GC state for a numerical formulation."""
	configuration = problem.initial_configuration
	if not isinstance(configuration, GCInitialConfiguration):
		raise TypeError(f"{formulation_name} requires a GC configuration.")
	if not isinstance(problem.dynamics, DynamicalSystem):
		raise TypeError("GC formulation requires DynamicalSystem.")
	if track_energy and not isinstance(
		problem.dynamics,
		ExtendedHamiltonianSystem,
	):
		raise TypeError("Energy tracking requires ExtendedHamiltonianSystem.")
	physical = problem.initial_state
	particle_count = configuration.layout.particle_count(physical)
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
		coupling_frequency=coupling_frequency,
		physical_size=physical.size,
		particle_count=particle_count,
		track_energy=bool(track_energy),
		supports_stage_projection=supports_stage_projection,
		dynamics_name=type(problem.dynamics).__name__,
		initial_internal_state=initial_internal_state,
	)


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
		return _prepare_gc(
			problem,
			track_energy=track_energy,
			coupling_frequency=self.coupling_frequency,
			supports_stage_projection=False,
			formulation_name=type(self).__name__,
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
		return _prepare_gc(
			problem,
			track_energy=track_energy,
			coupling_frequency=None,
			supports_stage_projection=True,
			formulation_name=type(self).__name__,
		)


__all__ = [
	"GCExtendedFormulation",
	"GCStageProjectedFormulation",
	"gc_coupling_matrix",
]
