# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Method-owned exact cyclotron and electric-kick operations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from contracts.dynamics.protocols import CyclotronSplitSystem, HamiltonianSystem
from contracts.dynamics.coordinates import StateLayout
from contracts.methods.maps import Projection
from dynamics.coordinates import describe_coordinates
from dynamics.fc import FullCyclotronDynamics
from methods._momentum import _updated_momentum
from formulations.state import SingleFormulation


def _real_imaginary(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	return np.asarray(value.real), np.asarray(value.imag)


@dataclass(frozen=True, slots=True, init=False)
class FCSplitMaps:
	"""Immutable operations on supplied FC work arrays; no initial state is kept.

	Optional trailing map momentum has physical kappa normalization. These maps
	retain the established direct/adjoint interface for signed substeps.
	"""

	dynamics: CyclotronSplitSystem
	physical_size: int
	particle_count: int
	track_energy: bool

	def __init__(self, dynamics: CyclotronSplitSystem, physical_size: int, *,
	             coordinates: StateLayout | None = None, track_energy: bool = False) -> None:
		declaration: StateLayout | None = coordinates if coordinates is not None else getattr(dynamics, 'coordinates', None)
		declared: StateLayout | None = getattr(dynamics, 'coordinates', None)
		if coordinates is not None and declared is not None and describe_coordinates(coordinates) != describe_coordinates(declared):
			raise ValueError('Explicit coordinates disagree with dynamics.')
		if declaration is not None and declaration.state_dimension != dynamics.state_dimension:
			raise ValueError('Coordinate dimension disagrees with dynamics.')
		if declaration is None or describe_coordinates(declaration) != FullCyclotronDynamics.coordinates:
			raise TypeError('FC split maps require declared (x, y, vx, vy) coordinates.')
		if not isinstance(dynamics, CyclotronSplitSystem):
			raise TypeError('FC split maps require CyclotronSplitSystem dynamics.')
		if track_energy and not isinstance(dynamics, HamiltonianSystem):
			raise TypeError('Energy tracking requires HamiltonianSystem.')
		size: int = SingleFormulation(physical_size).physical_size
		if size % declaration.state_dimension:
			raise ValueError('FC maps require complete physical particle blocks.')
		object.__setattr__(self, 'dynamics', dynamics)
		object.__setattr__(self, 'physical_size', size)
		object.__setattr__(self, 'particle_count', size // declaration.state_dimension)
		object.__setattr__(self, 'track_energy', bool(track_energy))

	@property
	def dynamics_name(self) -> str:
		return type(self.dynamics).__name__

	def _unpack(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
		expected: int = self.physical_size + (self.particle_count if self.track_energy else 0)
		if state.ndim == 0 or state.shape[0] != expected:
			raise ValueError('The FC map changed the workspace shape.')
		return state[:self.physical_size], state[self.physical_size:] if self.track_energy else None

	def _cyclotron_step(self, state: np.ndarray, duration: float) -> np.ndarray:
		"""Advance the exact field-free flow with its established arithmetic."""
		x: Any
		y: Any
		vx: Any
		vy: Any
		x, y, vx, vy = FullCyclotronDynamics.coordinates.split(state)
		rotation: Any = np.exp(-1j * duration * self.dynamics.larmor_frequency)
		x, y = _real_imaginary(x + 1j * y + 1j * self.dynamics.rho * np.sign(self.dynamics.eta)
		                      * (rotation - 1) * (vx + 1j * vy))
		vx, vy = _real_imaginary(rotation * (vx + 1j * vy))
		return np.asarray(FullCyclotronDynamics.coordinates.pack_components(x, y, vx, vy))

	def _kick(self, state: np.ndarray, duration: float, time: float) -> np.ndarray:
		x: Any
		y: Any
		vx: Any
		vy: Any
		x, y, vx, vy = FullCyclotronDynamics.coordinates.split(state)
		ax: Any
		ay: Any
		ax, ay = self.dynamics.electric_acceleration(time, x, y)
		return np.asarray(FullCyclotronDynamics.coordinates.pack_components(x, y, vx + duration * ax, vy + duration * ay))

	@staticmethod
	def _pack(physical: np.ndarray, momentum: np.ndarray | None) -> np.ndarray:
		return physical if momentum is None else np.concatenate((physical, momentum), axis=0)

	def direct_map(self, duration: float, t: float, state: np.ndarray) -> np.ndarray:
		"""Exact cyclotron flow, electric kick, then its passive quadrature."""
		physical: np.ndarray
		momentum: np.ndarray | None
		physical, momentum = self._unpack(state)
		physical = self._kick(self._cyclotron_step(physical, duration), duration, t)
		momentum = _updated_momentum(self.dynamics, momentum, duration, t, physical)
		return self._pack(physical, momentum)

	def adjoint_map(self, duration: float, t: float, state: np.ndarray) -> np.ndarray:
		"""Electric kick and passive quadrature followed by exact cyclotron flow."""
		physical: np.ndarray
		momentum: np.ndarray | None
		physical, momentum = self._unpack(state)
		physical = self._kick(physical, duration, t)
		momentum = _updated_momentum(self.dynamics, momentum, duration, t, physical)
		return self._pack(self._cyclotron_step(physical, duration), momentum)

	def project(self, internal_history: np.ndarray) -> Projection:
		"""Expose physical samples and separately supplied map momentum."""
		physical: np.ndarray
		momentum: np.ndarray | None
		physical, momentum = self._unpack(internal_history)
		return physical, {} if momentum is None else {'extended_momentum': momentum}


__all__ = ['FCSplitMaps']
