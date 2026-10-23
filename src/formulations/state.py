"""Model-independent physical copies with optional passive energy coordinates.

One physical copy contains d component-major blocks of N particles. Single and
double representations contain d*N and 2*d*N entries, respectively; tracking
appends one time and one normalized momentum block, each of size N. Methods
own initial values, step preparation, projection and passive quadrature.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np

from contracts.arrays import array_namespace
from contracts.formulations.layout import StateLayout
from contracts.states.history import DiagnosticValue

if TYPE_CHECKING:
	from contracts.arrays import BackendArray
	from types import ModuleType


@dataclass(frozen=True)
class SingleFormulation:
	"""One physical copy and optional ``(t, kappa)`` blocks per particle.

	Only coordinate structure is retained. ``lift`` creates physical copies;
	``pack`` appends caller-supplied auxiliary values to those copies. Neither
	operation evaluates dynamics or selects initial values, solvers or projection.
	"""

	layout: StateLayout
	particle_count: int
	track_energy: bool = False
	physical_size: int = field(init=False)
	copies: ClassVar[int] = 1

	def __post_init__(self) -> None:
		"""Snapshot a valid structural layout without binding a physical problem."""
		if not isinstance(self.layout, StateLayout):
			raise TypeError("A state formulation requires a structural StateLayout.")
		dimension: int = self.layout.state_dimension
		if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension <= 0:
			raise ValueError("The layout dimension must be a positive integer.")
		if isinstance(self.particle_count, bool) or not isinstance(self.particle_count, Integral) or self.particle_count <= 0:
			raise ValueError("The particle count must be a positive integer.")
		object.__setattr__(self, "layout", deepcopy(self.layout))
		object.__setattr__(self, "particle_count", int(self.particle_count))
		object.__setattr__(self, "physical_size", int(dimension) * self.particle_count)

	@property
	def spatial_size(self) -> int:
		"""Number of physical entries across all copies, excluding auxiliaries."""
		return self.copies * self.physical_size

	@property
	def dimension(self) -> int:
		"""Packed internal size, including optional time and momentum blocks."""
		return self.spatial_size + (2 * self.particle_count if self.track_energy else 0)

	def _array(self, state: Any, size: int) -> Any:
		"""Validate static dimensions without converting traced array values."""
		value: BackendArray = array_namespace(state).asarray(state)
		if value.ndim == 0 or value.shape[0] != size:
			raise ValueError(f"Expected a packed state with leading dimension {size}.")
		return value

	def lift(self, physical: Any) -> Any:
		"""Build one or two equal physical copies, without auxiliary coordinates.

	The input has shape ``(d*N, *sample_axes)``. NumPy output owns its data,
	and its copy blocks can be mutated independently; JAX remains functional.
	"""
		value: BackendArray = self._array(physical, self.physical_size)
		return self.join_copies(*(value for _ in range(self.copies)))

	def join_copies(self, *physical: Any) -> Any:
		"""Pack explicitly supplied copies, including shifted projection candidates."""
		if len(physical) != self.copies:
			raise ValueError(f"{type(self).__name__} requires {self.copies} physical copies.")
		xp: ModuleType = array_namespace(*physical)
		values: tuple[Any, ...] = tuple(self._array(xp.asarray(value), self.physical_size) for value in physical)
		if any(value.shape != values[0].shape for value in values[1:]):
			raise ValueError("Physical copies must have matching sample axes.")
		return xp.concatenate(values, axis=0)

	def split_copies(self, spatial: Any) -> tuple[Any, ...]:
		"""Read physical copy blocks from a spatial array without projecting them."""
		value: BackendArray = self._array(spatial, self.spatial_size)
		return tuple(value[index * self.physical_size:(index + 1) * self.physical_size]
			for index in range(self.copies))

	def pack(self, spatial: Any, time: Any, momentum: Any = None) -> Any:
		"""Append caller-supplied auxiliaries to already prepared physical copies.

	``spatial`` has shape ``(copies*d*N, *sample_axes)`` and ``momentum`` has
	shape ``(N, *sample_axes)``. Time broadcasts to the latter shape. Kappa has
	the physical Hamiltonian normalization, including in the FC representation.
	"""
		xp: ModuleType = array_namespace(spatial, time, momentum)
		value: BackendArray = self._array(xp.asarray(spatial), self.spatial_size)
		parts: list[Any] = [value]
		if self.track_energy:
			if momentum is None:
				raise ValueError("Energy momentum must have one component per particle.")
			kappa: BackendArray = xp.asarray(momentum)
			if kappa.shape != (self.particle_count, *value.shape[1:]):
				raise ValueError("Energy momentum must have one component per particle.")
			parts += [xp.broadcast_to(time, kappa.shape), kappa]
		return xp.concatenate(parts, axis=0)

	def components(self, state: Any) -> Any:
		"""View an internal array as ``(coordinates, particles, *sample_axes)``."""
		value: BackendArray = self._array(state, self.dimension)
		return value.reshape(self.dimension // self.particle_count, self.particle_count, *value.shape[1:])

	def physical(self, state: Any) -> Any:
		"""Read the first physical copy of an accepted internal state.

	Doubled callers must first apply their method's projection and repack equal
	copies. This operation never averages or corrects an off-diagonal input.
	"""
		value: BackendArray = self._array(state, self.dimension)
		return value[:self.physical_size]

	def time(self, state: Any) -> Any:
		"""Return prescribed particle times, or None without energy tracking."""
		return self.components(state)[-2] if self.track_energy else None

	def momentum(self, state: Any) -> Any:
		"""Return passive physical kappa, or None without energy tracking."""
		return self.components(state)[-1] if self.track_energy else None

	def metadata(self) -> dict[str, DiagnosticValue]:
		"""Keep stable saved labels and distinguish physical and internal sizes."""
		name: str = "physical" if self.copies == 1 else "duplicated"
		return {
			"state_formulation": name + ("_with_energy" if self.track_energy else ""),
			"track_energy": self.track_energy,
			"accepted_internal_state_dimension": self.dimension,
			"base_splitting_state_dimension": self.dimension,
			"spatial_state_dimension": self.spatial_size,
			"observer_state_dimension": self.physical_size,
			"observer_state_kind": "physical_map",
			"energy_feedback": False,
		}

	def extract_sample(self, time: float, state: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
		"""Export a host sample, validating prescribed time within round-off."""
		physical: np.ndarray = self.physical(state)
		momentum: np.ndarray | None = self.momentum(state)
		if momentum is None:
			return physical, {}
		particle_times: np.ndarray = self.time(state)
		tolerance: float = float(16 * np.finfo(float).eps * max(1.0, abs(time)))
		if not np.allclose(particle_times, time, rtol=0.0, atol=tolerance):
			raise ValueError("Each particle time must match the sample time within round-off.")
		return physical, {"extended_momentum": momentum}

	def diagnose_history(self, times: np.ndarray, physical: np.ndarray,
		auxiliary: dict[str, np.ndarray], *, energy: np.ndarray | None = None,
	) -> dict[str, DiagnosticValue]:
		"""Assemble energy diagnostics from physical values evaluated by the caller.

	This output conversion evaluates no dynamics. The method supplies H on
	the saved states; the representation only combines H with stored kappa.
	"""
		momentum: np.ndarray | None = auxiliary.get("extended_momentum")
		if momentum is None:
			return {}
		if energy is None:
			raise ValueError("Energy diagnostics require caller-supplied Hamiltonian values.")
		energy = np.asarray(energy, dtype=float)
		if energy.ndim == 1:
			energy = energy[np.newaxis, :]
		if energy.shape != momentum.shape or not np.all(np.isfinite(energy)):
			raise ValueError("Hamiltonian history must be finite with shape (particles, times).")
		balance: np.ndarray = energy + momentum
		error: np.ndarray = balance - balance[:, :1]
		return {
			"extended_time": np.broadcast_to(times, momentum.shape).copy(),
			"extended_momentum": momentum,
			"extended_momentum_normalization": "physical_kappa",
			"physical_hamiltonian": energy,
			"hamiltonian": energy,
			"energy_drift": energy - energy[:, :1],
			"generalized_energy": balance,
			"generalized_energy_error": error,
			"energy_error": float(np.max(np.abs(error))),
		}

	def extract_history(self, times: np.ndarray, history: np.ndarray,
		*, energy: np.ndarray | None = None,
	) -> tuple[np.ndarray, dict[str, DiagnosticValue]]:
		"""Export host history with an optional pre-evaluated Hamiltonian array."""
		physical: np.ndarray = self.physical(history)
		momentum: np.ndarray | None = self.momentum(history)
		if momentum is None:
			return physical, {}
		particle_times: np.ndarray = self.time(history)
		tolerance: float = float(16 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(times)))))
		if not np.allclose(particle_times, np.broadcast_to(times, momentum.shape), rtol=0.0, atol=tolerance):
			raise ValueError("Each particle time must match the output times within round-off.")
		return physical, self.diagnose_history(times, physical, {"extended_momentum": momentum}, energy=energy)


@dataclass(frozen=True)
class DoubleFormulation(SingleFormulation):
	"""Two complete physical copies with a single optional ``(t, kappa)`` pair.

	For N particles with d physical components, dimensions are 2*d*N without
	tracking and (2*d+2)*N with tracking. GC gives 4N/6N and FC gives 8N/10N.
	Method-specific projection and implicit solvers are separate operations.
	"""

	copies: ClassVar[int] = 2


__all__ = ["SingleFormulation", "DoubleFormulation"]
