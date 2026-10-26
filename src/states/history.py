"""Physical samples recorded during a run and exposed as its completed result."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from contracts.dynamics.coordinates import StateLayout
from dynamics.coordinates import PhysicalCoordinates, describe_coordinates
from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepValue


def _readonly(value: np.ndarray) -> np.ndarray:
	owned: np.ndarray = np.array(value, copy=True)
	owned.setflags(write=False)
	return owned


def _prepared_recording_inputs(
	times: np.ndarray, layout: StateLayout, state_size: int, initial: np.ndarray | None,
) -> tuple[np.ndarray, PhysicalCoordinates]:
	"""Own the ordered schedule and validate the independent packed layout."""
	if not isinstance(layout, StateLayout):
		raise TypeError("`layout` must implement StateLayout.")
	t: np.ndarray = np.array(times, dtype=float, copy=True)
	if t.ndim != 1 or not t.size or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
		raise ValueError("Sample times must be a finite non-empty increasing vector.")
	owned_layout: PhysicalCoordinates = describe_coordinates(layout)
	owned_layout.validate_packed_state_layout(
		np.asarray(initial) if initial is not None else np.zeros(state_size)
	)
	return t, owned_layout


def _validated_history(values: np.ndarray, shape: tuple[int, ...], layout: StateLayout) -> np.ndarray:
	"""Validate a complete physical history without converting its dtype."""
	value: np.ndarray = np.asarray(values)
	if value.shape != shape or not np.all(np.isfinite(value)):
		raise ValueError("Physical states must be finite with shape (samples, state_size).")
	layout.validate_packed_state_layout(value.T)
	return value


def _owned_initial_snapshot(initial: np.ndarray | None, state_size: int) -> np.ndarray | None:
	"""Copy the optional physical snapshot independently of recorded samples."""
	value: None | np.ndarray = None if initial is None else _readonly(np.asarray(initial))
	if value is not None and (value.shape != (state_size,) or not np.all(np.isfinite(value))):
		raise ValueError("The initial snapshot must match the physical state dimension.")
	return value


def _owned_diagnostics(diagnostics: Mapping[str, DiagnosticValue]) -> dict[str, DiagnosticValue]:
	"""Validate all entries before freezing independent diagnostic arrays."""
	for name, value in diagnostics.items():
		if not isinstance(name, str) or not name:
			raise ValueError("Diagnostic names must be non-empty strings.")
		if not isinstance(value, (np.ndarray, float, int, str, bool, np.number)):
			raise TypeError("Diagnostics must be scalar values or NumPy arrays.")
	return {
		key: _readonly(value) if isinstance(value, np.ndarray) else value
		for key, value in diagnostics.items()
	}


class States:
	"""Own physical rows and timestamps, first recording and then read-only.

	``prepare`` reserves a schedule; ``record`` fills its rows in order.
	Only recorded rows are exposed. ``finalize`` rejects incomplete output
	and seals the same object. Direct construction accepts complete arrays,
	including archives and device results, and produces an already sealed result.
	The class knows neither numerical methods nor internal coordinate extensions.
	"""

	def __init__(self, t: np.ndarray, states: np.ndarray,
	             diagnostics: Mapping[str, DiagnosticValue] | None = None, *,
	             layout: StateLayout, initial_state: np.ndarray | None = None) -> None:
		self._allocate(t, layout, np.asarray(states).dtype,
		               np.asarray(states).shape[1] if np.ndim(states) == 2 else 0, initial_state)
		value: np.ndarray = _validated_history(states, self._values.shape, self._layout)
		self._values[:] = value
		self._count = self._times.size
		self.finalize(diagnostics)

	def _allocate(self, times: np.ndarray, layout: StateLayout, dtype: Any,
	              state_size: int, initial: np.ndarray | None) -> None:
		t: np.ndarray
		owned_layout: PhysicalCoordinates
		t, owned_layout = _prepared_recording_inputs(times, layout, state_size, initial)
		self._times = t
		self._values = np.empty((t.size, state_size), dtype=dtype)
		self._layout = owned_layout
		self._count = 0
		self._complete = False
		self._initial = _owned_initial_snapshot(initial, state_size)
		self._diagnostics: dict[str, DiagnosticValue] = {}
		self._auxiliary: dict[str, np.ndarray] = {}
		self._rows: dict[str, list[StepValue]] = {}
		self._shapes: dict[str, tuple[int, ...]] = {}
		self._starts: list[float] = []
		self._ends: list[float] = []
		self._sizes: list[float] = []

	@classmethod
	def prepare(cls, times: np.ndarray, initial: np.ndarray, *, layout: StateLayout) -> States:
		"""Reserve one result without publishing any unrecorded rows."""
		value: np.ndarray = np.asarray(initial, dtype=float)
		if value.ndim != 1 or not value.size or not np.all(np.isfinite(value)):
			raise ValueError("The initial physical state must be a finite non-empty vector.")
		result: States = cls.__new__(cls)
		result._allocate(times, layout, value.dtype, value.size, value)
		return result

	def _require_recording(self) -> None:
		if self._complete:
			raise RuntimeError("Completed States cannot be modified.")

	def record(self, time: float, values: np.ndarray, *,
	           auxiliary: Mapping[str, np.ndarray] | None = None) -> None:
		"""Copy the next physical sample and its optional auxiliary data."""
		self._require_recording()
		if self._count >= self._times.size or time != self._times[self._count]:
			raise ValueError("Record samples in the requested time order.")
		value: np.ndarray = np.asarray(values)
		if value.shape != (self._values.shape[1],) or not np.all(np.isfinite(value)):
			raise ValueError("A physical sample changed shape or became non-finite.")
		parts: dict[str, np.ndarray] = dict(auxiliary or {})
		if self._count == 0:
			self._auxiliary = {key: np.empty((*np.shape(part), self._times.size), dtype=np.asarray(part).dtype)
			                   for key, part in parts.items()}
		if parts.keys() != self._auxiliary.keys():
			raise ValueError("Sample auxiliary names changed during integration.")
		for key, part in parts.items():
			if np.shape(part) != self._auxiliary[key].shape[:-1] or not np.all(np.isfinite(part)):
				raise ValueError("Sample auxiliary data changed shape or became non-finite.")
		self._values[self._count] = value
		for key, part in parts.items():
			self._auxiliary[key][..., self._count] = part
		self._count += 1

	def record_history(self, values: np.ndarray) -> None:
		"""Copy a complete physical history into an empty, still-open recorder.

		The execution boundary supplies shape (samples, state_size) in bulk.
		Diagnostics are supplied at finalization; this operation never seals the
		result or overwrites samples already recorded by a streaming controller.
		"""
		self._require_recording()
		if self._count:
			raise RuntimeError("Bulk history recording requires an empty States recorder.")
		value: np.ndarray = _validated_history(values, self._values.shape, self._layout)
		self._values[:] = value
		self._count = self._times.size

	def record_step(self, info: StepInfo, statistics: Mapping[str, StepValue]) -> None:
		"""Record main-step metrics on their own axis, independently of samples."""
		self._require_recording()
		if not self._starts:
			self._rows = {key: [] for key in statistics}
			self._shapes = {key: np.shape(value) for key, value in statistics.items()}
		if statistics.keys() != self._rows.keys():
			raise ValueError("Step statistic names changed during integration.")
		for key, value in statistics.items():
			if np.shape(value) != self._shapes[key]:
				raise ValueError(f"The shape of step statistic {key!r} changed.")
		for key, value in statistics.items():
			self._rows[key].append(value.copy() if isinstance(value, np.ndarray) else value)
		self._starts.append(info.time)
		self._ends.append(info.end_time)
		self._sizes.append(info.duration)

	def finalize(self, diagnostics: Mapping[str, DiagnosticValue] | None = None, *,
	             aliases: Mapping[str, str] | None = None) -> States:
		"""Validate completeness and seal this result, without another container."""
		self._require_recording()
		if self._count != self._times.size:
			raise RuntimeError("Cannot finalize incomplete States.")
		combined: dict[str, DiagnosticValue] = dict(diagnostics or {})
		if self._starts:
			combined.update({key: np.asarray(values) for key, values in self._rows.items()})
			combined.update(step_count=len(self._starts), step_start_times=np.asarray(self._starts),
			                step_times=np.asarray(self._ends), step_sizes=np.asarray(self._sizes))
		if aliases and all(name in combined for name in aliases.values()):
			combined.update({alias: combined[name] for alias, name in aliases.items()})
		self._diagnostics = _owned_diagnostics(combined)
		self._values.setflags(write=False)
		self._times.setflags(write=False)
		self._auxiliary = {}  # Completed diagnostics own the saved auxiliary data.
		self._rows = {}
		self._shapes = {}
		self._starts = []
		self._ends = []
		self._sizes = []
		self._complete = True
		return self

	def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
		if not self.complete:
			raise TypeError("Only completed States can cross an execution boundary.")
		return (_restore_states, (self.times, self.values, self._diagnostics, self._layout, self._initial, 2))

	@property
	def complete(self) -> bool:
		return self._complete

	@property
	def times(self) -> np.ndarray:
		return self._times if self._complete else _readonly(self._times[:self._count])

	@property
	def values(self) -> np.ndarray:
		"""Read-only physical rows: values[i] is the state at times[i]."""
		return self._values if self._complete else _readonly(self._values[:self._count])

	@property
	def t(self) -> np.ndarray:
		"""Read-only time access retained for scientific plotting consumers."""
		return self.times

	@property
	def states(self) -> np.ndarray:
		"""Read-only packed-history access retained for array consumers."""
		return self.values

	@property
	def layout(self) -> PhysicalCoordinates:
		return self._layout

	@property
	def initial_state(self) -> np.ndarray | None:
		return None if self._initial is None else self._initial.copy()

	@property
	def particle_count(self) -> int:
		return self._values.shape[1] // self._layout.state_dimension

	@property
	def auxiliary(self) -> Mapping[str, np.ndarray]:
		"""Independent recorded auxiliaries for output energy diagnostics."""
		return MappingProxyType({key: _readonly(value[..., :self._count]) for key, value in self._auxiliary.items()})

	@property
	def diagnostics(self) -> Mapping[str, DiagnosticValue]:
		return MappingProxyType(self._diagnostics)

	def components(self, layout: StateLayout | None = None) -> tuple[np.ndarray, ...]:
		"""Component views of shape (particles, samples) for vectorized physics."""
		selected = self._layout if layout is None else layout
		if not isinstance(selected, StateLayout) or describe_coordinates(selected) != self._layout:
			raise TypeError("The supplied layout is incompatible with these states.")
		return selected.split(self.values.T)

	def split(self) -> tuple[np.ndarray, ...]:
		return self.components()

	def positions(self) -> tuple[np.ndarray, np.ndarray]:
		return self._layout.positions(self.values.T)

	def velocities(self) -> tuple[np.ndarray, np.ndarray]:
		try:
			return self._layout.velocities(self.values.T)
		except ValueError as error:
			# Preserve the result accessor's error for conventions without velocities.
			raise TypeError(str(error)) from error

	def select(self, indices: Any) -> States:
		"""Select increasing samples, retaining their original times and layout.

	Whole-run diagnostics remain attached unchanged; they have independent axes.
	"""
		chosen: np.ndarray = np.atleast_1d(np.arange(self._count)[indices])
		return States(self.times[chosen], self.values[chosen], self.diagnostics, layout=self._layout, initial_state=self._initial)

	def with_diagnostics(self, diagnostics: Mapping[str, DiagnosticValue], *, replace: bool = False) -> States:
		"""Return a sealed copy with execution metadata added by a boundary."""
		if not self.complete:
			raise RuntimeError("Execution metadata requires a complete result.")
		return States(self.times, self.values, dict(diagnostics) if replace else {**self.diagnostics, **diagnostics}, layout=self._layout, initial_state=self._initial)

	@property
	def n_steps(self) -> int:
		"""Number of accepted main steps, independent of the sample count."""
		return int(self._diagnostics.get("step_count", 0))

	@property
	def k(self) -> np.ndarray | None:
		value: np.ndarray | float | int | str | None = self._diagnostics.get("extended_momentum")
		return None if value is None else np.asarray(value)

	@property
	def err(self) -> float | None:
		value: np.ndarray | float | int | str | None = self._diagnostics.get("energy_error")
		return None if value is None else float(value)


def _restore_states(times: np.ndarray, values: np.ndarray, diagnostics: dict[str, DiagnosticValue],
                    layout: StateLayout, initial: np.ndarray | None, history_version: int = 1) -> States:
	"""Decode pre-row pickles explicitly; square histories cannot reveal their axes."""
	if history_version not in (1, 2):
		raise ValueError("Unsupported pickled States history version.")
	return States(times, values.T if history_version == 1 else values, diagnostics, layout=layout, initial_state=initial)


__all__ = ["States"]
