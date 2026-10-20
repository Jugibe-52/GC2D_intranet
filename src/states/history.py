"""Physical samples recorded during a run and exposed as its completed result."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import numpy as np

from contracts.configuration import StateLayout
from contracts.result import DiagnosticValue
from contracts.step import StepInfo, StepValue


def _readonly(value: np.ndarray) -> np.ndarray:
	owned = np.array(value, copy=True)
	owned.setflags(write=False)
	return owned


class States:
	"""Own physical columns and timestamps, first recording and then read-only.

	``prepare`` reserves a schedule; ``record`` fills its columns in order.
	Only recorded columns are exposed. ``finalize`` rejects incomplete output
	and seals the same object. Direct construction accepts complete arrays,
	including archives and device results, and produces an already sealed result.
	The class knows neither numerical methods nor internal coordinate extensions.
	"""

	def __init__(self, t: np.ndarray, states: np.ndarray,
	             diagnostics: Mapping[str, DiagnosticValue] | None = None, *,
	             layout: StateLayout, initial_state: np.ndarray | None = None) -> None:
		self._allocate(t, layout, np.asarray(states).dtype,
		               np.asarray(states).shape[0] if np.ndim(states) else 0, initial_state)
		value = np.asarray(states)
		if value.shape != self._values.shape or not np.all(np.isfinite(value)):
			raise ValueError("Physical states must be finite with shape (state_size, samples).")
		self._layout.validate_packed_state_layout(value)
		self._values[:] = value
		self._count = self._times.size
		self.finalize(diagnostics)

	def _allocate(self, times: np.ndarray, layout: StateLayout, dtype: Any,
	              state_size: int, initial: np.ndarray | None) -> None:
		if not isinstance(layout, StateLayout):
			raise TypeError("`layout` must implement StateLayout.")
		t = np.array(times, dtype=float, copy=True)
		if t.ndim != 1 or not t.size or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
			raise ValueError("Sample times must be a finite non-empty increasing vector.")
		owned_layout = deepcopy(layout)
		owned_layout.validate_packed_state_layout(
			np.asarray(initial) if initial is not None else np.zeros(state_size)
		)
		self._times = t
		self._values = np.empty((state_size, t.size), dtype=dtype)
		self._layout = owned_layout
		self._count = 0
		self._complete = False
		self._initial = None if initial is None else _readonly(np.asarray(initial))
		if self._initial is not None and (self._initial.shape != (state_size,) or not np.all(np.isfinite(self._initial))):
			raise ValueError("The initial snapshot must match the physical state dimension.")
		self._diagnostics: dict[str, DiagnosticValue] = {}
		self._auxiliary: dict[str, np.ndarray] = {}
		self._rows: dict[str, list[StepValue]] = {}
		self._shapes: dict[str, tuple[int, ...]] = {}
		self._starts: list[float] = []
		self._ends: list[float] = []
		self._sizes: list[float] = []

	@classmethod
	def prepare(cls, times: np.ndarray, initial: np.ndarray, *, layout: StateLayout) -> States:
		"""Reserve one result without publishing any unrecorded columns."""
		value = np.asarray(initial, dtype=float)
		if value.ndim != 1 or not value.size or not np.all(np.isfinite(value)):
			raise ValueError("The initial physical state must be a finite non-empty vector.")
		result = cls.__new__(cls)
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
		value = np.asarray(values)
		if value.shape != (self._values.shape[0],) or not np.all(np.isfinite(value)):
			raise ValueError("A physical sample changed shape or became non-finite.")
		parts = dict(auxiliary or {})
		if self._count == 0:
			self._auxiliary = {key: np.empty((*np.shape(part), self._times.size), dtype=np.asarray(part).dtype)
			                   for key, part in parts.items()}
		if parts.keys() != self._auxiliary.keys():
			raise ValueError("Sample auxiliary names changed during integration.")
		for key, part in parts.items():
			if np.shape(part) != self._auxiliary[key].shape[:-1] or not np.all(np.isfinite(part)):
				raise ValueError("Sample auxiliary data changed shape or became non-finite.")
		self._values[:, self._count] = value
		for key, part in parts.items():
			self._auxiliary[key][..., self._count] = part
		self._count += 1

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
		for name, value in combined.items():
			if not isinstance(name, str) or not name:
				raise ValueError("Diagnostic names must be non-empty strings.")
			if not isinstance(value, (np.ndarray, float, int, str, bool, np.number)):
				raise TypeError("Diagnostics must be scalar values or NumPy arrays.")
		self._diagnostics = {key: _readonly(value) if isinstance(value, np.ndarray) else value
		                     for key, value in combined.items()}
		self._values.setflags(write=False)
		self._times.setflags(write=False)
		self._auxiliary = {}  # Formulation diagnostics now own needed auxiliary data.
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
		return (_restore_states, (self.times, self.values, self._diagnostics, self._layout, self._initial))

	@property
	def complete(self) -> bool:
		return self._complete

	@property
	def times(self) -> np.ndarray:
		return self._times if self._complete else _readonly(self._times[:self._count])

	@property
	def values(self) -> np.ndarray:
		return self._values if self._complete else _readonly(self._values[:, :self._count])

	@property
	def t(self) -> np.ndarray:
		"""Read-only time access retained for scientific plotting consumers."""
		return self.times

	@property
	def states(self) -> np.ndarray:
		"""Read-only packed-history access retained for array consumers."""
		return self.values

	@property
	def layout(self) -> StateLayout:
		return deepcopy(self._layout)

	@property
	def initial_state(self) -> np.ndarray | None:
		return None if self._initial is None else self._initial.copy()

	@property
	def particle_count(self) -> int:
		return self._values.shape[0] // self._layout.state_dimension

	@property
	def auxiliary(self) -> Mapping[str, np.ndarray]:
		"""Independent recorded auxiliaries for formulation-level diagnostics."""
		return MappingProxyType({key: _readonly(value[..., :self._count]) for key, value in self._auxiliary.items()})

	@property
	def diagnostics(self) -> Mapping[str, DiagnosticValue]:
		return MappingProxyType(self._diagnostics)

	def components(self, layout: StateLayout | None = None) -> tuple[np.ndarray, ...]:
		selected = self._layout if layout is None else layout
		if not isinstance(selected, StateLayout) or selected.state_dimension != self._layout.state_dimension:
			raise TypeError("The supplied layout is incompatible with these states.")
		return selected.split(self.values)

	def split(self) -> tuple[np.ndarray, ...]:
		return self.components()

	def positions(self) -> tuple[np.ndarray, np.ndarray]:
		return self._layout.positions(self.values)

	def velocities(self) -> tuple[np.ndarray, np.ndarray]:
		components: Any = self.components()
		if not hasattr(components, 'vx') or not hasattr(components, 'vy'):
			raise TypeError("This layout does not define velocity components.")
		return components.vx, components.vy

	def select(self, indices: Any) -> States:
		"""Select increasing samples, retaining their original times and layout.

	Whole-run diagnostics remain attached unchanged; they have independent axes.
	"""
		chosen = np.atleast_1d(np.arange(self._count)[indices])
		return States(self.times[chosen], self.values[:, chosen], self.diagnostics, layout=self._layout, initial_state=self._initial)

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
		value = self._diagnostics.get("extended_momentum")
		return None if value is None else np.asarray(value)

	@property
	def err(self) -> float | None:
		value = self._diagnostics.get("energy_error")
		return None if value is None else float(value)


def _restore_states(times: np.ndarray, values: np.ndarray, diagnostics: dict[str, DiagnosticValue],
                    layout: StateLayout, initial: np.ndarray | None) -> States:
	return States(times, values, diagnostics, layout=layout, initial_state=initial)


__all__ = ["States"]
