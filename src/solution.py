"""Immutable physical trajectory returned by every simulation method."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType

import numpy as np

from contracts.result import DiagnosticValue
from contracts.configuration import InitialConfiguration, StateLayout


def _readonly_array(
	value: np.ndarray,
	*,
	dtype: type[float] | None = None,
) -> np.ndarray:
	"""Own one NumPy value and expose it through a read-only array."""
	result = (
		np.array(value, copy=True)
		if dtype is None
		else np.array(value, dtype=dtype, copy=True)
	)
	result.setflags(write=False)
	return result


def _validated_history(
	t: np.ndarray,
	states: np.ndarray,
	layout: StateLayout,
) -> tuple[np.ndarray, np.ndarray]:
	"""Own finite ``(T,)`` times and ``(state_size, T)`` states for one layout."""
	if not isinstance(layout, StateLayout):
		raise TypeError("`layout` must implement StateLayout.")
	times = _readonly_array(t, dtype=float)
	state_history = _readonly_array(states)
	if (
		times.ndim != 1
		or times.size < 2
		or not np.all(np.isfinite(times))
		or np.any(np.diff(times) <= 0)
		or state_history.ndim != 2
		or state_history.shape[1] != times.size
		or state_history.shape[0] == 0
		or not np.all(np.isfinite(state_history))
	):
		raise ValueError(
			"`t` and `states` must be finite arrays with shapes "
			"(T,) and (state_size, T), with increasing times."
		)
	layout.validate_packed_state_layout(state_history)
	return times, state_history


def _readonly_diagnostics(
	diagnostics: Mapping[str, DiagnosticValue] | None,
) -> Mapping[str, DiagnosticValue]:
	"""Validate diagnostic names and own read-only copies of their array values."""
	normalized: dict[str, DiagnosticValue] = {}
	for name, value in dict(diagnostics or {}).items():
		if not isinstance(name, str) or not name:
			raise ValueError("Diagnostic names must be non-empty strings.")
		normalized[name] = (
			_readonly_array(value) if isinstance(value, np.ndarray) else value
		)
	return MappingProxyType(normalized)


class Solution:
	"""Immutable sampled physical trajectory and named numerical diagnostics.

	``t`` has shape ``(saved_times,)`` and ``states`` has shape
	``(physical_state_size, saved_times)``. ``source`` preserves initial-condition
	provenance; an independent layout snapshot interprets the result. The optional
	initial state and diagnostic arrays are copied when the result is created.

	Read-only ``y``, ``trajectory``, ``n_steps``, ``k`` and ``err`` properties
	remain available while versioned experiment notebooks migrate to canonical
	names.
	"""

	__slots__ = ("_diagnostics", "_source", "_states", "_t", "_layout", "_initial_state")

	def __init__(
		self,
		*,
		t: np.ndarray,
		states: np.ndarray,
		source: InitialConfiguration,
		diagnostics: Mapping[str, DiagnosticValue] | None = None,
		layout: StateLayout | None = None,
		initial_state: np.ndarray | None = None,
	) -> None:
		"""Validate, own and freeze one complete physical result."""
		if not isinstance(source, InitialConfiguration):
			raise TypeError("`source` must implement InitialConfiguration.")
		layout_snapshot = deepcopy(source.layout if layout is None else layout)
		times, state_history = _validated_history(t, states, layout_snapshot)
		initial = source.initial_state if initial_state is None else initial_state
		if initial is not None:
			initial = _readonly_array(initial, dtype=float)
			if initial.shape != (state_history.shape[0],) or not np.all(np.isfinite(initial)):
				raise ValueError("The initial-state snapshot must match the physical history.")
		normalized_diagnostics = _readonly_diagnostics(diagnostics)
		self._t = times
		self._states = state_history
		self._source = source
		self._layout = layout_snapshot
		self._initial_state = initial
		self._diagnostics = normalized_diagnostics

	@property
	def t(self) -> np.ndarray:
		"""Read-only saved times with shape ``(T,)``."""
		return self._t

	@property
	def states(self) -> np.ndarray:
		"""Read-only physical history with shape ``(state_size, T)``."""
		return self._states

	@property
	def source(self) -> InitialConfiguration:
		"""Original initial configuration retained as result provenance."""
		return self._source

	@property
	def layout(self) -> StateLayout:
		"""Independent copy of the physical layout captured for this trajectory."""
		return deepcopy(self._layout)

	@property
	def initial_state(self) -> np.ndarray | None:
		"""Independent initial-state snapshot, or None for histories without one."""
		return None if self._initial_state is None else self._initial_state.copy()

	@property
	def diagnostics(self) -> Mapping[str, DiagnosticValue]:
		"""Read-only method and formulation diagnostics."""
		return self._diagnostics

	@property
	def y(self) -> np.ndarray:
		"""Deprecated read-only alias for :attr:`states`."""
		return self.states

	@property
	def trajectory(self) -> InitialConfiguration:
		"""Deprecated read-only alias for :attr:`source`."""
		return self.source

	@property
	def n_steps(self) -> int:
		"""Deprecated view of the common step-count diagnostic."""
		return int(self.diagnostics.get("step_count", 0))

	@property
	def k(self) -> np.ndarray | None:
		"""Deprecated view of extended time-conjugate momentum."""
		value = self.diagnostics.get("extended_momentum")
		return None if value is None else np.asarray(value)

	@property
	def err(self) -> float | None:
		"""Deprecated view of maximum generalized-energy drift."""
		value = self.diagnostics.get("energy_error")
		return None if value is None else float(value)

	def components(
		self,
		layout: StateLayout | None = None,
	) -> tuple[np.ndarray, ...]:
		"""Return physical component blocks over the complete time history."""
		selected = self._layout if layout is None else layout
		if not isinstance(selected, StateLayout):
			raise TypeError("`layout` must implement StateLayout.")
		if selected.state_dimension != self._layout.state_dimension:
			raise TypeError("The supplied layout is incompatible with this solution.")
		return selected.split(self.states)

	def positions(self) -> tuple[np.ndarray, np.ndarray]:
		"""Return both physical position histories."""
		return self._layout.positions(self.states)


__all__ = ["Solution"]
