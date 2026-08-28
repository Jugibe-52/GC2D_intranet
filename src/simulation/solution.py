"""Immutable physical trajectory returned by every simulation method."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from ._result import DiagnosticValue
from .configuration import InitialConfiguration, StateLayout


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


class Solution:
	"""Immutable sampled physical trajectory and named numerical diagnostics.

	``t`` has shape ``(saved_times,)`` and ``states`` has shape
	``(physical_state_size, saved_times)``. ``source`` owns the initial-state
	layout used to interpret each state column. Diagnostic arrays are copied and
	marked read-only when the result is created.

	Read-only ``y``, ``trajectory``, ``n_steps``, ``k`` and ``err`` properties
	remain available while versioned experiment notebooks migrate to canonical
	names.
	"""

	__slots__ = ("_diagnostics", "_source", "_states", "_t")

	def __init__(
		self,
		*,
		t: np.ndarray,
		states: np.ndarray,
		source: InitialConfiguration,
		diagnostics: Mapping[str, DiagnosticValue] | None = None,
	) -> None:
		"""Validate, own and freeze one complete physical result."""
		if not isinstance(source, InitialConfiguration):
			raise TypeError("`source` must implement InitialConfiguration.")
		layout = source.layout
		if not isinstance(layout, StateLayout):
			raise TypeError("`source.layout` must implement StateLayout.")
		times = _readonly_array(t, dtype=float)
		state_history = _readonly_array(states)
		if (
			times.ndim != 1
			or times.size < 2
			or not np.all(np.isfinite(times))
			or np.any(np.diff(times) <= 0)
			or state_history.ndim != 2
			or state_history.shape != (state_history.shape[0], times.size)
			or state_history.shape[0] == 0
			or not np.all(np.isfinite(state_history))
		):
			raise ValueError(
				"`t` and `states` must be finite arrays with shapes "
				"(T,) and (state_size, T), with increasing times."
			)
		layout.validate_packed_state_layout(state_history)
		normalized_diagnostics: dict[str, DiagnosticValue] = {}
		for name, value in dict(diagnostics or {}).items():
			if not isinstance(name, str) or not name:
				raise ValueError("Diagnostic names must be non-empty strings.")
			normalized_diagnostics[name] = (
				_readonly_array(value) if isinstance(value, np.ndarray) else value
			)
		self._t = times
		self._states = state_history
		self._source = source
		self._diagnostics = MappingProxyType(normalized_diagnostics)

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
		"""Initial configuration that defines the physical state layout."""
		return self._source

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
		selected = self.source.layout if layout is None else layout
		if not isinstance(selected, StateLayout):
			raise TypeError("`layout` must implement StateLayout.")
		if selected.state_dimension != self.source.layout.state_dimension:
			raise TypeError("The supplied layout is incompatible with this solution.")
		return selected.split(self.states)

	def positions(self) -> tuple[np.ndarray, np.ndarray]:
		"""Return both physical position histories."""
		return self.source.layout.positions(self.states)


__all__ = ["Solution"]
