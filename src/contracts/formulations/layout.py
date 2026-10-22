"""Structural interpretation of packed physical state components."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class StateLayout(Protocol):
	"""Interpret component-major planar particle states independently of their source.

	Layouts contain coordinate interpretation only, support ``deepcopy``, and do
	not own the initial state or physical parameters. A problem copies the layout
	when it snapshots its initial values. Built-in implementations live in
	``formulations.layout``; external implementations need no inheritance.
	"""

	state_dimension: ClassVar[int]

	def validate_packed_state_layout(self, state: np.ndarray) -> np.ndarray:
		"""Validate and return one packed state or state-history layout."""

	def split(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
		"""Return physical component blocks, preserving trailing sample axes."""

	def particle_count(self, state: np.ndarray) -> int:
		"""Return the number of represented particles."""

	def positions(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Return both physical position blocks."""


__all__ = ["StateLayout"]
