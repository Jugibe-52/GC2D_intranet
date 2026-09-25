"""Structural contracts for initial states and packed state layouts."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class StateLayout(Protocol):
	"""Interpret component-major planar particle states independently of their source."""

	state_dimension: ClassVar[int]

	def validate_packed_state_layout(self, state: np.ndarray) -> np.ndarray:
		"""Validate and return one packed state or state-history layout."""

	def split(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
		"""Return physical component blocks, preserving trailing sample axes."""

	def particle_count(self, state: np.ndarray) -> int:
		"""Return the number of represented particles."""

	def positions(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Return both physical position blocks."""


@runtime_checkable
class InitialConfiguration(Protocol):
	"""Provide an optional initial physical state and its independent layout."""

	@property
	def initial_state(self) -> np.ndarray | None:
		"""Return an independent initial-state copy, or ``None`` when unset."""

	@property
	def layout(self) -> StateLayout:
		"""Return the layout used to interpret the physical state."""


__all__ = ["InitialConfiguration", "StateLayout"]
