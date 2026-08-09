"""Structural contract for initial state and component-layout providers."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class InitialConfiguration(Protocol):
	"""Initial physical state with operations for its packed component layout."""

	state_dimension: ClassVar[int]

	@property
	def initial_state(self) -> np.ndarray | None:
		"""Return an independent initial-state copy, or ``None`` when unset."""

	def validate_packed_state(self, state: np.ndarray) -> np.ndarray:
		"""Validate and return one packed state or state history."""

	def split(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
		"""Return physical component blocks, preserving trailing sample axes."""

	@classmethod
	def pack_components(cls, *components: np.ndarray) -> np.ndarray:
		"""Pack equally shaped physical blocks into the public state layout."""

	def particle_count(self, state: np.ndarray) -> int:
		"""Return the number of represented particles."""

	def positions(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Return both physical position blocks."""


__all__ = ["InitialConfiguration"]
