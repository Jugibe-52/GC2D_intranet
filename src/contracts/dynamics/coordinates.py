"""Structural physical-coordinate declarations, independent of numerical methods."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StateLayout(Protocol):
	"""Describe component order and physical roles without inheriting a model."""

	@property
	def state_dimension(self) -> int: ...
	def snapshot(self) -> StateLayout:
		"""Return an immutable NumPy coordinate convention independent of the source."""
		...
	def validate_packed_state_layout(self, state: Any) -> Any: ...
	def split(self, state: Any) -> tuple[Any, ...]: ...
	def particle_count(self, state: Any) -> int: ...
	def positions(self, state: Any) -> tuple[Any, Any]: ...


__all__ = ['StateLayout']
