"""Explicit custom coordinate conventions for structural integration tests."""

from typing import Any
from dynamics.coordinates import PhysicalCoordinates


class ThreeCoordinates(PhysicalCoordinates):
	"""Three blocks with the first two interpreted as planar positions."""
	__slots__ = ()

	@property
	def state_dimension(self) -> int:
		return 3

	def positions(self, state: Any) -> tuple[Any, Any]:
		blocks = self.split(state)
		return blocks[0], blocks[1]


class UnclassifiedCoordinates(PhysicalCoordinates):
	"""Two algebraic components without physical position or velocity roles."""
	__slots__ = ()

	@property
	def state_dimension(self) -> int:
		return 2


class ReorderedCoordinates(UnclassifiedCoordinates):
	"""Two position blocks stored in y, x order, explicitly unlike GC."""
	__slots__ = ()

	def positions(self, state: Any) -> tuple[Any, Any]:
		blocks = self.split(state)
		return blocks[1], blocks[0]


class FiveCoordinates(PhysicalCoordinates):
	"""Five blocks interpreted directly as y, x, vy, vx and an extra value."""
	__slots__ = ()

	@property
	def state_dimension(self) -> int:
		return 5

	def positions(self, state: Any) -> tuple[Any, Any]:
		blocks = self.split(state)
		return blocks[1], blocks[0]

	def velocities(self, state: Any) -> tuple[Any, Any]:
		blocks = self.split(state)
		return blocks[3], blocks[2]
