"""Public order-4 implicit ABBA configuration on the shared family runtime."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Literal
from methods.extended.abba import _ABBAImplicitMethod
from methods.extended.configuration import ProjectionPlacement, _validate_projection_placement


@dataclass(slots=True)
class ABBA4Implicit(_ABBAImplicitMethod):
	"""Fourth-order triple jump with one outer symmetric Hairer projection.

	One complete step applies signed substeps ``(gamma h, delta h, gamma h)``.
	Both copies remain separate through all three base maps. One multiplier
	shifts the input and output of the whole composition. The placement
	keyword accepts only this construction; per-map ABBA4 projection is removed.
	"""

	projection_placement: ProjectionPlacement = "around_complete_composition"

	order: ClassVar[Literal[2, 4, 6]] = 4

	def __post_init__(self) -> None:
		"""Validate shared solver options and the ABBA4 placement selector."""
		_ABBAImplicitMethod.__post_init__(self)
		self.projection_placement = _validate_projection_placement(self.projection_placement)


__all__ = [
	"ABBA4Implicit",
]
