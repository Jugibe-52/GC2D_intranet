"""Public order-4 implicit ABBA configuration on the shared family runtime."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Literal
from methods.extended.abba import _ABBAImplicitMethod
import warnings
from typing import Any
from methods.extended.configuration import ProjectionPlacement, _validate_projection_placement


@dataclass(slots=True)
class ABBA4Implicit(_ABBAImplicitMethod):
	"""Fourth-order triple jump with one outer symmetric Hairer projection.

	One complete step applies signed substeps ``(gamma h, delta h, gamma h)``.
	Both copies remain separate through all three base maps. One multiplier
	shifts the input and output of the whole composition. The legacy placement
	keyword accepts only this construction; per-map ABBA4 projection is removed.
	"""

	projection_placement: ProjectionPlacement = "around_complete_composition"

	order: ClassVar[Literal[2, 4, 6]] = 4

	def __post_init__(self) -> None:
		"""Validate shared solver options and the ABBA4 placement selector."""
		_ABBAImplicitMethod.__post_init__(self)
		self.projection_placement = _validate_projection_placement(self.projection_placement)



def ABBA4ImplicitSingleProjection(
	*args: Any,
	**kwargs: Any,
) -> ABBA4Implicit:
	"""Build the deprecated single-outer-projection ABBA4 configuration."""
	warnings.warn(
		"ABBA4ImplicitSingleProjection is deprecated; use ABBA4Implicit("
		"projection_placement='around_complete_composition') instead.",
		DeprecationWarning,
		stacklevel=2,
	)
	placement = kwargs.pop(
		"projection_placement",
		"around_complete_composition",
	)
	if placement != "around_complete_composition":
		raise ValueError(
			"ABBA4ImplicitSingleProjection requires "
			"projection_placement='around_complete_composition'."
		)
	kwargs["projection_placement"] = "around_complete_composition"
	return ABBA4Implicit(*args, **kwargs)


__all__ = ["ABBA4Implicit", "ABBA4ImplicitSingleProjection"]
