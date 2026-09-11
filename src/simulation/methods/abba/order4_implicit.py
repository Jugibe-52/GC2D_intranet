"""Public order-4 implicit ABBA configuration on the shared family runtime."""
from __future__ import annotations
from dataclasses import dataclass
from ..._result import IntegrationData
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from ._implicit import _ABBAImplicitConfig
from .preparation import prepare_abba
from .runtime import integrate_abba
import warnings
from typing import Any
from ._coefficients import _ABBA4_COEFFICIENTS as _ABBA4_COEFFICIENTS
from ._configuration import ProjectionPlacement, _validate_projection_placement
from .composition import (
    _ComposedABBAStep as _ComposedABBAStep,
    _solve_composed_abba_step as _solve_composed_abba_step,
    _solve_abba4_step as _solve_abba4_step,
)


@dataclass(frozen=True, slots=True)
class ABBA4Implicit(_ABBAImplicitConfig):
	"""Fourth-order triple jump with configurable projection placement.

	One complete step applies signed substeps ``(gamma h, delta h, gamma h)``.
	The selected placement either projects every signed ABBA map independently or
	keeps both copies separate through the complete composition and projects once
	around it. These placements define distinct numerical maps while sharing one
	public configuration type.
	"""

	projection_placement: ProjectionPlacement = "after_each_abba_map"

	def __post_init__(self) -> None:
		"""Validate shared solver options and the ABBA4 placement selector."""
		_ABBAImplicitConfig.__post_init__(self)
		object.__setattr__(
			self,
			"projection_placement",
			_validate_projection_placement(self.projection_placement),
		)

	def integrate(
		self,
		problem: InitialValueProblem,
		request: SimulationRequest,
	) -> IntegrationData:
		"""Prepare the selected step recipe and run the shared ABBA coordinator."""
		prepared = prepare_abba(problem, self, request, order=4,
			projection_placement=self.projection_placement)
		return integrate_abba(prepared, request)


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
