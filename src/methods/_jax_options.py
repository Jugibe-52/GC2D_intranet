"""Hashable family-specific controls resolved before JAX traces a time loop."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from solvers._jax import NonlinearOptions

from methods.extended.core.composition import Composition


@dataclass(frozen=True)
class ExplicitOptions:
	"""Explicit physical maps need only the optional quadrature selection."""

	capture_quadrature: bool
	copies: ClassVar[int] = 1


@dataclass(frozen=True)
class ImplicitOptions:
	"""Classical implicit methods carry one resolved nonlinear configuration."""

	capture_quadrature: bool
	solve: NonlinearOptions
	copies: ClassVar[int] = 1


@dataclass(frozen=True)
class CompositionOptions:
	"""A signed recipe with arithmetic averaging or one outer nonlinear solve."""

	capture_quadrature: bool
	recipe: Composition
	coupling: float | None
	projection: Literal["reduced_multiplier", "simultaneous_state_multiplier"]
	solve: NonlinearOptions | None
	publish_substeps: bool = False
	copies: ClassVar[int] = 2


__all__: list[str] = []
