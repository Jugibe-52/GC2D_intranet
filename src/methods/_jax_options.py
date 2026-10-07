"""Hashable family-specific controls resolved before JAX traces a time loop."""

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import jax.numpy as jnp


@dataclass(frozen=True)
class NonlinearOptions:
	"""Stopping and Jacobian controls for one physical nonlinear problem."""

	atol: float
	rtol: float
	max_iterations: int
	jacobian: Literal["analytic", "finite_difference"]
	relative_step: float
	solver: Literal["newton", "broyden"] = "newton"

	def tolerance(self, state: Any) -> Any:
		"""Preserve the CPU's global physical-state infinity-norm scaling."""
		return self.atol + self.rtol * jnp.maximum(1., jnp.max(jnp.abs(state)))


@dataclass(frozen=True)
class ExplicitOptions:
	"""Explicit physical maps need only the passive-energy selection."""

	track_energy: bool
	copies: ClassVar[int] = 1


@dataclass(frozen=True)
class ImplicitOptions:
	"""Classical implicit methods carry one resolved nonlinear configuration."""

	track_energy: bool
	solve: NonlinearOptions
	copies: ClassVar[int] = 1


@dataclass(frozen=True)
class CompositionOptions:
	"""A signed recipe with arithmetic averaging or one outer nonlinear solve."""

	track_energy: bool
	coefficients: tuple[float, ...]
	coupling: float | None
	projection: Literal["reduced_multiplier", "simultaneous_state_multiplier"]
	solve: NonlinearOptions | None
	publish_substeps: bool = False
	copies: ClassVar[int] = 2


__all__: list[str] = []
