"""Simultaneous nonlinear formulation of implicit symmetric-projected ABBA."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from ._implicit_abba import _ImplicitABBA
from ._projected_abba import (
	_ProjectedStep,
	_solve_simultaneous_projected_step,
)


@dataclass(frozen=True, slots=True)
class ImplicitABBA2(_ImplicitABBA):
	"""Implicit ABBA formulation 2 using simultaneous equation (21).

	Each Newton iteration solves for the duplicated output state and projection
	multiplier together in one exact six-dimensional system per independent
	guiding-centre particle.
	"""

	_step_solver: ClassVar[Callable[..., _ProjectedStep]] = (
		_solve_simultaneous_projected_step
	)
	_solver_formulation: ClassVar[str] = "implicit_2_simultaneous_equation_21"


__all__ = ["ImplicitABBA2"]
