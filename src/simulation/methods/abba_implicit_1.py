"""Reduced nonlinear formulation of implicit symmetric-projected ABBA."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from ._implicit_abba import _ImplicitABBA
from ._projected_abba import _ProjectedStep, _solve_projected_step


@dataclass(frozen=True, slots=True)
class ImplicitABBA1(_ImplicitABBA):
	"""Implicit ABBA formulation 1 using the reduced multiplier equation.

	Each Newton iteration solves the exact two-dimensional equation (11) for
	the projection multiplier of every independent guiding-centre particle.
	"""

	_step_solver: ClassVar[Callable[..., _ProjectedStep]] = _solve_projected_step
	_solver_formulation: ClassVar[str] = "implicit_1_reduced_equation_11"


__all__ = ["ImplicitABBA1"]
