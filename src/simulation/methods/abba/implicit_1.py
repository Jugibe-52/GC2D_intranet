"""Reduced nonlinear formulation of implicit symmetric-projected ABBA."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from ._implicit import _ImplicitABBA
from ._projection import _ProjectedStep, _solve_projected_step


@dataclass(frozen=True, slots=True)
class ImplicitABBA1(_ImplicitABBA):
	"""Implicit ABBA formulation 1 using the reduced multiplier equation.

	Newton uses the exact two-dimensional equation-(11) Jacobian for every
	independent particle. Broyden instead updates a shared residual-Jacobian
	approximation from explicit evaluations of the same reduced residual.
	"""

	_step_solver: ClassVar[Callable[..., _ProjectedStep]] = _solve_projected_step
	_solver_formulation: ClassVar[str] = "reduced_multiplier"


__all__ = ["ImplicitABBA1"]
