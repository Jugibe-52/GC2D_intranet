"""Simultaneous nonlinear formulation of implicit symmetric-projected ABBA."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from ._implicit import _ImplicitABBA
from ._projection import (
	_ProjectedStep,
	_solve_simultaneous_projected_step,
)


@dataclass(frozen=True, slots=True)
class ImplicitABBA2(_ImplicitABBA):
	"""Implicit ABBA formulation 2 using simultaneous equation (21).

	The duplicated output and projection multiplier are simultaneous unknowns.
	Newton uses exact six-dimensional particle blocks; Broyden updates an
	asymptotically initialized Jacobian from explicit residual evaluations.
	"""

	_step_solver: ClassVar[Callable[..., _ProjectedStep]] = (
		_solve_simultaneous_projected_step
	)
	_solver_formulation: ClassVar[str] = "simultaneous_state_multiplier"


__all__ = ["ImplicitABBA2"]
