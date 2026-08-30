"""Simultaneous output-multiplier formulation of Hairer-projected BM4."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from ._implicit import (
	_ImplicitBM4,
	_ProjectedBM4Step,
	_solve_simultaneous_projected_bm4_step,
)


@dataclass(frozen=True, slots=True)
class BM4Implicit2(_ImplicitBM4):
	"""BM4 with a Newton or Broyden simultaneous projection solve."""

	_step_solver: ClassVar[Callable[..., _ProjectedBM4Step]] = (
		_solve_simultaneous_projected_bm4_step
	)
	_solver_formulation: ClassVar[str] = "bm4_implicit_2_simultaneous"


__all__ = ["BM4Implicit2"]
