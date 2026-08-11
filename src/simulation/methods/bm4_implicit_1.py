"""Reduced multiplier formulation of Hairer-projected BM4."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from ._implicit_bm4 import (
	_ImplicitBM4,
	_ProjectedBM4Step,
	_solve_reduced_projected_bm4_step,
)


@dataclass(frozen=True, slots=True)
class BM4Implicit1(_ImplicitBM4):
	"""BM4 with a reduced symmetric-projection multiplier solve."""

	_step_solver: ClassVar[Callable[..., _ProjectedBM4Step]] = (
		_solve_reduced_projected_bm4_step
	)
	_solver_formulation: ClassVar[str] = "bm4_implicit_1_reduced"


__all__ = ["BM4Implicit1"]
