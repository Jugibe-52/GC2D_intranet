"""Structural contracts consumed by studies.poincare_automation."""

from __future__ import annotations

from typing import Literal, TypedDict


class _NewtonControls(TypedDict):
	newton_absolute_tolerance: float
	newton_relative_tolerance: float
	newton_max_iterations: int
	newton_jacobian_relative_step: float
	newton_jacobian_method: Literal["analytic"]


__all__ = []
