"""Public order-6 implicit ABBA configuration on the shared family runtime."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Literal
from methods.extended.abba import _ABBAImplicitMethod


@dataclass(slots=True)
class ABBA6Implicit(_ABBAImplicitMethod):
	"""Sixth-order symmetric composition of seven implicit ABBA maps.

	Each outer step applies Yoshida's palindromic seven-stage coefficients to
	complete ``ABBA2Implicit`` maps. Two substeps run backward in time. Every
	signed substep uses the same selected projection formulation, nonlinear
	solver, and state strategy, and solves an independent projection problem.
	Physical conjugate-momentum tracking is optional and triangular.
	"""

	order: ClassVar[Literal[2, 4, 6]] = 6


__all__ = ["ABBA6Implicit"]
