"""Semi-implicit ABBA integration with exact physical tangent propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .abba import SymmetricProjectedABBA


@dataclass(frozen=True, slots=True)
class SemiImplicitABBA(SymmetricProjectedABBA):
	"""Advance projected ABBA states and their exact ideal-root tangent.

	The physical trajectory is identical to ``SymmetricProjectedABBA``. After each
	converged projection solve, this variant evaluates the exact state Jacobian by
	the implicit-function formula, propagates the accumulated tangent from the
	initial state, and provides the local matrix to the optional step observer.
	"""

	_exact_tangent: ClassVar[bool] = True


__all__ = ["SemiImplicitABBA"]
