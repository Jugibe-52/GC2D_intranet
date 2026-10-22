"""Structural contracts consumed by methods.extended.core.records."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING
import numpy as np
from contracts.solvers.nonlinear import NewtonObserver
if TYPE_CHECKING:
	from methods.extended.core.records import ProjectedMapResult


class ProjectedMap(Protocol):
	"""One physical projection with an explicitly scoped Newton callback."""

	def __call__(
		self, t: float, state: np.ndarray, h: float, *,
		newton_observer: NewtonObserver | None = None,
	) -> ProjectedMapResult:
		"""Evaluate a map; diagnostic callers omit the optional callback."""


__all__ = ['ProjectedMap']
