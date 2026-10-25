"""Structural contracts consumed by methods.extended.core.records."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING
from contracts.arrays import BackendArray
if TYPE_CHECKING:
	from methods.extended.core.records import ProjectedMapResult


class ProjectedMap(Protocol):
	"""One physical projection with explicitly scoped iteration observation."""

	def __call__(
		self, t: float, state: BackendArray, h: float, *,
		observe: bool = False,
	) -> ProjectedMapResult:
		"""Evaluate a map; diagnostic callers leave iteration observation disabled."""


__all__ = ['ProjectedMap']
