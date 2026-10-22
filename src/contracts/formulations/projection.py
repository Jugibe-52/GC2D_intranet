"""Structural contracts consumed by formulations.projection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Literal, Protocol, TypeAlias


ProjectionName: TypeAlias = Literal['reduced_multiplier', 'simultaneous_state_multiplier']


MapEvaluation: TypeAlias = Callable[[Any], tuple[Any, Any]]


class ProjectionFormulation(Protocol):
	"""Build one root problem from a doubled physical composition.

	Array operations preserve NumPy/JAX. The supplied map returns its doubled
	output and an opaque payload; equations do not interpret stage records.
	"""

	name: ClassVar[ProjectionName]
	unknown_components: ClassVar[int]

	def prepare(self, state: Any, mapping: MapEvaluation) -> tuple[Any, Any]: ...
	def multiplier(self, unknown: Any, size: int) -> Any: ...
	def residual(self, unknown: Any, mapped: Any, size: int) -> Any: ...
	def particle_jacobian(self, base: Any) -> Any: ...
	def dense_jacobian(self, base: Any, size: int) -> Any: ...
	def initial_jacobian(self, state: Any) -> Any: ...
	def reconstruct(self, unknown: Any, mapped: Any, size: int) -> Any: ...
	def multiplier_slice(self, size: int) -> slice: ...


__all__ = ['ProjectionName', 'MapEvaluation', 'ProjectionFormulation']
