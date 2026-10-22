"""Reduced and simultaneous spatial projection equations, independent of methods."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, TYPE_CHECKING, TypeAlias

from contracts.arrays import array_namespace


if TYPE_CHECKING:
	from contracts.arrays import BackendArray
	from types import ModuleType


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


@dataclass(frozen=True, slots=True)
class ReducedProjection:
	"""Solve for mu in R^(2N); the physical output is reconstructed from the map."""

	name: ClassVar[ProjectionName] = 'reduced_multiplier'
	unknown_components: ClassVar[int] = 2

	def prepare(self, state: Any, mapping: MapEvaluation) -> tuple[Any, Any]:
		return array_namespace(state).zeros(state.size), None

	def multiplier(self, unknown: Any, size: int) -> Any:
		return unknown

	def residual(self, unknown: Any, mapped: Any, size: int) -> Any:
		return mapped[:size] - mapped[size:] + 2 * unknown

	def particle_jacobian(self, base: Any) -> Any:
		"""Assemble N independent 2x2 blocks from doubled-map derivatives."""
		xp: ModuleType = array_namespace(base)
		identity: BackendArray = xp.broadcast_to(xp.eye(2), (base.shape[0], 2, 2))
		return base[:, :2, :2] - base[:, :2, 2:] - base[:, 2:, :2] + base[:, 2:, 2:] + 2 * identity

	def dense_jacobian(self, base: Any, size: int) -> Any:
		xp: ModuleType = array_namespace(base)
		identity: BackendArray = xp.eye(size)
		normal: BackendArray = xp.concatenate((identity, -identity), axis=0)
		return normal.T @ base @ normal + 2 * identity

	def initial_jacobian(self, state: Any) -> Any:
		"""The reduced residual derivative is 4I at zero integration step."""
		return 4 * array_namespace(state).eye(state.size)

	def reconstruct(self, unknown: Any, mapped: Any, size: int) -> Any:
		return ((mapped[:size] + unknown) + (mapped[size:] - unknown)) / 2

	def multiplier_slice(self, size: int) -> slice:
		return slice(None)


@dataclass(frozen=True, slots=True)
class SimultaneousProjection:
	"""Solve for (u, v, mu) in R^(6N), imposing equality of physical copies."""

	name: ClassVar[ProjectionName] = 'simultaneous_state_multiplier'
	unknown_components: ClassVar[int] = 6

	def prepare(self, state: Any, mapping: MapEvaluation) -> tuple[Any, Any]:
		"""Reuse the initial unshifted map once in the first residual evaluation."""
		xp: ModuleType = array_namespace(state)
		size = state.size
		mapped: BackendArray
		payload: Any
		mapped, payload = mapping(xp.concatenate((state, state)))
		initial: BackendArray = xp.concatenate((mapped, xp.zeros(size)))
		cached = (xp.concatenate((xp.zeros(2 * size), mapped[:size] - mapped[size:])), payload)
		return initial, cached

	def multiplier(self, unknown: Any, size: int) -> Any:
		return unknown[2 * size:]

	def residual(self, unknown: Any, mapped: Any, size: int) -> Any:
		xp: ModuleType = array_namespace(unknown, mapped)
		u: BackendArray
		v: BackendArray
		mu: BackendArray
		u, v, mu = unknown[:size], unknown[size:2 * size], unknown[2 * size:]
		return xp.concatenate((u - mu - mapped[:size], v + mu - mapped[size:], u - v))

	def particle_jacobian(self, base: Any) -> Any:
		"""Assemble N independent 6x6 blocks, retaining component ordering."""
		xp: ModuleType = array_namespace(base)
		count = base.shape[0]
		identity: BackendArray = xp.broadcast_to(xp.eye(2), (count, 2, 2))
		normal: BackendArray = xp.concatenate((identity, -identity), axis=1)
		constraint: BackendArray = xp.concatenate((identity, -identity), axis=2)
		identity4: BackendArray = xp.broadcast_to(xp.eye(4), (count, 4, 4))
		return xp.concatenate((
			xp.concatenate((identity4, -(identity4 + base) @ normal), axis=2),
			xp.concatenate((constraint, xp.zeros((count, 2, 2))), axis=2)), axis=1)

	def dense_jacobian(self, base: Any, size: int) -> Any:
		xp: ModuleType = array_namespace(base)
		identity: BackendArray = xp.eye(size)
		normal: BackendArray = xp.concatenate((identity, -identity), axis=0)
		return xp.block([[xp.eye(2 * size), -(xp.eye(2 * size) + base) @ normal],
			[normal.T, xp.zeros((size, size))]])

	def initial_jacobian(self, state: Any) -> Any:
		xp: ModuleType = array_namespace(state)
		size = state.size
		identity: BackendArray = xp.eye(size)
		normal: BackendArray = xp.concatenate((identity, -identity), axis=0)
		return xp.block([[xp.eye(2 * size), -2 * normal], [normal.T, xp.zeros((size, size))]])

	def reconstruct(self, unknown: Any, mapped: Any, size: int) -> Any:
		return (unknown[:size] + unknown[size:2 * size]) / 2

	def multiplier_slice(self, size: int) -> slice:
		return slice(2 * size, None)


ProjectionChoice: TypeAlias = ProjectionName | ReducedProjection | SimultaneousProjection


def resolve_projection(value: ProjectionChoice) -> ProjectionFormulation:
	"""Normalize public names or exact built-in equations without backend overrides."""
	if type(value) in (ReducedProjection, SimultaneousProjection):
		assert isinstance(value, (ReducedProjection, SimultaneousProjection))
		return value
	if isinstance(value, str):
		if value == 'reduced_multiplier':
			return ReducedProjection()
		if value == 'simultaneous_state_multiplier':
			return SimultaneousProjection()
	raise ValueError("`projection_formulation` must be 'reduced_multiplier', "
		"'simultaneous_state_multiplier', ReducedProjection() or SimultaneousProjection().")


__all__ = ['ProjectionFormulation', 'ReducedProjection', 'SimultaneousProjection']
