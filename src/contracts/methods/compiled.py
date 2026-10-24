"""Backend-neutral boundary for a prepared, device-resident fixed-step map."""

from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
	from jax import Device


class CompiledStep(Protocol):
	"""A hashable callable owning its kernel, physical binding and static controls.

	Array values remain backend-native. The coordinator only owns the time grid,
	so it must not inspect concrete numerical methods or reconstruct their options.
	"""

	def __hash__(self) -> int: ...

	@property
	def device(self) -> "Device": ...

	@property
	def physical_dimension(self) -> int: ...

	@property
	def copies(self) -> int: ...

	@property
	def finite_difference(self) -> bool: ...

	def __call__(self, time: Any, state: Any, step: Any) -> tuple[Any, Any, dict[str, Any], Any]:
		"""Return physical values, optional quadrature, work counters and convergence."""
		...


__all__ = ["CompiledStep"]
