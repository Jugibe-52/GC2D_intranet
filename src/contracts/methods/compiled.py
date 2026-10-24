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
	def track_energy(self) -> bool: ...

	@property
	def finite_difference(self) -> bool: ...

	def __call__(self, time: Any, state: Any, step: Any, momentum: Any = None) -> tuple[Any, Any, dict[str, Any], Any]:
		"""Return physical values, separate momentum, work counters and convergence."""
		...

	def energy(self, times: Any, physical: Any) -> Any:
		"""Evaluate physical energy with the same prepared field as the step."""
		...


__all__ = ["CompiledStep"]
