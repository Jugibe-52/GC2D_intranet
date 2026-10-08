"""Canonical component-major physical states, independent of their initial source."""

from __future__ import annotations

from typing import Any, ClassVar, NamedTuple

from contracts.arrays import array_namespace


class PackedStateLayout:
	"""Reusable operations for one component-major state layout.

	``state_dimension`` is the number of scalar components stored per particle;
	concrete layouts define the physical meaning and order of those blocks.
	For ``N`` particles, a state has shape
	``(state_dimension * N, *sample_axes)`` and each component returned by
	:meth:`split` has shape ``(N, *sample_axes)``.  The optional ``sample_axes``
	usually represent saved integration times and are never part of the particle
	count. NumPy and JAX inputs retain their array backend, including JAX
	values being traced by a compiled or differentiated function.
	"""

	__slots__ = ()

	# Number of component blocks on the leading state axis.  Subclasses provide
	# the value because GC and FC states carry different physical variables.
	state_dimension: ClassVar[int]

	def split(self, state: Any) -> tuple[Any, ...]:
		"""Split the leading axis into equally sized physical components.

		A single state has shape ``(state_dimension * N,)``.  A solution with
		shape ``(state_dimension * N, *sample_axes)`` is split along axis zero,
		leaving every component with shape ``(N, *sample_axes)``.
		"""
		value = array_namespace(state).asarray(state)
		blocks = value.reshape(
			(self.state_dimension, -1, *value.shape[1:])
		)
		return tuple(blocks)

	def as_blocks(self, state: Any) -> Any:
		"""Expose a packed state as ``(components, particles, *samples)``.

		The returned array is a reshape view whenever NumPy can preserve the input
		memory layout.  Making both physical axes explicit avoids repeating manual
		block offsets inside numerical algorithms.
		"""
		value = self.validate_packed_state_layout(state)
		particle_count = value.shape[0] // self.state_dimension
		return value.reshape(
			(self.state_dimension, particle_count, *value.shape[1:])
		)

	def validate_packed_state_layout(self, state: Any) -> Any:
		"""Validate and return a component-major state-array layout.

		The leading axis must contain a non-zero whole number of physical
		component blocks.  :meth:`as_blocks` calls this before reshaping;
		integrators can call it when only layout validation is required.
		"""
		value = array_namespace(state).asarray(state)
		if (
			value.ndim == 0
			or value.shape[0] == 0
			or value.shape[0] % self.state_dimension
		):
			raise ValueError(
				f"The first state dimension must be divisible by {self.state_dimension} "
				f"for {self.__class__.__name__}."
			)
		return value

	def from_blocks(self, blocks: Any) -> Any:
		"""Flatten ``(components, particles, *samples)`` into state layout.

		This is the inverse view operation of :meth:`as_blocks`.  It is useful for
		internal algorithms that already produce all component blocks in one array
		and therefore need not concatenate them individually. The input backend
		is preserved.
		"""
		value = array_namespace(blocks).asarray(blocks)
		if (
			value.ndim < 2
			or value.shape[0] != self.state_dimension
			or value.shape[1] == 0
		):
			raise ValueError(
				f"Blocks for {self.__class__.__name__} must have shape "
				f"({self.state_dimension}, N, *sample_axes) with N greater than zero."
			)
		return value.reshape(
			(self.state_dimension * value.shape[1], *value.shape[2:])
		)

	@classmethod
	def pack_components(cls, *components: Any) -> Any:
		"""Pack named-constructor inputs without requiring a trajectory instance.

		Every component has shape ``(N, *sample_axes)``. The concrete layout
		class supplies their required count and physical order through
		``state_dimension`` and the order in which callers pass the arrays. Any JAX
		component selects JAX for the complete packed result.
		"""
		if len(components) != cls.state_dimension:
			raise ValueError(
				f"{cls.__name__} requires {cls.state_dimension} components."
			)
		xp = array_namespace(*components)
		values = tuple(xp.asarray(component) for component in components)
		if not values or values[0].ndim == 0 or values[0].shape[0] == 0:
			raise ValueError("State components must be non-empty arrays.")
		if any(value.shape != values[0].shape for value in values[1:]):
			raise ValueError("All state components must have the same shape.")
		# Stack once to make the component axis explicit, then flatten that axis and
		# the particle axis through a view. This keeps packing as one allocation.
		blocks = xp.stack(values, axis=0)
		return blocks.reshape(
			(cls.state_dimension * blocks.shape[1], *blocks.shape[2:])
		)

	def particle_count(self, state: Any) -> int:
		"""Return ``N`` from the leading axis, ignoring all sample axes."""
		return int(self.as_blocks(state).shape[1])

class GCState(NamedTuple):
	"""Guiding-centre coordinates with matching particle/sample dimensions.

	Each field has shape ``(N, *sample_axes)``: axis zero identifies the
	particle (or contour vertex) and trailing axes identify solution samples.
	"""

	x: Any
	y: Any


class GCStateLayout(PackedStateLayout):
	"""Interpret packed guiding-centre coordinates in ``[x, y]`` order."""

	__slots__ = ()
	state_dimension = 2

	def split(self, state: Any) -> GCState:
		"""Return named ``x`` and ``y`` blocks, preserving sample axes."""
		x, y = super().split(state)
		return GCState(x, y)

	def positions(self, state: Any) -> tuple[Any, Any]:
		"""Return the two guiding-centre coordinate blocks."""
		components = self.split(state)
		return components.x, components.y


class FCState(NamedTuple):
	"""FC position and velocity coordinates with matching dimensions.

	Every field has shape ``(N, *sample_axes)``.  ``vx`` and ``vy`` are the
	velocity coordinates stored by the normalized model. Their physical scaling
	is owned by :class:`dynamics.FullCyclotronDynamics`.
	"""

	x: Any
	y: Any
	vx: Any
	vy: Any


class FCStateLayout(PackedStateLayout):
	"""Interpret packed full-cyclotron states in ``[x, y, vx, vy]`` order."""

	__slots__ = ()
	state_dimension = 4

	def split(self, state: Any) -> FCState:
		"""Return named position and velocity blocks, preserving sample axes."""
		x, y, vx, vy = super().split(state)
		return FCState(x, y, vx, vy)

	def positions(self, state: Any) -> tuple[Any, Any]:
		"""Return the two full-cyclotron position blocks."""
		components = self.split(state)
		return components.x, components.y


__all__ = [
	"PackedStateLayout",
	"GCState", "GCStateLayout",
	"FCState", "FCStateLayout",
]
