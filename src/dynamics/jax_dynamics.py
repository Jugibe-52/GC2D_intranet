"""GC and FC equations inherited with explicit JAX coordinate operations.

Direct compiled evaluation requires a JaxPotential. Execution preparation
supplies it automatically, sharing the source's already prepared physical data.
"""

from typing import ClassVar

from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics
from dynamics.jax_coordinates import JaxPhysicalCoordinates


class JaxGuidingCenterDynamics(GuidingCenterDynamics):
	"""The existing GC physics evaluated through JAX coordinates and potential."""

	coordinates: ClassVar[JaxPhysicalCoordinates] = JaxPhysicalCoordinates(
		GuidingCenterDynamics.coordinates.component_names,
		GuidingCenterDynamics.coordinates.position_names,
		GuidingCenterDynamics.coordinates.velocity_names,
	)


class JaxFullCyclotronDynamics(FullCyclotronDynamics):
	"""The existing FC physics evaluated through JAX coordinates and potential."""

	coordinates: ClassVar[JaxPhysicalCoordinates] = JaxPhysicalCoordinates(
		FullCyclotronDynamics.coordinates.component_names,
		FullCyclotronDynamics.coordinates.position_names,
		FullCyclotronDynamics.coordinates.velocity_names,
	)


__all__ = ["JaxGuidingCenterDynamics", "JaxFullCyclotronDynamics"]
