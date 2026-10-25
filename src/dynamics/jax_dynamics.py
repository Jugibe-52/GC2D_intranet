"""GC and FC equations inherited with explicit JAX coordinate operations.

Direct compiled evaluation requires a JaxPotential. Execution preparation
supplies it automatically, sharing the source's already prepared physical data.
"""

from typing import ClassVar

from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics
from dynamics.jax_coordinates import JaxGCCoordinates, JaxFCCoordinates


class JaxGuidingCenterDynamics(GuidingCenterDynamics):
	"""The existing GC physics evaluated through JAX coordinates and potential."""

	coordinates: ClassVar[JaxGCCoordinates] = JaxGCCoordinates()


class JaxFullCyclotronDynamics(FullCyclotronDynamics):
	"""The existing FC physics evaluated through JAX coordinates and potential."""

	coordinates: ClassVar[JaxFCCoordinates] = JaxFCCoordinates()


__all__ = ["JaxGuidingCenterDynamics", "JaxFullCyclotronDynamics"]
