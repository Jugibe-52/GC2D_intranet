"""Structural contracts for method-owned direct and adjoint numerical maps."""

from typing import Protocol, TypeAlias
from contracts.arrays import BackendArray
from contracts.states.history import DiagnosticValue

Projection: TypeAlias = tuple[BackendArray, dict[str, DiagnosticValue]]


class DirectAdjointMaps(Protocol):
	"""Operations on caller-supplied work arrays, without initial-state ownership."""

	def direct_map(self, duration: float, t: float, state: BackendArray) -> BackendArray: ...
	def adjoint_map(self, duration: float, t: float, state: BackendArray) -> BackendArray: ...
	def project(self, internal_history: BackendArray) -> Projection: ...


class StageProjectedMaps(DirectAdjointMaps, Protocol):
	"""Numerical maps optionally exposing arithmetic projection of their copies."""

	@property
	def supports_stage_projection(self) -> bool: ...
	def project_internal_state(self, state: BackendArray) -> BackendArray: ...


__all__ = ['Projection', 'DirectAdjointMaps', 'StageProjectedMaps']
