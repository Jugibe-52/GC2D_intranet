"""Structural contracts for method-owned direct and adjoint numerical maps."""

from typing import Protocol, TypeAlias
import numpy as np
from contracts.states.history import DiagnosticValue

Projection: TypeAlias = tuple[np.ndarray, dict[str, DiagnosticValue]]


class DirectAdjointMaps(Protocol):
	"""Operations on caller-supplied work arrays, without initial-state ownership."""

	def direct_map(self, duration: float, t: float, state: np.ndarray) -> np.ndarray: ...
	def adjoint_map(self, duration: float, t: float, state: np.ndarray) -> np.ndarray: ...
	def project(self, internal_history: np.ndarray) -> Projection: ...


class StageProjectedMaps(DirectAdjointMaps, Protocol):
	"""Numerical maps optionally exposing arithmetic projection of their copies."""

	@property
	def supports_stage_projection(self) -> bool: ...
	def project_internal_state(self, state: np.ndarray) -> np.ndarray: ...


__all__ = ['Projection', 'DirectAdjointMaps', 'StageProjectedMaps']
