"""Structural contracts consumed by studies._trajectory_accuracy."""

from __future__ import annotations

from typing import Protocol


class ReferenceAccuracyConfig(Protocol):
	"""Configuration fields required to validate a stored GC reference."""

	@property
	def rho(self) -> float:
		"""Return the normalized gyroradius."""
		...

	@property
	def t_span(self) -> tuple[float, float]:
		"""Return the integration interval."""
		...

	@property
	def save_interval(self) -> float | None:
		"""Return the common saved-time interval after configuration validation."""
		...


__all__ = ['ReferenceAccuracyConfig']
