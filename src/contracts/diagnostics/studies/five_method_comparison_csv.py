"""Structural contracts consumed by diagnostics.studies.five_method_comparison_csv."""

from __future__ import annotations

from typing import Protocol
from contracts.studies.comparison import ComparisonReadView


class _WritableComparison(ComparisonReadView, Protocol):
	"""Additional numerical controls required to preserve the CSV schema."""

	@property
	def config(self) -> _ComparisonConfigView: ...


class _ComparisonConfigView(Protocol):
	@property
	def step_count(self) -> int: ...


__all__ = []
