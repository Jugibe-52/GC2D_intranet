"""Long-time comparison of three geometric methods and fourth-order SDIRK."""

from __future__ import annotations

from contracts.comparison import (ThreeMethodNewtonSummary)

from ._comparison import (
	COMPARISON_LABELS, ComparisonConfig, ComparisonResult,
	implicit_summaries, run_alternating_comparison,
)

from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Mapping


from initial_conditions import GCInitialConfiguration
from potential import Potential


FOUR_METHOD_SDIRK_METHODS: tuple[str, ...] = (
	"ABBA4Implicit",
	"GaussLegendre4",
	"BM4Implicit",
	"SDIRK4",
)
FOUR_METHOD_SDIRK_LABELS: Mapping[str, str] = MappingProxyType(
	{name: COMPARISON_LABELS[name] for name in FOUR_METHOD_SDIRK_METHODS}
)


@dataclass(frozen=True, slots=True)
class FourMethodSDIRKComparisonConfig(ComparisonConfig):
	"""Common physical grid, Newton controls, references, and timings."""


@dataclass(frozen=True, slots=True)
class FourMethodSDIRKSummary(ThreeMethodNewtonSummary):
	"""Accuracy, runtime, and nonlinear work for one of the four methods."""


@dataclass(frozen=True, slots=True)
class FourMethodSDIRKComparisonResult(ComparisonResult):
	"""Reference and aligned solutions for the configured Newton methods."""

	config: FourMethodSDIRKComparisonConfig
	config_type: ClassVar[type[ComparisonConfig]] = FourMethodSDIRKComparisonConfig
	required_methods: ClassVar[tuple[str, ...] | None] = FOUR_METHOD_SDIRK_METHODS
	allow_missing_solver: ClassVar[bool] = False

	def summaries(self) -> tuple[FourMethodSDIRKSummary, ...]:
		"""Combine shared accuracy and nonlinear-work reductions."""
		return implicit_summaries(self, FourMethodSDIRKSummary)


def run_four_method_sdirk_comparison(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	config: FourMethodSDIRKComparisonConfig,
) -> FourMethodSDIRKComparisonResult:
	"""Run the named Newton methods with alternating timing order and an audited reference."""
	if not isinstance(config, FourMethodSDIRKComparisonConfig):
		raise TypeError("`config` must be FourMethodSDIRKComparisonConfig.")
	return run_alternating_comparison(potential, initial_configuration, config=config,
		method_names=FOUR_METHOD_SDIRK_METHODS, result_type=FourMethodSDIRKComparisonResult)


__all__ = [
	"FOUR_METHOD_SDIRK_LABELS",
	"FOUR_METHOD_SDIRK_METHODS",
	"FourMethodSDIRKComparisonConfig",
	"FourMethodSDIRKComparisonResult",
	"FourMethodSDIRKSummary",
	"run_four_method_sdirk_comparison",
]
