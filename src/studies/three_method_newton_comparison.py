"""Multi-trajectory comparison of three fourth-order Newton integrators."""

from __future__ import annotations

from ._comparison import (
	COMPARISON_LABELS, ComparisonConfig, ComparisonResult,
	implicit_summaries, run_alternating_comparison,
)

from contracts.comparison import (EnergyAccuracySeries, ThreeMethodNewtonSummary,)

from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Mapping


from initial_conditions import GCInitialConfiguration
from potential import Potential


THREE_METHOD_NEWTON_METHODS: tuple[str, ...] = (
	"ABBA4Implicit",
	"GaussLegendre4",
	"BM4Implicit",
)
THREE_METHOD_NEWTON_LABELS: Mapping[str, str] = MappingProxyType(
	{name: COMPARISON_LABELS[name] for name in THREE_METHOD_NEWTON_METHODS}
)


@dataclass(frozen=True, slots=True)
class ThreeMethodNewtonComparisonConfig(ComparisonConfig):
	"""Common physical grid, Newton controls, references, and timings."""


@dataclass(frozen=True, slots=True)
class ThreeMethodNewtonComparisonResult(ComparisonResult):
	"""Reference and aligned solutions for the configured Newton methods."""

	config: ThreeMethodNewtonComparisonConfig
	config_type: ClassVar[type[ComparisonConfig]] = ThreeMethodNewtonComparisonConfig
	required_methods: ClassVar[tuple[str, ...] | None] = THREE_METHOD_NEWTON_METHODS
	allow_missing_solver: ClassVar[bool] = True

	def summaries(self) -> tuple[ThreeMethodNewtonSummary, ...]:
		"""Combine shared accuracy and nonlinear-work reductions."""
		return implicit_summaries(self, ThreeMethodNewtonSummary)


def run_three_method_newton_comparison(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	config: ThreeMethodNewtonComparisonConfig,
) -> ThreeMethodNewtonComparisonResult:
	"""Run the named Newton methods with alternating timing order and an audited reference."""
	if not isinstance(config, ThreeMethodNewtonComparisonConfig):
		raise TypeError("`config` must be ThreeMethodNewtonComparisonConfig.")
	return run_alternating_comparison(potential, initial_configuration, config=config,
		method_names=THREE_METHOD_NEWTON_METHODS, result_type=ThreeMethodNewtonComparisonResult)


__all__ = [
	"THREE_METHOD_NEWTON_LABELS",
	"THREE_METHOD_NEWTON_METHODS",
	"EnergyAccuracySeries",
	"ThreeMethodNewtonComparisonConfig",
	"ThreeMethodNewtonComparisonResult",
	"ThreeMethodNewtonSummary",
	"run_three_method_newton_comparison",
]
