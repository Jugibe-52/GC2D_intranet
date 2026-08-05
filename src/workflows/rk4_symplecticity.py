"""RK4 studies of physical guiding-centre area and symplecticity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from classes import Area, Potential, RK4

from ._gc_symplecticity import (
	GCConvergenceOrder,
	GCSymplecticityConfig,
	GCSymplecticityResult,
	GCSymplecticitySummary,
	_run_gc_symplecticity_study,
)


@dataclass(frozen=True, slots=True)
class RK4SymplecticityConfig(GCSymplecticityConfig):
	"""Configuration defaults for an RK4 GC symplecticity study."""

	block_prefix: str = "rk4_symplecticity"


@dataclass(frozen=True, slots=True)
class RK4SymplecticitySummary(GCSymplecticitySummary):
	"""Maximum RK4 errors observed for one integration step."""


@dataclass(frozen=True, slots=True)
class RK4ConvergenceOrder(GCConvergenceOrder):
	"""Empirical RK4 defect order between consecutive integration steps."""


@dataclass(frozen=True, slots=True)
class RK4SymplecticityResult(GCSymplecticityResult):
	"""RK4 solutions, physical-flow Jacobians and analysis helpers."""

	method_name: ClassVar[str] = "RK4"
	summary_type: ClassVar[type[GCSymplecticitySummary]] = RK4SymplecticitySummary

	def summaries(self) -> tuple[RK4SymplecticitySummary, ...]:
		"""Return RK4 summary values in configured step order."""
		return cast(
			tuple[RK4SymplecticitySummary, ...],
			GCSymplecticityResult.summaries(self),
		)

	def convergence_orders(self) -> tuple[RK4ConvergenceOrder, ...]:
		"""Estimate the order of the maximum accumulated symplecticity defect."""
		summaries = self.summaries()
		orders: list[RK4ConvergenceOrder] = []
		for coarse, fine in zip(summaries, summaries[1:]):
			if (
				coarse.max_flow_defect <= 0
				or fine.max_flow_defect <= 0
				or np.isclose(coarse.step, fine.step)
			):
				value = float("nan")
			else:
				value = float(
					np.log(coarse.max_flow_defect / fine.max_flow_defect)
					/ np.log(coarse.step / fine.step)
				)
			orders.append(
				RK4ConvergenceOrder(
					coarse_label=coarse.label,
					fine_label=fine.label,
					value=value,
				)
			)
		return tuple(orders)

	def print_summary(self) -> None:
		"""Print RK4 errors and empirical defect convergence."""
		GCSymplecticityResult.print_summary(self)
		print("\nEmpirical order of the maximum accumulated symplecticity defect:")
		for order in self.convergence_orders():
			print(
				f"  {order.coarse_label} -> {order.fine_label}: "
				f"{order.value:.6f}"
			)

	def plot_convergence(self) -> tuple[Figure, Axes]:
		"""Plot RK4 symplecticity defects against the integration step size."""
		return self._plot_step_defects(
			title="RK4 symplecticity-defect convergence",
			xlabel=r"RK4 step $\Delta t$",
		)


def run_rk4_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: RK4SymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> RK4SymplecticityResult:
	"""Run synchronized RK4 steps and persist physical GC flow diagnostics."""
	if not isinstance(config, RK4SymplecticityConfig):
		raise TypeError("`config` must be an RK4SymplecticityConfig instance.")
	return _run_gc_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=config,
		method_factory=lambda observer: RK4(
			progress=config.progress,
			step_observer=observer,
		),
		result_type=RK4SymplecticityResult,
		project_root=project_root,
		metadata=metadata,
	)


__all__ = [
	"RK4ConvergenceOrder",
	"RK4SymplecticityConfig",
	"RK4SymplecticityResult",
	"RK4SymplecticitySummary",
	"run_rk4_symplecticity_study",
]
