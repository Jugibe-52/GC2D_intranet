"""Symplecticity studies for ABBA with Hairer's symmetric projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from initial_conditions import Area
from potential import Potential
from simulation import SymmetricProjectedABBA

from ._gc_symplecticity import (
	GCSymplecticityConfig,
	GCSymplecticityResult,
	GCSymplecticitySummary,
	_run_gc_symplecticity_study,
)
from ._validation import positive_finite, positive_integer


@dataclass(frozen=True, slots=True)
class ABBASymplecticityConfig(GCSymplecticityConfig):
	"""Reproducible grids and exact-Newton parameters for an ABBA GC study."""

	block_prefix: str = "abba_symplecticity"
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12

	def __post_init__(self) -> None:
		"""Validate common study parameters and the implicit projection solve."""
		GCSymplecticityConfig.__post_init__(self)
		object.__setattr__(
			self,
			"newton_absolute_tolerance",
			positive_finite(
				self.newton_absolute_tolerance,
				"newton_absolute_tolerance",
			),
		)
		object.__setattr__(
			self,
			"newton_relative_tolerance",
			positive_finite(
				self.newton_relative_tolerance,
				"newton_relative_tolerance",
			),
		)
		object.__setattr__(
			self,
			"newton_max_iterations",
			positive_integer(
				self.newton_max_iterations,
				"newton_max_iterations",
			),
		)


@dataclass(frozen=True, slots=True)
class ABBASymplecticitySummary(GCSymplecticitySummary):
	"""Maximum ABBA defects and Newton statistics for one step size."""


@dataclass(frozen=True, slots=True)
class ABBAProjectionMultiplierOrder:
	"""Empirical scaling of the maximum projection multiplier between steps."""

	coarse_label: str
	fine_label: str
	value: float


@dataclass(frozen=True, slots=True)
class ABBASymplecticityResult(GCSymplecticityResult):
	"""Projected ABBA solutions and numerical-floor diagnostics."""

	method_name: ClassVar[str] = "SymmetricProjectedABBA"
	summary_type: ClassVar[type[GCSymplecticitySummary]] = ABBASymplecticitySummary

	def summaries(self) -> tuple[ABBASymplecticitySummary, ...]:
		"""Return ABBA summary values in configured step order."""
		return cast(
			tuple[ABBASymplecticitySummary, ...],
			GCSymplecticityResult.summaries(self),
		)

	def projection_multiplier_orders(
		self,
	) -> tuple[ABBAProjectionMultiplierOrder, ...]:
		"""Estimate the expected cubic small-step scaling of ``mu``."""
		rows = self.summaries()
		orders: list[ABBAProjectionMultiplierOrder] = []
		for coarse, fine in zip(rows, rows[1:]):
			coarse_norm = coarse.max_projection_multiplier_norm
			fine_norm = fine.max_projection_multiplier_norm
			if (
				coarse_norm is None
				or fine_norm is None
				or coarse_norm <= 0
				or fine_norm <= 0
				or np.isclose(coarse.step, fine.step)
			):
				value = float("nan")
			else:
				value = float(
					np.log(coarse_norm / fine_norm)
					/ np.log(coarse.step / fine.step)
				)
			orders.append(
				ABBAProjectionMultiplierOrder(
					coarse_label=coarse.label,
					fine_label=fine.label,
					value=value,
				)
			)
		return tuple(orders)

	def print_summary(self) -> None:
		"""Print finite-tolerance ABBA defects without inferring trajectory order."""
		GCSymplecticityResult.print_summary(self)
		print(
			"\nThe observer differentiates the emitted finite-tolerance ABBA map by "
			"centered differences. The reported defects include differentiation, "
			"Newton, and floating-point floors; no trajectory convergence order is "
			"inferred."
		)
		print("\nEmpirical scaling of the maximum projection multiplier (expected ~3):")
		for order in self.projection_multiplier_orders():
			print(
				f"  {order.coarse_label} -> {order.fine_label}: "
				f"{order.value:.6f}"
			)

	def plot_defect_floor(self) -> tuple[Figure, Axes]:
		"""Plot measured ABBA defects across steps as a numerical floor."""
		return self._plot_step_defects(
			title="Projected ABBA symplecticity-defect numerical floor",
			xlabel=r"ABBA step $\Delta t$",
		)


def run_abba_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: ABBASymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> ABBASymplecticityResult:
	"""Run projected ABBA steps and persist physical GC flow diagnostics."""
	if not isinstance(config, ABBASymplecticityConfig):
		raise TypeError("`config` must be an ABBASymplecticityConfig instance.")
	study_metadata = {
		**dict(metadata or {}),
		"newton_absolute_tolerance": config.newton_absolute_tolerance,
		"newton_relative_tolerance": config.newton_relative_tolerance,
		"newton_max_iterations": config.newton_max_iterations,
		"newton_initial_multiplier": "zero",
		"newton_residual_norm": "infinity",
		"abba_stage_times": "t_n,t_n,t_n_plus_h,t_n_plus_h",
		"step_jacobian": "centered_difference_of_emitted_solver_map",
	}
	return _run_gc_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=config,
		method_factory=lambda observer: SymmetricProjectedABBA(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			progress=config.progress,
			step_observer=observer,
		),
		result_type=ABBASymplecticityResult,
		project_root=project_root,
		metadata=study_metadata,
	)


__all__ = [
	"ABBAProjectionMultiplierOrder",
	"ABBASymplecticityConfig",
	"ABBASymplecticityResult",
	"ABBASymplecticitySummary",
	"run_abba_symplecticity_study",
]
