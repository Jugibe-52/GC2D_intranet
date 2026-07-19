"""High-level result returned by reusable simulation workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from .solution import Solution
from .system import System

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
	"""A system, its numerical solution and wall-clock execution time."""

	system: System
	solution: Solution
	elapsed: float
	fig: Figure | None = None
	ax: Axes | None = None

	@property
	def sol(self) -> Solution:
		"""Compact alias retained for notebook ergonomics."""
		return self.solution

	def positions(self) -> tuple[np.ndarray, np.ndarray]:
		return tuple(np.asarray(value) for value in self.system.get_positions(self.solution.y))  # type: ignore[return-value]

	def plotted_positions(self, modulo: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
		x, y = self.positions()
		use_modulo = bool(getattr(self.system, "modulo", False)) if modulo is None else modulo
		period = self.system.grid.period
		if use_modulo and period is not None:
			x = ((x - self.system.grid.xmin) % period) + self.system.grid.xmin
			y = ((y - self.system.grid.ymin) % period) + self.system.grid.ymin
		return x, y

	def trajectories(self, modulo: bool | None = None) -> np.ndarray:
		x, y = self.plotted_positions(modulo=modulo)
		return np.stack((x, y), axis=-1)

	def initial_conditions(self, modulo: bool | None = None) -> np.ndarray:
		return self.trajectories(modulo=modulo)[:, 0, :]

	def plot_poincare(
		self,
		modulo: bool | None = None,
		ax: Axes | None = None,
		grid: bool | None = None,
		decimal_grid: bool = False,
		grid_step: float = 0.5,
		**plot_kwargs: Any,
	) -> tuple[Figure, Axes]:
		"""Plot every stored point as a Poincare section."""
		if ax is None:
			fig, ax = plt.subplots(1, 1, figsize=(6, 6))
		else:
			fig = cast(Figure, ax.figure)
		x, y = self.plotted_positions(modulo=modulo)
		use_modulo = bool(getattr(self.system, "modulo", False)) if modulo is None else modulo
		use_grid = bool(getattr(self.system, "show_grid", False)) if grid is None else grid
		if use_modulo or use_grid or decimal_grid:
			period = self.system.grid.period or 2 * np.pi
			ax.set_xlim(self.system.grid.xmin, self.system.grid.xmin + period)
			ax.set_ylim(self.system.grid.ymin, self.system.grid.ymin + period)
		default_kwargs: dict[str, Any] = {
			"markersize": 3 if self.system.kind == "gc" else 1,
			"markeredgecolor": "none",
		}
		default_kwargs.update(plot_kwargs)
		ax.plot(x, y, ".", **default_kwargs)
		if decimal_grid:
			if grid_step <= 0:
				raise ValueError("`grid_step` must be positive.")
			step_text = f"{grid_step:.10f}".rstrip("0").rstrip(".")
			decimals = len(step_text.rsplit(".", 1)[1]) if "." in step_text else 1
			ax.xaxis.set_major_locator(MultipleLocator(grid_step))
			ax.yaxis.set_major_locator(MultipleLocator(grid_step))
			ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{max(decimals, 1)}f"))
			ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{max(decimals, 1)}f"))
			use_grid = True
		if use_grid:
			ax.grid(True, which="major", linewidth=0.5, alpha=0.35)
		ax.set_xlabel("$x$")
		ax.set_ylabel("$y$")
		ax.set_aspect("equal")
		self.fig, self.ax = fig, ax
		return fig, ax


__all__ = ["SimulationResult"]
