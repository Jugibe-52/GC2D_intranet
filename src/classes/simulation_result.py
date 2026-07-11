import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from pyhamsys import OdeSolution

if TYPE_CHECKING:
	from classes.fourier_system import FourierSystem

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
	system: "FourierSystem"
	sol: OdeSolution
	elapsed: float
	fig: Figure | None = None
	ax: Axes | None = None

	def _get_xy_trayectorys(self) -> tuple[np.ndarray, np.ndarray]:
		n_traj = int(self.system.Ntraj)
		if n_traj <= 0:
			raise ValueError(f"Invalid number of trajectories: Ntraj={self.system.Ntraj!r}.")
		if self.sol.y.shape[0] < 2 * n_traj:
			raise ValueError(
				f"Solution has shape {self.sol.y.shape}, expected at least "
				f"{2 * n_traj} rows for {n_traj} trajectories."
			)

		x = self.sol.y[:n_traj]
		y = self.sol.y[n_traj:2 * n_traj]
		return x, y

	def get_plot_trayectorys(self, modulo: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
		x, y = self._get_xy_trayectorys()
		use_modulo = getattr(self.system, 'modulo', False) if modulo is None else modulo
		if use_modulo:
			x, y = x % (2 * np.pi), y % (2 * np.pi)
		return x, y

	def get_trayectorys(self, modulo: bool | None = None) -> np.ndarray:
		x, y = self.get_plot_trayectorys(modulo=modulo)
		return np.stack((x, y), axis=-1)

	def get_initials_conditions(self, modulo: bool | None = None) -> np.ndarray:
		return self.get_trayectorys(modulo=modulo)[:, 0, :]

	def plot_poincare(
		self,
		modulo: bool | None = None,
		ax: Axes | None = None,
		grid: bool | None = None,
		decimal_grid: bool = False,
		grid_step: float = 0.5,
		**plot_kwargs: Any,
	) -> tuple[Figure, Axes]:
		system = self.system
		logger.info(
			"Plotting Poincare section: traj=%s modulo=%s",
			system.traj_type,
			getattr(system, 'modulo', False) if modulo is None else modulo,
		)
		if ax is None:
			fig, ax = plt.subplots(1, 1, figsize=(6, 6))
		else:
			fig = cast(Figure, ax.figure)
		x, y = self.get_plot_trayectorys(modulo=modulo)
		use_modulo = getattr(system, 'modulo', False) if modulo is None else modulo
		use_grid = getattr(system, 'grid', False) if grid is None else grid
		if use_modulo or use_grid or decimal_grid:
			ax.set_xlim(0, 2 * np.pi)
			ax.set_ylim(0, 2 * np.pi)
			if use_modulo and not decimal_grid:
				ax.set_xticks([0, np.pi, 2 * np.pi])
				ax.set_yticks([0, np.pi, 2 * np.pi])
				ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
				ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
		default_kwargs: dict[str, Any] = {
			'markersize': 3 if system.traj_type == 'gc' else 1,
			'markeredgecolor': 'none',
		}
		default_kwargs.update(plot_kwargs)
		ax.plot(x, y, '.', **default_kwargs)
		if decimal_grid:
			if grid_step <= 0:
				raise ValueError(f"`grid_step` must be positive, got {grid_step!r}.")
			step_text = f"{grid_step:.10f}".rstrip('0').rstrip('.')
			decimals = len(step_text.rsplit('.', 1)[1]) if '.' in step_text else 0
			decimals = max(decimals, 1)
			ax.xaxis.set_major_locator(MultipleLocator(grid_step))
			ax.yaxis.set_major_locator(MultipleLocator(grid_step))
			ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{decimals}f'))
			ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{decimals}f'))
			use_grid = True
		if use_grid:
			ax.grid(True, which='major', linewidth=0.5, alpha=0.35)
		ax.set_xlabel('$x$')
		ax.set_ylabel('$y$')
		ax.set_aspect('equal')
		self.fig = fig
		self.ax = ax
		return fig, ax
