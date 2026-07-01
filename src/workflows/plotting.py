import logging
from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from classes.potential_system import PotentialSystem
from classes.fourier_system import FourierSystem
from classes.simulation_result import SimulationResult

logger = logging.getLogger(__name__)


def plot_poincare(
	result: SimulationResult,
	modulo: bool | None = None,
	ax: Any = None,
	grid: bool | None = None,
	decimal_grid: bool = False,
	grid_step: float = 0.5,
	**plot_kwargs: Any,
) -> tuple[Any, Any]:
	return result.plot_poincare(
		modulo=modulo,
		ax=ax,
		grid=grid,
		decimal_grid=decimal_grid,
		grid_step=grid_step,
		**plot_kwargs,
	)


def _white_centered_cmap(vmin: float, vmax: float) -> tuple[Any, Any]:
	if vmin >= 0:
		cmap = plt.get_cmap('Reds')
		norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
		return cmap, norm
	if vmax <= 0:
		cmap = plt.get_cmap('Blues_r')
		norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
		return cmap, norm
	norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
	return plt.get_cmap('RdBu_r'), norm


def plot_potential(system: PotentialSystem, dx: int = 0, dy: int = 0, nx: int = 512, ny: int = 512) -> None:
	x = np.linspace(system.xmin, system.xmax + system.dx, nx, endpoint=False)
	y = np.linspace(system.ymin, system.ymax + system.dy, ny, endpoint=False)
	if system.fields[0] is not None:
		Z = system.interpolators[0](x, y, dx=dx, dy=dy)
		vmin, vmax = Z.min(), Z.max()
		cmap, norm = _white_centered_cmap(vmin, vmax)
		plt.figure(figsize=(6, 5))
		c = plt.pcolormesh(x, y, Z.T, shading='auto', cmap=cmap, norm=norm)
		plt.title(f'Potential (dx={dx}, dy={dy}) (freq=0)')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.colorbar(c)
		plt.tight_layout()
	if system.fields[1] is not None:
		for interpolator, freq in zip(system.interpolators[1], system.freqs):
			Zr, Zi = interpolator[0](x, y, dx=dx, dy=dy), interpolator[1](x, y, dx=dx, dy=dy)
			vmin_real, vmax_real = Zr.min(), Zr.max()
			vmin_imag, vmax_imag = Zi.min(), Zi.max()
			fig, axs = plt.subplots(1, 2, figsize=(12, 5))
			cmap_real, norm_real = _white_centered_cmap(vmin_real, vmax_real)
			c1 = axs[0].pcolormesh(x, y, Zr.T, shading='auto', cmap=cmap_real, norm=norm_real)
			axs[0].set_title(f'Real part of (dx={dx}, dy={dy}) potential (freq={freq})')
			axs[0].set_xlabel('x')
			axs[0].set_ylabel('y')
			fig.colorbar(c1, ax=axs[0])
			cmap_imag, norm_imag = _white_centered_cmap(vmin_imag, vmax_imag)
			c2 = axs[1].pcolormesh(x, y, Zi.T, shading='auto', cmap=cmap_imag, norm=norm_imag)
			axs[1].set_title(f'Imaginary part of (dx={dx}, dy={dy}) potential (freq={freq})')
			axs[1].set_xlabel('x')
			axs[1].set_ylabel('y')
			fig.colorbar(c2, ax=axs[1])
			plt.tight_layout()
	plt.show()


def plot_sol(
	system: PotentialSystem,
	sol: Any,
	wrap: bool = False,
	xlim: tuple[float, float] | None = None,
	ylim: tuple[float, float] | None = None,
	**kwargs: Any,
) -> None:
	x, y = system.get_positions(sol.y)
	xmin, xmax = xlim or (system.xmin, system.xmax)
	ymin, ymax = ylim or (system.ymin, system.ymax)
	if wrap:
		x, y = system.wrap_or_clip(x, y)
	plt.plot(x.T, y.T, '.', **kwargs)
	plt.xlabel('x')
	plt.ylabel('y')
	if wrap:
		plt.xlim(xmin, xmax)
		plt.ylim(ymin, ymax)
	plt.show()


def fft_phi_grid(system: FourierSystem, t: float = 0.0, n: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	x = np.linspace(0, 2 * np.pi, n, endpoint=False)
	y = np.linspace(0, 2 * np.pi, n, endpoint=False)
	X, Y = np.meshgrid(x, y, indexing='ij')
	state = np.concatenate((X.ravel(), Y.ravel()))
	vx, vy = np.split(system.y_dot(t, state), 2)
	return X, Y, vx.reshape(n, n), vy.reshape(n, n)


def plot_fft_phi(
	system: FourierSystem,
	t: float = 0.0,
	n: int = 40,
	kind: Literal['quiver', 'stream', 'magnitude'] = 'quiver',
	ax: Any = None,
	show_magnitude: bool = True,
	density: float = 1.5,
	**kwargs: Any,
) -> tuple[Any, Any]:
	if kind not in {'quiver', 'stream', 'magnitude'}:
		raise ValueError("`kind` must be 'quiver', 'stream' or 'magnitude'.")
	X, Y, vx, vy = fft_phi_grid(system, t=t, n=n)
	speed = np.sqrt(vx**2 + vy**2)
	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	else:
		fig = ax.figure
	if show_magnitude or kind == 'magnitude':
		mesh = ax.pcolormesh(X.T, Y.T, speed.T, shading='auto', cmap=kwargs.pop('cmap', 'viridis'))
		fig.colorbar(mesh, ax=ax, label=r'$|\dot{x}, \dot{y}|$')
	if kind == 'quiver':
		default_kwargs = {'pivot': 'mid', 'scale': None}
		default_kwargs.update(kwargs)
		ax.quiver(X.T, Y.T, vx.T, vy.T, **default_kwargs)
	elif kind == 'stream':
		default_kwargs = {'color': 'white' if show_magnitude else None}
		default_kwargs.update(kwargs)
		if default_kwargs['color'] is None:
			default_kwargs.pop('color')
		ax.streamplot(X[:, 0], Y[0, :], vx.T, vy.T, density=density, **default_kwargs)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	ax.set_title(r'Field from $\mathrm{fft\_phi\_}$')
	ax.set_aspect('equal')
	ax.set_xlim(0, 2 * np.pi)
	ax.set_ylim(0, 2 * np.pi)
	return fig, ax


def plot_symplectic_poincare(system: FourierSystem, sol: Any) -> tuple[Any, Any]:
	logger.info("Plotting Poincare section: traj=%s modulo=%s", system.traj_type, getattr(system, 'modulo', False))
	fig, ax = plt.subplots(1, 1)
	if system.traj_type == 'gc':
		x, y = np.split(sol.y, 2)
	elif system.CheckEnergy:
		x, y = np.split(sol.y, 5)[:2]
	else:
		x, y = np.split(sol.y, 4)[:2]
	if getattr(system, 'modulo', False):
		x, y = x % (2 * np.pi), y % (2 * np.pi)
		ax.set_xlim(0, 2 * np.pi)
		ax.set_ylim(0, 2 * np.pi)
	ax.plot(x, y, '.', markersize=3 if system.traj_type == 'gc' else 1, markeredgecolor='none')
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	ax.set_aspect('equal')
	if system.SaveData:
		extension = getattr(system, 'extension', '.png')
		basename = f'{type(system).__name__}_A{system.A:.2f}_RHO{system.rho:.4f}'.replace('.', '')
		filename = f'{basename}{extension}'
		fig.savefig(filename, dpi=getattr(system, 'dpi', 200))
		logger.info("Figure saved in %s", filename)
	if 'agg' not in plt.get_backend().lower():
		plt.pause(0.5)
	return fig, ax
