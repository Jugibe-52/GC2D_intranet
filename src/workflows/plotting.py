"""Plotting helpers for composed simulation systems."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure

from classes import Potential, System
from classes.system import SimulationResult, Solution

logger = logging.getLogger(__name__)


def plot_poincare(
	result: SimulationResult,
	modulo: bool | None = None,
	ax: Axes | None = None,
	grid: bool | None = None,
	decimal_grid: bool = False,
	grid_step: float = 0.5,
	**plot_kwargs: Any,
) -> tuple[Figure, Axes]:
	return result.plot_poincare(
		modulo=modulo,
		ax=ax,
		grid=grid,
		decimal_grid=decimal_grid,
		grid_step=grid_step,
		**plot_kwargs,
	)


def _white_centered_cmap(
	vmin: float,
	vmax: float,
) -> tuple[Colormap, Normalize]:
	if np.isclose(vmin, vmax):
		delta = abs(vmin) * 0.01 or 1.0
		return plt.get_cmap("RdBu_r"), mcolors.Normalize(
			vmin=vmin - delta,
			vmax=vmax + delta,
		)
	if vmin >= 0:
		return plt.get_cmap("Reds"), mcolors.Normalize(vmin=vmin, vmax=vmax)
	if vmax <= 0:
		return plt.get_cmap("Blues_r"), mcolors.Normalize(vmin=vmin, vmax=vmax)
	return (
		plt.get_cmap("RdBu_r"),
		mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
	)


def _potential_from(value: Potential | System) -> Potential:
	return value.effective_potential if isinstance(value, System) else value


def plot_potential(
	value: Potential | System,
	dx: int = 0,
	dy: int = 0,
	nx: int = 512,
	ny: int = 512,
	*,
	t: float = 0.0,
) -> None:
	"""Plot a Potential directly, or the effective potential of a System."""
	potential = _potential_from(value)
	grid = potential.grid
	x_stop = grid.xmin + grid.period if grid.period is not None else grid.xmax
	y_stop = grid.ymin + grid.period if grid.period is not None else grid.ymax
	x = np.linspace(grid.xmin, x_stop, nx, endpoint=grid.period is None)
	y = np.linspace(grid.ymin, y_stop, ny, endpoint=grid.period is None)
	x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
	field = np.asarray(
		potential.field_at_time(t, x_mesh, y_mesh, dx=dx, dy=dy)
	)
	vmin, vmax = float(np.nanmin(field)), float(np.nanmax(field))
	cmap, norm = _white_centered_cmap(vmin, vmax)
	fig, ax = plt.subplots(figsize=(6, 5))
	mesh = ax.pcolormesh(x, y, field.T, shading="auto", cmap=cmap, norm=norm)
	ax.set(
		title=f"Potential (dx={dx}, dy={dy}, t={t:g})",
		xlabel="x",
		ylabel="y",
		aspect="equal",
	)
	fig.colorbar(mesh, ax=ax)
	fig.tight_layout()
	plt.show()


def plot_sol(
	system: System,
	solution: Solution,
	wrap: bool = False,
	xlim: tuple[float, float] | None = None,
	ylim: tuple[float, float] | None = None,
	**kwargs: Any,
) -> tuple[Figure, Axes]:
	"""Plot positions from a physical Solution using Trajectory state layout."""
	x, y = system.get_positions(solution.y)
	xmin, xmax = xlim or (system.grid.xmin, system.grid.xmax)
	ymin, ymax = ylim or (system.grid.ymin, system.grid.ymax)
	if wrap:
		x, y = system.grid.wrap_or_clip(x, y)
	fig, ax = plt.subplots(1, 1)
	ax.plot(x.T, y.T, ".", **kwargs)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	if wrap:
		ax.set_xlim(xmin, xmax)
		ax.set_ylim(ymin, ymax)
	plt.show()
	return fig, ax


def fft_phi_grid(
	system: System,
	t: float = 0.0,
	n: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Evaluate the effective planar drift field on a uniform periodic grid."""
	x = np.linspace(
		system.grid.xmin,
		system.grid.xmin + (system.grid.period or 2 * np.pi),
		n,
		endpoint=False,
	)
	y = np.linspace(
		system.grid.ymin,
		system.grid.ymin + (system.grid.period or 2 * np.pi),
		n,
		endpoint=False,
	)
	x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
	ex, ey = system.electric_field(t, x_mesh, y_mesh)
	return x_mesh, y_mesh, np.asarray(ey), np.asarray(-ex)


def plot_fft_phi(
	system: System,
	t: float = 0.0,
	n: int = 40,
	kind: Literal["quiver", "stream", "magnitude"] = "quiver",
	ax: Axes | None = None,
	show_magnitude: bool = True,
	density: float = 1.5,
	**kwargs: Any,
) -> tuple[Figure, Axes]:
	"""Plot the effective drift field of a Fourier or grid-backed System."""
	if kind not in {"quiver", "stream", "magnitude"}:
		raise ValueError("kind must be 'quiver', 'stream' or 'magnitude'.")
	x_mesh, y_mesh, vx, vy = fft_phi_grid(system, t=t, n=n)
	speed = np.hypot(vx, vy)
	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	else:
		fig = cast(Figure, ax.figure)
	if show_magnitude or kind == "magnitude":
		mesh = ax.pcolormesh(
			x_mesh.T,
			y_mesh.T,
			speed.T,
			shading="auto",
			cmap=kwargs.pop("cmap", "viridis"),
		)
		fig.colorbar(mesh, ax=ax, label=r"$|\dot{x}, \dot{y}|$")
	if kind == "quiver":
		default_kwargs: dict[str, Any] = {"pivot": "mid", "scale": None}
		default_kwargs.update(kwargs)
		ax.quiver(x_mesh.T, y_mesh.T, vx.T, vy.T, **default_kwargs)
	elif kind == "stream":
		default_kwargs = {"color": "white" if show_magnitude else None}
		default_kwargs.update(kwargs)
		if default_kwargs["color"] is None:
			default_kwargs.pop("color")
		ax.streamplot(
			x_mesh[:, 0],
			y_mesh[0, :],
			vx.T,
			vy.T,
			density=density,
			**default_kwargs,
		)
	ax.set_xlabel("$x$")
	ax.set_ylabel("$y$")
	ax.set_title("Effective drift field")
	ax.set_aspect("equal")
	ax.set_xlim(float(x_mesh.min()), float(x_mesh.max()))
	ax.set_ylim(float(y_mesh.min()), float(y_mesh.max()))
	return fig, ax


def plot_symplectic_poincare(
	system: System,
	solution: Solution,
) -> tuple[Figure, Axes]:
	"""Compatibility wrapper around the generic Poincare representation."""
	logger.info(
		"Plotting Poincare section: trajectory=%s modulo=%s",
		system.kind,
		getattr(system, "modulo", False),
	)
	result = SimulationResult(system=system, solution=solution, elapsed=0.0)
	fig, ax = result.plot_poincare()
	options = getattr(system, "options", None)
	if options is not None and bool(getattr(options, "save_plot", False)):
		output_dir = Path(getattr(options, "output_dir", "."))
		output_dir.mkdir(parents=True, exist_ok=True)
		output_name = str(getattr(options, "output_name", "notebook"))
		extension = str(getattr(options, "extension", ".png"))
		filename = output_dir / (
			f"{output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{extension}"
		)
		fig.savefig(filename, dpi=int(getattr(options, "dpi", 200)))
		logger.info("Figure saved in %s", filename)
	if "agg" not in plt.get_backend().lower():
		plt.pause(0.5)
	return fig, ax


__all__ = [
	"fft_phi_grid",
	"plot_fft_phi",
	"plot_poincare",
	"plot_potential",
	"plot_sol",
	"plot_symplectic_poincare",
]
