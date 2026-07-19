"""Persistence helpers for simulation solutions and figures."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from matplotlib.figure import Figure

from classes import System
from classes.system import Solution
from workflows.params import get_workflow_options

logger = logging.getLogger(__name__)


def timestamped_output_path(
	output_dir: str | Path,
	output_name: str,
	extension: str,
) -> Path:
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	return output_path / (
		f"{output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{extension}"
	)


def save_data(system: System, solution: Solution) -> None:
	"""Save a configured Fourier workflow result in the established NPZ layout."""
	options = get_workflow_options(system)
	if not options.save_data:
		logger.debug("Data export disabled; skipping NumPy export")
		return
	output_dir = Path(options.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	logger.info(
		"Saving simulation data: trajectory=%s samples=%d",
		system.kind,
		solution.t.size,
	)
	x, y = system.get_positions(solution.y)
	payload: dict[str, object] = dict(options.parameters)
	payload.update({"t": solution.t, "x": x, "y": y})
	velocities = system.get_velocities(solution.y)
	if velocities is not None:
		vx, vy = velocities
		payload.update({"vx": vx, "vy": vy})
	if hasattr(solution, "k"):
		payload["k"] = solution.k
	if hasattr(solution, "err"):
		payload["err"] = solution.err
	payload.update(
		{
			"date": datetime.now().strftime("%B %d, %Y"),
			"author": "cristel.chandre@cnrs.fr",
		}
	)
	filename = timestamped_output_path(
		output_dir,
		options.output_name,
		".npz",
	)
	np.savez_compressed(filename, **cast(dict[str, Any], payload))
	logger.info("Results saved in %s", filename)


def save_potential_data(
	solution: Solution,
	output_dir: str | Path,
	output_name: str,
) -> Path:
	"""Save a GridPotential simulation in a compact generic layout."""
	filename = timestamped_output_path(output_dir, output_name, ".npz")
	payload: dict[str, object] = {"t": solution.t, "y": solution.y}
	if hasattr(solution, "err"):
		payload["err"] = solution.err
	if hasattr(solution, "k"):
		payload["k"] = solution.k
	np.savez_compressed(filename, **cast(dict[str, Any], payload))
	logger.info("Potential results saved in %s", filename)
	return filename


def save_figure(
	fig: Figure,
	output_dir: str | Path,
	output_name: str,
	extension: str = ".png",
	dpi: int = 200,
) -> Path:
	filename = timestamped_output_path(output_dir, output_name, extension)
	fig.savefig(filename, dpi=dpi)
	logger.info("Figure saved in %s", filename)
	return filename


__all__ = [
	"save_data",
	"save_figure",
	"save_potential_data",
	"timestamped_output_path",
]
