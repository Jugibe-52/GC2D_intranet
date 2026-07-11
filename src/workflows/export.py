import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from matplotlib.figure import Figure
from pyhamsys import OdeSolution

from classes.fourier_system import FourierSystem

logger = logging.getLogger(__name__)


def timestamped_output_path(output_dir: str | Path, output_name: str, extension: str) -> Path:
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	return output_path / f'{output_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}{extension}'


def save_data(system: FourierSystem, sol: OdeSolution) -> None:
	if not getattr(system, "SaveData", False):
		logger.debug("SaveData disabled; skipping NumPy export")
		return
	output_dir = Path(getattr(system, "output_dir", "."))
	output_dir.mkdir(parents=True, exist_ok=True)
	logger.info("Saving simulation data: traj=%s samples=%d", system.traj_type, sol.t.size)
	if system.traj_type == 'gc':
		x, y = np.split(sol.y, 2)
	elif system.traj_type == 'fo':
		if system.CheckEnergy:
			x, y, vx, vy, _ = np.split(sol.y, 5)
		else:
			x, y, vx, vy = np.split(sol.y, 4)
	payload: dict[str, object] = dict(system.DictParams)
	payload.update({'t': sol.t, 'x': x, 'y': y})
	if system.traj_type == 'fo':
		payload.update({'vx': vx, 'vy': vy})
	if system.CheckEnergy:
		payload.update({'k': sol.k})
	payload.update({'date': datetime.now().strftime("%B %d, %Y"), 'author': 'cristel.chandre@cnrs.fr'})
	output_name = getattr(system, "output_name", "notebook")
	filename = timestamped_output_path(output_dir, output_name, ".npz")
	np.savez_compressed(filename, **cast(dict[str, Any], payload))
	logger.info("Results saved in %s", filename)


def save_potential_data(sol: OdeSolution, output_dir: str | Path, output_name: str) -> Path:
	"""Save a PotentialSystem solution in NumPy format."""
	filename = timestamped_output_path(output_dir, output_name, ".npz")
	payload: dict[str, object] = {"t": sol.t, "y": sol.y}
	if hasattr(sol, "err"):
		payload["err"] = sol.err
	if hasattr(sol, "k"):
		payload["k"] = sol.k
	np.savez_compressed(filename, **cast(dict[str, Any], payload))
	logger.info("Potential results saved in %s", filename)
	return filename


def save_figure(fig: Figure, output_dir: str | Path, output_name: str, extension: str = ".png", dpi: int = 200) -> Path:
	filename = timestamped_output_path(output_dir, output_name, extension)
	fig.savefig(filename, dpi=dpi)
	logger.info("Figure saved in %s", filename)
	return filename
