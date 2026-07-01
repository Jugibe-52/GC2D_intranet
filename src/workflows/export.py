import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from pyhamsys import OdeSolution
from scipy.io import savemat

from classes.fourier_system import FourierSystem

logger = logging.getLogger(__name__)


def save_data(system: FourierSystem, sol: OdeSolution) -> None:
	if not system.SaveData:
		logger.debug("SaveData disabled; skipping MATLAB export")
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
	mdic = system.DictParams.copy()
	mdic.update({'t': sol.t, 'x': x, 'y': y})
	if system.traj_type == 'fo':
		mdic.update({'vx': vx, 'vy': vy})
	if system.CheckEnergy:
		mdic.update({'k': sol.k})
	mdic.update({'date': datetime.now().strftime(" %B %d, %Y\n"), 'author': 'cristel.chandre@cnrs.fr'})
	output_name = getattr(system, "output_name", "notebook")
	filename = output_dir / f'{output_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mat'
	savemat(filename, mdic)
	logger.info("Results saved in %s", filename)
