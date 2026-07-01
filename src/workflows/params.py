import logging
from typing import Any

import numpy as np

from classes.fourier_system import FourierSystem
from config_logging import simulation_label

logger = logging.getLogger(__name__)


def to_symp_params(params: dict[str, Any]) -> dict[str, Any]:
	params = params.copy()
	method = params.get('Method', 'poincare_gc')
	traj_type = params.get('traj_type', method.rsplit('_', 1)[-1])
	if traj_type not in {'gc', 'fo'}:
		raise ValueError(f"Cannot infer trajectory type from Method={method!r}.")
	params['traj_type'] = traj_type
	params.setdefault('eta', params.get('rho', 0))
	if traj_type == 'fo' and params['eta'] == 0:
		raise ValueError("Full-orbit integrations require a non-zero `eta` parameter.")
	logger.debug("Normalized parameters: %s eta=%s", simulation_label(params), params.get('eta'))
	return params


def make_params(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
	params = base.copy()
	params.update(overrides)
	logger.debug("Preparing notebook parameters with overrides=%s", sorted(overrides))
	if params.get('init') == 'selected':
		n_traj = params.get('Ntraj')
		if 'x0' not in overrides and 'x0' in params:
			logger.debug("Trimming selected x0 initial conditions to Ntraj=%s", n_traj)
			params['x0'] = np.asarray(params['x0'])[:n_traj]
		if 'y0' not in overrides and 'y0' in params:
			logger.debug("Trimming selected y0 initial conditions to Ntraj=%s", n_traj)
			params['y0'] = np.asarray(params['y0'])[:n_traj]
	params = to_symp_params(params)
	logger.info("Prepared notebook parameters: %s init=%s", simulation_label(params), params.get('init'))
	return params


def make_system(params: dict[str, Any]) -> FourierSystem:
	params = to_symp_params(params)
	logger.info("Building system: %s", simulation_label(params))
	return FourierSystem(params)


def ensure_system(case: FourierSystem | dict[str, Any]) -> FourierSystem:
	if isinstance(case, FourierSystem):
		return case
	return make_system(case)
