import logging
from collections.abc import Mapping
from typing import cast

import numpy as np

from classes.fourier_system import FourierSystem
from config_logging import simulation_label
from contracts import FourierParams, InitialConditionKind, ParameterMap, TrajectoryKind

logger = logging.getLogger(__name__)


def _int_param(value: object, name: str) -> int:
	if isinstance(value, (int, np.integer)):
		return int(value)
	if isinstance(value, (float, np.floating, str)):
		return int(value)
	raise TypeError(f"`{name}` must be numeric, got {type(value).__name__}.")


def _float_param(value: object, name: str) -> float:
	if isinstance(value, (int, float, np.integer, np.floating, str)):
		return float(value)
	raise TypeError(f"`{name}` must be numeric, got {type(value).__name__}.")


def to_symp_params(raw_params: Mapping[str, object]) -> FourierParams:
	"""Validate and normalize an untyped parameter mapping."""
	params: ParameterMap = dict(raw_params)
	method = params.get('Method', 'poincare_gc')
	if not isinstance(method, str):
		raise TypeError(f"`Method` must be a string, got {type(method).__name__}.")
	traj_type = params.get('traj_type', method.rsplit('_', 1)[-1])
	if traj_type not in {'gc', 'fo'}:
		raise ValueError(f"Cannot infer trajectory type from Method={method!r}.")
	params['traj_type'] = traj_type
	# eta controls the higher-order effective-potential correction.  It is
	# independent of the Larmor radius rho, so omitted GC configurations start
	# with the leading-order (eta = 0) model.
	params.setdefault('eta', 0)
	if traj_type == 'fo' and params['eta'] == 0:
		raise ValueError("Full-orbit integrations require a non-zero `eta` parameter.")
	params.setdefault('init', 'fixed')
	params.setdefault('TimeStep', 0.1 if traj_type == 'gc' else 0.005)
	params.setdefault('ode_solver', 'BM4')
	params.setdefault('CheckEnergy', True)
	required = ('M', 'A', 'rho', 'eta', 'Ntraj', 'Tf')
	missing = [key for key in required if key not in params]
	if missing:
		raise ValueError(f"Missing required Fourier parameters: {', '.join(missing)}.")
	params['M'] = _int_param(params['M'], 'M')
	params['A'] = _float_param(params['A'], 'A')
	params['rho'] = _float_param(params['rho'], 'rho')
	params['eta'] = _float_param(params['eta'], 'eta')
	params['Ntraj'] = _int_param(params['Ntraj'], 'Ntraj')
	params['Tf'] = _int_param(params['Tf'], 'Tf')
	params['TimeStep'] = _float_param(params['TimeStep'], 'TimeStep')
	params['ode_solver'] = str(params['ode_solver'])
	params['CheckEnergy'] = bool(params['CheckEnergy'])
	init = params['init']
	if init not in {'random', 'fixed', 'selected'}:
		raise ValueError(f"Invalid initial-condition type: {init!r}.")
	params['init'] = init
	logger.debug("Normalized parameters: %s eta=%s", simulation_label(params), params.get('eta'))
	return cast(FourierParams, params)


def make_params(base: Mapping[str, object], **overrides: object) -> FourierParams:
	params: ParameterMap = dict(base)
	params.update(overrides)
	logger.debug("Preparing notebook parameters with overrides=%s", sorted(overrides))
	if params.get('init') == 'selected':
		n_traj = _int_param(params.get('Ntraj', 0), 'Ntraj')
		if 'x0' not in overrides and 'x0' in params:
			logger.debug("Trimming selected x0 initial conditions to Ntraj=%s", n_traj)
			params['x0'] = np.asarray(params['x0'])[:n_traj]
		if 'y0' not in overrides and 'y0' in params:
			logger.debug("Trimming selected y0 initial conditions to Ntraj=%s", n_traj)
			params['y0'] = np.asarray(params['y0'])[:n_traj]
	normalized = to_symp_params(params)
	logger.info("Prepared notebook parameters: %s init=%s", simulation_label(normalized), normalized.get('init'))
	return normalized


def make_system(params: Mapping[str, object]) -> FourierSystem:
	params = to_symp_params(params)
	logger.info("Building system: %s", simulation_label(params))
	return FourierSystem(params)


def ensure_system(case: FourierSystem | Mapping[str, object]) -> FourierSystem:
	if isinstance(case, FourierSystem):
		return case
	return make_system(case)
