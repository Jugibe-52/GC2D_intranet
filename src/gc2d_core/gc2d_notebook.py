import os
import time
import logging
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from pyhamsys import solve_ivp_symp, solve_ivp_sympext

from .gc2d_symp import GC2Dt
from .logging_config import simulation_label

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
	system: GC2Dt
	sol: object
	elapsed: float
	fig: object = None
	ax: object = None


def to_symp_params(params: dict) -> dict:
	params = params.copy()
	method = params.get('Method', 'poincare_gc')
	traj_type = params.get('traj_type', method.rsplit('_', 1)[-1])
	if traj_type not in {'gc', 'fo'}:
		raise ValueError(f"Cannot infer trajectory type from Method={method!r}.")
	params['traj_type'] = traj_type
	params.setdefault('eta', params.get('rho', 0))
	if traj_type == 'fo' and params['eta'] == 0:
		raise ValueError("Full-orbit integrations require a non-zero `eta` parameter.")
	return params


def make_params(base: dict, **overrides) -> dict:
	params = base.copy()
	params.update(overrides)
	if params.get('init') == 'selected':
		n_traj = params.get('Ntraj')
		if 'x0' not in overrides and 'x0' in params:
			params['x0'] = np.asarray(params['x0'])[:n_traj]
		if 'y0' not in overrides and 'y0' in params:
			params['y0'] = np.asarray(params['y0'])[:n_traj]
	return to_symp_params(params)


def integrate_case(params: dict) -> SimulationResult:
	params = to_symp_params(params)
	logger.info("Building system: %s", simulation_label(params))
	system = GC2Dt(params)
	y0 = system.initial_conditions(type=system.init)
	logger.info("Initial conditions ready: shape=%s init=%s", y0.shape, system.init)
	t_eval = 2 * np.pi * np.arange(0, system.Tf + 1)
	logger.info(
		"Starting integration: %s solver=%s step=%s samples=%d",
		simulation_label(params),
		system.ode_solver,
		system.TimeStep,
		len(t_eval),
	)
	start = time.time()
	if system.traj_type == 'gc':
		sol = solve_ivp_sympext(
			system,
			(0, t_eval.max()),
			y0,
			step=system.TimeStep,
			t_eval=t_eval,
			method=system.ode_solver,
			check_energy=system.CheckEnergy,
		)
	else:
		sol = solve_ivp_symp(
			system.chi,
			system.chi_star,
			(0, t_eval.max()),
			y0,
			step=system.TimeStep,
			t_eval=t_eval,
			method=system.ode_solver,
		)
		sol = system.rectify_sol(sol, check_energy=system.CheckEnergy)
	elapsed = time.time() - start
	logger.info("Integration finished in %.2f seconds: %s", elapsed, simulation_label(params))
	if system.CheckEnergy:
		logger.info("Energy error: %s", sol.err)
	return SimulationResult(system=system, sol=sol, elapsed=elapsed)


def plot_poincare(result: SimulationResult, modulo: bool = None, ax=None, **plot_kwargs):
	system, sol = result.system, result.sol
	logger.info("Plotting Poincare section: traj=%s modulo=%s", system.traj_type, getattr(system, 'modulo', False) if modulo is None else modulo)
	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	else:
		fig = ax.figure
	if system.traj_type == 'gc':
		x, y = np.split(sol.y, 2)
	else:
		x, y = np.split(sol.y, 4)[:2]
	use_modulo = getattr(system, 'modulo', False) if modulo is None else modulo
	if use_modulo:
		x, y = x % (2 * np.pi), y % (2 * np.pi)
		ax.set_xlim(0, 2 * np.pi)
		ax.set_ylim(0, 2 * np.pi)
		ax.set_xticks([0, np.pi, 2 * np.pi])
		ax.set_yticks([0, np.pi, 2 * np.pi])
		ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
		ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
	default_kwargs = {'markersize': 3 if system.traj_type == 'gc' else 1, 'markeredgecolor': 'none'}
	default_kwargs.update(plot_kwargs)
	ax.plot(x, y, '.', **default_kwargs)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	ax.set_aspect('equal')
	result.fig = fig
	result.ax = ax
	return fig, ax


def run_case(params: dict, plot: bool = True) -> SimulationResult:
	result = integrate_case(params)
	if plot and params.get('Method', '').startswith('poincare'):
		plot_poincare(result)
	return result
