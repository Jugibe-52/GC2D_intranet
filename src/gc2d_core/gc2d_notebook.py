import os
import time
import logging
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from pyhamsys import solve_ivp_symp, solve_ivp_sympext

from .gc2d_symp import GC2Dt
from .logging_config import simulation_label

logger = logging.getLogger(__name__)

GREEN = "\033[32m"
RESET = "\033[0m"


@dataclass
class SimulationResult:
	system: GC2Dt
	sol: Any
	elapsed: float
	fig: Any = None
	ax: Any = None

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
		ax: Any = None,
		grid: bool | None = None,
		decimal_grid: bool = False,
		grid_step: float = 0.5,
		**plot_kwargs: Any,
	) -> tuple[Any, Any]:
		system = self.system
		logger.info("Plotting Poincare section: traj=%s modulo=%s", system.traj_type, getattr(system, 'modulo', False) if modulo is None else modulo)
		if ax is None:
			fig, ax = plt.subplots(1, 1, figsize=(6, 6))
		else:
			fig = ax.figure
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
		default_kwargs = {'markersize': 3 if system.traj_type == 'gc' else 1, 'markeredgecolor': 'none'}
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


def make_system(params: dict[str, Any]) -> GC2Dt:
	params = to_symp_params(params)
	logger.info("Building system: %s", simulation_label(params))
	return GC2Dt(params)


def _ensure_system(case: GC2Dt | dict[str, Any]) -> GC2Dt:
	if isinstance(case, GC2Dt):
		return case
	return make_system(case)


def integrate_case(case: GC2Dt | dict[str, Any]) -> SimulationResult:
	system = _ensure_system(case)
	y0 = system.initial_conditions(type=system.init)
	logger.info("Initial conditions ready: shape=%s init=%s", y0.shape, system.init)
	t_eval = 2 * np.pi * np.arange(0, system.Tf + 1)
	logger.info(
		"Starting integration: %s solver=%s step=%s samples=%d",
		simulation_label(system.DictParams),
		system.ode_solver,
		system.TimeStep,
		len(t_eval),
	)
	start = time.time()
	if system.traj_type == 'gc':
		logger.info("%sUsing guiding-center integrator: solve_ivp_sympext%s", GREEN, RESET)
		logger.info(
			"%ssolve_ivp_sympext parameters: t_span=%s y0_shape=%s step=%s "
			"t_eval_shape=%s method=%s check_energy=%s%s",
			GREEN,
			(0, t_eval.max()),
			y0.shape,
			system.TimeStep,
			t_eval.shape,
			system.ode_solver,
			system.CheckEnergy,
			RESET,
		)
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
		logger.info("Using full-orbit integrator: solve_ivp_symp + rectify_sol")
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
	logger.info("Integration finished in %.2f seconds: %s solution_shape=%s", elapsed, simulation_label(system.DictParams), sol.y.shape)
	if system.CheckEnergy:
		logger.info("Energy error: %s", sol.err)
	return SimulationResult(system=system, sol=sol, elapsed=elapsed)


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


def run_case(case: GC2Dt | dict[str, Any], plot: bool = True) -> SimulationResult:
	logger.info("Running notebook case: plot=%s", plot)
	result = integrate_case(case)
	method = getattr(result.system, 'Method', result.system.DictParams.get('Method', ''))
	if plot and method.startswith('poincare'):
		result.plot_poincare()
	elif plot:
		logger.debug("Skipping automatic plot for Method=%s", method)
	return result
