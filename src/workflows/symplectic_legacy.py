import logging
import time
from typing import Any

import numpy as np
from pyhamsys import solve_ivp_symp, solve_ivp_sympext

from config_logging import simulation_label
from workflows.export import save_data
from workflows.plotting import plot_symplectic_poincare

logger = logging.getLogger(__name__)


def run_method(system: Any) -> None:
	logger.info("Starting case: %s", simulation_label(system.DictParams))
	start = time.time()
	y0 = system.initial_conditions(type=system.init)
	logger.info("Initial conditions ready: shape=%s init=%s", y0.shape, system.init)
	t_eval = 2 * np.pi * np.arange(0, system.Tf + 1)
	logger.info("Starting integration: solver=%s step=%s samples=%d", system.ode_solver, system.TimeStep, len(t_eval))
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
	elif system.traj_type == 'fo':
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
	logger.info("Finished case in %.2f seconds: %s", time.time() - start, simulation_label(system.DictParams))
	if system.CheckEnergy:
		logger.info("Energy error: %s", sol.err)
	save_data(system, sol)
	if getattr(system, 'Method', '').startswith('poincare') and getattr(system, 'PlotResults', False):
		plot_symplectic_poincare(system, sol)
