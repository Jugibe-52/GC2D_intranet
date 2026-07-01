import logging
import time
from typing import Any

import numpy as np
from pyhamsys import solve_ivp_symp, solve_ivp_sympext

from classes.fourier_system import FourierSystem
from classes.simulation_result import SimulationResult
from config_logging import simulation_label
from workflows.params import ensure_system

logger = logging.getLogger(__name__)

GREEN = "\033[32m"
RESET = "\033[0m"


def integrate_simulation(case: FourierSystem | dict[str, Any]) -> SimulationResult:
	system = ensure_system(case)
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
