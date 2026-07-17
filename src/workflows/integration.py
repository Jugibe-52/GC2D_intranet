import logging
import time
from collections.abc import Mapping

import numpy as np
from pyhamsys import solve_ivp_symp, solve_ivp_sympext

from classes.fourier_system import FourierSystem
from classes.simulation_result import SimulationResult
from config_logging import simulation_label
from workflows.params import ensure_system

logger = logging.getLogger(__name__)

GREEN = "\033[32m"
RESET = "\033[0m"


def integrate_simulation(case: FourierSystem | Mapping[str, object]) -> SimulationResult:
	system = ensure_system(case)
	y0 = system.initial_conditions(type=system.init)
	logger.info("Initial conditions ready: shape=%s init=%s", y0.shape, system.init)
	period = 2 * np.pi
	t_final = period * system.Tf
	sample_count = system.Tf + 1
	logger.info(
		"Starting integration: %s solver=%s step=%s samples=%d",
		simulation_label(system.DictParams),
		system.ode_solver,
		system.TimeStep,
		sample_count,
	)
	start = time.time()
	if system.traj_type == 'gc':
		logger.info("%sUsing guiding-center integrator: solve_ivp_sympext%s", GREEN, RESET)
		logger.info(
			"%ssolve_ivp_sympext parameters: t_span=%s y0_shape=%s step=%s "
			"n_save_step=%s method=%s check_energy=%s%s",
			GREEN,
			(0, t_final),
			y0.shape,
			system.TimeStep,
			sample_count,
			system.ode_solver,
			system.CheckEnergy,
			RESET,
		)
		sol = solve_ivp_sympext(
			system,
			y0,
			step=system.TimeStep,
			t_span=(0, t_final),
			n_save_step=sample_count,
			method=system.ode_solver,
			check_energy=system.CheckEnergy,
		)
	else:
		logger.info("Using full-orbit integrator: solve_ivp_symp + rectify_sol")
		sol = solve_ivp_symp(
			system.chi,
			system.chi_star,
			(0, t_final),
			y0,
			step=system.TimeStep,
			n_save_step=sample_count,
			method=system.ode_solver,
		)
		sol = system.rectify_sol(sol, check_energy=system.CheckEnergy)
	elapsed = time.time() - start
	logger.info("Integration finished in %.2f seconds: %s solution_shape=%s", elapsed, simulation_label(system.DictParams), sol.y.shape)
	if system.CheckEnergy:
		logger.info("Energy error: %s", sol.err)
	return SimulationResult(system=system, sol=sol, elapsed=elapsed)
