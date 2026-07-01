###################################################################################################
##                      Parameters: https://github.com/cchandre/guiding_center_intranet                    ##
###################################################################################################

import argparse
import numpy as np
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pyhamsys import solve_ivp_symp, solve_ivp_sympext

from config import DEFAULT_CONFIG_GROUP, DEFAULT_CONFIG_VERSION, load_potential_config
from config_logging import configure_logging
from workflows_api import plot_sol

import logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the PotentialSystem HDF5/mock potential showcase from JSON.")
	parser.add_argument("--config", help="Path to the JSON configuration file.")
	parser.add_argument("--config-group", default=DEFAULT_CONFIG_GROUP, choices=("test", "assay"), help="Configuration group under conf/.")
	parser.add_argument("--config-version", default=DEFAULT_CONFIG_VERSION, help="Configuration folder version under conf/<group>/, e.g. v_1.")
	parser.add_argument("--version", help="Profile inside the JSON file.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	configure_logging()
	config = load_potential_config(
		args.config,
		version=args.version,
		config_group=args.config_group,
		config_version=args.config_version,
	)
	hs = config.build_system()
	n_traj = config.initial_condition_count()
	init = config.initial_condition_type()
	z0 = hs.initial_conditions(n_traj, type=init)
	traj_type = hs.traj["type"]
	logger.info(
		"Generated initial conditions: version=%s traj_type=%s Ntraj=%d init=%s shape=%s",
		config.version,
		traj_type,
		n_traj,
		init,
		z0.shape,
	)

	# plot_potential(hs)

	integration = config.integration
	n_max = int(integration.get("n_max", 50))
	time_step = float(integration.get("TimeStep", 2e-2))
	ode_solver = integration.get("ode_solver", "BM4")
	check_energy = bool(integration.get("CheckEnergy", True))
	t_eval = 2 * np.pi * np.arange(n_max)

	# lyap = hs.compute_lyapunov(2 * np.pi * n_max, z0, reortho_dt=1, tol=1e-10, solver='RK45')
	# print(lyap)

	# Poincare section
	logger.info("Starting integration: n_max=%d time_step=%s solver=%s", n_max, time_step, ode_solver)
	start = time.time()
	if traj_type == "gc":
		sol = solve_ivp_sympext(hs, (t_eval.min(), t_eval.max()), z0, step=time_step, t_eval=t_eval, method=ode_solver, check_energy=check_energy)
	else:
		sol = solve_ivp_symp(hs.chi, hs.chi_star, (t_eval.min(), t_eval.max()), z0, step=time_step, t_eval=t_eval, method=ode_solver)
		sol = hs.rectify_sol(sol, check_energy=check_energy)
	logger.info("Finished integration in %.2f seconds; solution shape=%s", time.time() - start, sol.y.shape)
	if hasattr(sol, "err"):
		logger.info("Energy error: %s", sol.err)
	if integration.get("plot", True):
		logger.info("Plotting solution")
		plot_sol(hs, sol, wrap=integration.get("wrap", True))


if __name__ == '__main__':
	main()
