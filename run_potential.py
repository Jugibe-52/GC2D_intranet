"""Run GC or FC trajectories over an HDF5 or mock GridPotential."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import (  # noqa: E402
	DEFAULT_CONFIG_GROUP,
	DEFAULT_CONFIG_VERSION,
	load_potential_config,
)
from config_logging import configure_logging  # noqa: E402
from workflows.export import save_figure, save_potential_data  # noqa: E402
from workflows_api import plot_sol  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run GC or FC trajectories over an HDF5/mock potential."
	)
	parser.add_argument("--config", help="Path to the JSON configuration file.")
	parser.add_argument(
		"--config-surface",
		default="terminal",
		choices=("terminal", "notebook"),
		help="Configuration surface under conf/.",
	)
	parser.add_argument(
		"--config-group",
		default=DEFAULT_CONFIG_GROUP,
		choices=("test", "assay"),
		help="Configuration group under conf/.",
	)
	parser.add_argument(
		"--config-version",
		default=DEFAULT_CONFIG_VERSION,
		help="Configuration version under conf/<surface>/potential/<group>/.",
	)
	parser.add_argument("--version", help="Profile inside the configuration file.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	configure_logging()
	config = load_potential_config(
		args.config,
		version=args.version,
		config_surface=args.config_surface,
		config_group=args.config_group,
		config_version=args.config_version,
	)
	if config.output_dir is not None:
		config.output_dir.mkdir(parents=True, exist_ok=True)
		logger.info("Output directory: %s", config.output_dir)

	system = config.build_system()
	initial_state = system.trajectory.state
	if initial_state is None:
		raise ValueError("Configured system has an uninitialized trajectory.")
	logger.info(
		"Generated initial conditions: version=%s trajectory=%s Ntraj=%d "
		"init=%s shape=%s",
		config.version,
		system.kind,
		system.trajectory.n_trajectories,
		system.trajectory.initialization,
		initial_state.shape,
	)

	n_max = int(config.integration.get("n_max", 50))
	time_step = float(config.solver.get("TimeStep", 2e-2))
	solver_method = str(config.solver.get("ode_solver", "BM4"))
	check_energy = bool(config.solver.get("CheckEnergy", True))
	final_time = 2 * np.pi * (n_max - 1)

	logger.info(
		"Starting integration: n_max=%d time_step=%s solver=%s",
		n_max,
		time_step,
		solver_method,
	)
	start = time.time()
	solution = system.simulate(
		t_span=(0.0, final_time),
		step=time_step,
		n_save_step=n_max,
		method=solver_method,
		check_energy=check_energy,
	)
	logger.info(
		"Finished integration in %.2f seconds; solution shape=%s",
		time.time() - start,
		solution.y.shape,
	)
	if hasattr(solution, "err"):
		logger.info("Energy error: %s", solution.err)

	output = config.output
	if output.get("data", False):
		save_potential_data(
			solution,
			config.output_dir or ".",
			config.output_name or config.version,
		)
	if output.get("plot", True):
		logger.info("Plotting solution")
		fig, _ = plot_sol(
			system,
			solution,
			wrap=output.get("wrap", True),
		)
		if config.output_dir is not None:
			save_figure(
				fig,
				config.output_dir,
				config.output_name or config.version,
				extension=output.get("extension", ".png"),
				dpi=int(output.get("dpi", 200)),
			)


if __name__ == "__main__":
	main()
