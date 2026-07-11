#
# BSD 2-Clause License
#
# Copyright (c) 2023, Cristel Chandre
# All rights reserved.
#

import os
import sys
from pathlib import Path
import argparse

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
import multiprocess

from config import DEFAULT_CONFIG_GROUP, DEFAULT_CONFIG_VERSION, load_fourier_config
from config_logging import configure_logging, simulation_label
from contracts import FourierParams
from workflows_api import run_workflow, save_data

import logging

logger = logging.getLogger(__name__)


def _run_workflow(params: FourierParams) -> None:
	logger.info("Starting case: %s", simulation_label(params))
	result = run_workflow(params, plot=params.get('PlotResults', False), save=False)
	logger.info("Finished case in %.2f seconds: %s", result.elapsed, simulation_label(params))
	if result.system.CheckEnergy:
		logger.info("Energy error: %s", result.sol.err)
	save_data(result.system, result.sol)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run FourierSystem cases from a JSON configuration.")
	parser.add_argument("--config", help="Path to the JSON configuration file.")
	parser.add_argument("--config-surface", default="terminal", choices=("terminal", "notebook"), help="Configuration surface under conf/.")
	parser.add_argument("--config-group", default=DEFAULT_CONFIG_GROUP, choices=("test", "assay"), help="Configuration group under conf/.")
	parser.add_argument("--config-version", default=DEFAULT_CONFIG_VERSION, help="Configuration file version under conf/<surface>/fourier/<group>/, e.g. v_1.")
	parser.add_argument("--version", help="Profile inside the configuration file.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	configure_logging()
	config = load_fourier_config(
		args.config,
		version=args.version,
		config_surface=args.config_surface,
		config_group=args.config_group,
		config_version=args.config_version,
	)
	if config.output_dir is not None:
		config.output_dir.mkdir(parents=True, exist_ok=True)
		logger.info("Output directory: %s", config.output_dir)
	parallelization = config.parallelization
	if parallelization == 'all':
		num_cores = multiprocess.cpu_count()
	else:
		num_cores = min(multiprocess.cpu_count(), int(parallelization))

	params_list = config.cases()
	logger.info(
		"Prepared %d case(s) from config version=%s; parallelization=%s; workers=%d",
		len(params_list),
		config.version,
		parallelization,
		num_cores,
	)
	if num_cores >= 2:
		with multiprocess.Pool(num_cores) as pool:
			pool.map(_run_workflow, params_list)
	else:
		for params in params_list:
			_run_workflow(params)
	logger.info("All cases finished")
	plt.show()


if __name__ == '__main__':
	main()
