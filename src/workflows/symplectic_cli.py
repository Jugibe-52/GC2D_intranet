import argparse
import logging

import multiprocess

from classes.fourier_system import FourierSystem
from config import DEFAULT_CONFIG_VERSION, load_fourier_config
from config_logging import configure_logging
from workflows.symplectic_legacy import run_method

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the legacy FourierSystem symplectic workflow from configuration.")
	parser.add_argument("--config", help="Path to the Python or JSON configuration file.")
	parser.add_argument("--config-group", default="assay", choices=("test", "assay"), help="Configuration group under conf/.")
	parser.add_argument("--config-version", default=DEFAULT_CONFIG_VERSION, help="Configuration file version under conf/fourier/<group>/, e.g. v_1.")
	parser.add_argument("--version", default="symplectic_grid", help="Profile inside the configuration file.")
	return parser.parse_args()


def _run_legacy_case(params: dict) -> None:
	run_method(FourierSystem(params))


def main() -> None:
	args = parse_args()
	configure_logging()
	config = load_fourier_config(
		args.config,
		version=args.version,
		config_group=args.config_group,
		config_version=args.config_version,
	)
	params_list = config.cases()
	parallelization = config.parallelization
	if parallelization == 'all':
		num_cores = multiprocess.cpu_count()
	else:
		num_cores = min(multiprocess.cpu_count(), int(parallelization))
	logger.info(
		"Prepared %d case(s) from config version=%s; parallelization=%s; workers=%d",
		len(params_list),
		config.version,
		parallelization,
		num_cores,
	)
	if num_cores >= 2:
		with multiprocess.Pool(num_cores) as pool:
			pool.map(_run_legacy_case, params_list)
	else:
		for params in params_list:
			_run_legacy_case(params)
	logger.info("All cases finished")
