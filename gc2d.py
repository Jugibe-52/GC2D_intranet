#
# BSD 2-Clause License
#
# Copyright (c) 2023, Cristel Chandre
# All rights reserved.
#

import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
import multiprocess

from gc2d_core.gc2d_dict import Parallelization, dict_list
from gc2d_core.logging_config import configure_logging, simulation_label
from gc2d_core.gc2d_notebook import run_case, to_symp_params

import logging

logger = logging.getLogger(__name__)


def _run_case(params: dict[str, Any]) -> None:
	logger.info("Starting case: %s", simulation_label(params))
	result = run_case(params, plot=params.get('PlotResults', False))
	logger.info("Finished case in %.2f seconds: %s", result.elapsed, simulation_label(params))
	if result.system.CheckEnergy:
		logger.info("Energy error: %s", result.sol.err)
	result.system.save_data(result.sol)


def main() -> None:
	configure_logging()
	if Parallelization == 'all':
		num_cores = multiprocess.cpu_count()
	else:
		num_cores = min(multiprocess.cpu_count(), Parallelization)

	params_list = [to_symp_params(dict_) for dict_ in dict_list]
	logger.info("Prepared %d case(s); parallelization=%s; workers=%d", len(params_list), Parallelization, num_cores)
	if num_cores >= 2:
		with multiprocess.Pool(num_cores) as pool:
			pool.map(_run_case, params_list)
	else:
		for params in params_list:
			_run_case(params)
	logger.info("All cases finished")
	plt.show()


if __name__ == '__main__':
	main()
