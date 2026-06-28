#
# BSD 2-Clause License
#
# Copyright (c) 2023, Cristel Chandre
# All rights reserved.
#

import os

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import matplotlib.pyplot as plt
import multiprocess

from gc2d_dict import Parallelization, dict_list
from gc2d_notebook import run_case, to_symp_params


def _run_case(params: dict):
	result = run_case(params, plot=params.get('PlotResults', False))
	print(f"\033[92m   Integration of {result.system.__str__()} \033[00m")
	print(f'\033[90m        Computation finished in {int(result.elapsed)} seconds \033[00m')
	if result.system.CheckEnergy:
		print(f'\033[90m           with error in energy = {result.sol.err}')
	result.system.save_data(result.sol)


def main() -> None:
	if Parallelization == 'all':
		num_cores = multiprocess.cpu_count()
	else:
		num_cores = min(multiprocess.cpu_count(), Parallelization)

	params_list = [to_symp_params(dict_) for dict_ in dict_list]
	if num_cores >= 2:
		with multiprocess.Pool(num_cores) as pool:
			pool.map(_run_case, params_list)
	else:
		for params in params_list:
			_run_case(params)
	plt.show()


if __name__ == '__main__':
	main()
