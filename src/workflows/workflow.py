import logging
from typing import Any

from classes.fourier_system import FourierSystem
from classes.simulation_result import SimulationResult
from workflows.export import save_data, save_figure
from workflows.integration import integrate_case

logger = logging.getLogger(__name__)


def run_workflow(case: FourierSystem | dict[str, Any], plot: bool = True, save: bool = True) -> SimulationResult:
	logger.info("Running notebook case: plot=%s save=%s", plot, save)
	result = integrate_case(case)
	method = getattr(result.system, 'Method', result.system.DictParams.get('Method', ''))
	if plot and method.startswith('poincare'):
		fig, _ = result.plot_poincare()
		if getattr(result.system, "SavePlot", False):
			save_figure(
				fig,
				getattr(result.system, "output_dir", "."),
				getattr(result.system, "output_name", "notebook"),
				extension=getattr(result.system, "extension", ".png"),
				dpi=int(getattr(result.system, "dpi", 200)),
			)
	elif plot:
		logger.debug("Skipping automatic plot for Method=%s", method)
	if save:
		save_data(result.system, result.sol)
	return result
