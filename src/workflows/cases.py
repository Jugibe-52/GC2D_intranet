import logging
from typing import Any

from classes.fourier_system import FourierSystem
from classes.simulation_result import SimulationResult
from workflows.integration import integrate_case

logger = logging.getLogger(__name__)


def run_case(case: FourierSystem | dict[str, Any], plot: bool = True) -> SimulationResult:
	logger.info("Running notebook case: plot=%s", plot)
	result = integrate_case(case)
	method = getattr(result.system, 'Method', result.system.DictParams.get('Method', ''))
	if plot and method.startswith('poincare'):
		result.plot_poincare()
	elif plot:
		logger.debug("Skipping automatic plot for Method=%s", method)
	return result
