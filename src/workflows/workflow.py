"""Reusable high-level simulation workflow."""

from __future__ import annotations

from collections.abc import Mapping
import logging

from classes import System
from classes.system import SimulationResult
from workflows.export import save_data, save_figure
from workflows.integration import integrate_simulation
from workflows.params import get_workflow_options

logger = logging.getLogger(__name__)


def run_workflow(
	case: System | Mapping[str, object],
	plot: bool = True,
	save: bool = True,
) -> SimulationResult:
	"""Integrate, optionally plot, and optionally persist a Fourier case."""
	logger.info("Running workflow: plot=%s save=%s", plot, save)
	result = integrate_simulation(case)
	options = get_workflow_options(result.system)
	if plot and options.workflow_method.startswith("poincare"):
		fig, _ = result.plot_poincare()
		if options.save_plot:
			save_figure(
				fig,
				options.output_dir,
				options.output_name,
				extension=options.extension,
				dpi=options.dpi,
			)
	elif plot:
		logger.debug(
			"Skipping automatic plot for Method=%s",
			options.workflow_method,
		)
	if save:
		save_data(result.system, result.solution)
	return result


__all__ = ["run_workflow"]
