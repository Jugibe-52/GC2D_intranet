"""Compatibility workflow delegating numerical work to System.simulate."""

from __future__ import annotations

import logging

from classes import System
from config_logging import simulation_label
from workflows.export import save_data
from workflows.integration import integrate_simulation
from workflows.params import get_workflow_options
from workflows.plotting import plot_symplectic_poincare

logger = logging.getLogger(__name__)


def run_method(system: System) -> None:
	"""Run an already composed System through the shared integration workflow."""
	options = get_workflow_options(system)
	logger.info("Starting case: %s", simulation_label(options.parameters))
	result = integrate_simulation(system)
	logger.info(
		"Finished case in %.2f seconds: %s",
		result.elapsed,
		simulation_label(options.parameters),
	)
	if options.check_energy and hasattr(result.solution, "err"):
		logger.info("Energy error: %s", result.solution.err)
	save_data(system, result.solution)
	if options.workflow_method.startswith("poincare") and options.plot_results:
		plot_symplectic_poincare(system, result.solution)


__all__ = ["run_method"]
