"""High-level integration workflow for composed systems."""

from __future__ import annotations

from collections.abc import Mapping
import logging
import time

import numpy as np

from classes import System
from classes.system import SimulationResult
from config_logging import simulation_label
from workflows.params import ensure_system, get_workflow_options

logger = logging.getLogger(__name__)


def integrate_simulation(
	case: System | Mapping[str, object],
) -> SimulationResult:
	"""Integrate a Fourier case exclusively through System.simulate."""
	system = ensure_system(case)
	options = get_workflow_options(system)
	initial_state = system.initial_state()
	logger.info(
		"Initial conditions ready: shape=%s init=%s",
		initial_state.shape,
		system.trajectory.initialization,
	)
	final_time = 2 * np.pi * options.periods
	sample_count = options.periods + 1
	logger.info(
		"Starting integration: %s system=%s solver=%s step=%s samples=%d",
		simulation_label(options.parameters),
		type(system).__name__,
		options.solver_method,
		options.time_step,
		sample_count,
	)
	start = time.time()
	solution = system.simulate(
		initial_state,
		t_span=(0.0, final_time),
		step=options.time_step,
		n_save_step=sample_count,
		method=options.solver_method,
		check_energy=options.check_energy,
	)
	elapsed = time.time() - start
	logger.info(
		"Integration finished in %.2f seconds: %s solution_shape=%s",
		elapsed,
		simulation_label(options.parameters),
		solution.y.shape,
	)
	if options.check_energy and hasattr(solution, "err"):
		logger.info("Energy error: %s", solution.err)
	return SimulationResult(system=system, solution=solution, elapsed=elapsed)


__all__ = ["integrate_simulation"]
