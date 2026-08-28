"""Simulation orchestration independent of concrete systems and methods."""

from __future__ import annotations

import numpy as np

from .methods import NumericalMethod
from .problem import InitialValueProblem
from .request import SimulationRequest
from .solution import Solution


class SimulationRunner:
	"""Validate, integrate, and build one trajectory-like physical solution."""

	def simulate(
		self,
		problem: InitialValueProblem,
		method: NumericalMethod,
		request: SimulationRequest,
	) -> Solution:
		"""Run a compatible numerical method and attach the source configuration."""
		if not isinstance(problem, InitialValueProblem):
			raise TypeError("`problem` must be an InitialValueProblem.")
		if not isinstance(method, NumericalMethod):
			raise TypeError("`method` must implement NumericalMethod.")
		if not isinstance(request, SimulationRequest):
			raise TypeError("`request` must be a SimulationRequest.")
		data = method.integrate(problem, request)
		times = np.asarray(data.t, dtype=float)
		states = np.asarray(data.states)
		initial_state = problem.initial_state
		if (
			times.shape != request.output_times.shape
			or not np.array_equal(times, request.output_times)
			or states.ndim != 2
			or states.shape[0] != initial_state.size
			or states.shape[1] != times.size
			or not np.all(np.isfinite(states))
		):
			raise ValueError("The numerical method returned incompatible physical output.")
		if not np.array_equal(states[:, 0], initial_state):
			raise ValueError("The numerical method did not preserve the initial state.")
		problem.initial_configuration.validate_packed_state_layout(states)
		return Solution(
			t=times,
			states=states,
			source=problem.initial_configuration,
			diagnostics=data.diagnostics,
		)


def simulate(
	problem: InitialValueProblem,
	method: NumericalMethod,
	request: SimulationRequest,
) -> Solution:
	"""Convenience façade over :class:`SimulationRunner`."""
	return SimulationRunner().simulate(problem, method, request)


__all__ = ["SimulationRunner", "simulate"]
