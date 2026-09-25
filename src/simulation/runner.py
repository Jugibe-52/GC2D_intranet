"""Simulation orchestration independent of concrete systems and methods."""

from __future__ import annotations

import numpy as np

from methods.base import NumericalMethod
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from solution import Solution


def simulate(
	problem: InitialValueProblem,
	method: NumericalMethod,
	request: SimulationRequest,
) -> Solution:
	"""Integrate one problem and verify that its solution matches the request.

	Solution owns structural validation and immutable output storage. This
	boundary checks agreement with the requested times and initial state.
	"""
	if not isinstance(problem, InitialValueProblem):
		raise TypeError("`problem` must be an InitialValueProblem.")
	if not isinstance(method, NumericalMethod):
		raise TypeError("`method` must implement NumericalMethod.")
	if not isinstance(request, SimulationRequest):
		raise TypeError("`request` must be a SimulationRequest.")
	data = method.integrate(problem, request)
	solution = Solution(
		t=data.t,
		states=data.states,
		source=problem.initial_configuration,
		diagnostics=data.diagnostics,
	)
	initial_state = problem.initial_state
	if (
		not np.array_equal(solution.t, request.output_times)
		or solution.states.shape[0] != initial_state.size
	):
		raise ValueError("The numerical method returned incompatible physical output.")
	if not np.array_equal(solution.states[:, 0], initial_state):
		raise ValueError("The numerical method did not preserve the initial state.")
	return solution


__all__ = [
	"simulate",
]
