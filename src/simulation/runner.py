"""Simulation orchestration independent of concrete systems and methods."""

from __future__ import annotations

import numpy as np

from methods.base import NumericalMethod
from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from execution.execution import Execution
from solution import Solution


def _validate_simulation_inputs(
	problem: InitialValueProblem,
	method: NumericalMethod,
	request: SimulationRequest,
	execution: Execution | None,
) -> None:
	"""Check the public orchestration boundary before selecting its executor."""
	if not isinstance(problem, InitialValueProblem):
		raise TypeError("`problem` must be an InitialValueProblem.")
	if not isinstance(method, NumericalMethod):
		raise TypeError("`method` must implement NumericalMethod.")
	if not isinstance(request, SimulationRequest):
		raise TypeError("`request` must be a SimulationRequest.")
	if execution is not None and not isinstance(execution, Execution):
		raise TypeError("`execution` must be an Execution instance or None.")


def _validate_solution_matches_request(
	solution: Solution,
	problem: InitialValueProblem,
	request: SimulationRequest,
) -> None:
	"""Check requested sampling and the preserved initial physical state."""
	initial_state = problem.initial_state
	if (
		not np.array_equal(solution.t, request.output_times)
		or solution.states.shape[0] != initial_state.size
	):
		raise ValueError("The numerical method returned incompatible physical output.")
	if not np.array_equal(solution.states[:, 0], initial_state):
		raise ValueError("The numerical method did not preserve the initial state.")


def simulate(
	problem: InitialValueProblem,
	method: NumericalMethod,
	request: SimulationRequest,
	*, execution: Execution | None = None,
	options: ExecutionOptions | None = None,
) -> Solution:
	"""Integrate one problem and verify that its solution matches the request.

	Solution owns structural validation and immutable output storage. This
	boundary checks agreement with the requested times and initial state.
	The executor receives the complete job; backend options are separate from
	where it runs. Omission selects local NumPy/SciPy execution.
	"""
	_validate_simulation_inputs(problem, method, request, execution)
	executor = Execution() if execution is None else execution
	data = executor.run(problem, method, request, options=options)
	solution = Solution(
		t=data.t,
		states=data.states,
		source=problem.initial_configuration,
		diagnostics=data.diagnostics,
	)
	_validate_solution_matches_request(solution, problem, request)
	return solution


__all__ = [
	"simulate",
]
