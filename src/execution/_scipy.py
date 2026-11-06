"""Prepare SciPy intervals and request dense output for scheduled samples."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import numpy as np

from contracts.methods.step import StepInfo, StepResult, StepValue
from execution.state import IntegrationState
from contracts.methods.adaptive.scipy import AdaptiveStepDetails
from methods.adaptive.scipy import _AdaptiveMethod
from methods.base import IntegrationMethod
from request import SimulationRequest


class ScipyAdaptiveController:
    """Keep interval setup and output scheduling outside numerical advancement."""

    sampling_tolerance = 0.0

    def prepare(self, method: _AdaptiveMethod, request: SimulationRequest) -> None:
        """Create one live solver with independent input for this time interval."""
        start: float
        end: float
        start, end = request.t_span
        method.current_state = IntegrationState(start, method.initial_state)
        method.solver = method._make_solver(start, method.initial_state.copy(), end, method.max_step)
        method.previous_counts = np.zeros(3, dtype=int)
        method.accepted_steps = 0

    def steps(
        self, method: IntegrationMethod[AdaptiveStepDetails], request: SimulationRequest,
    ) -> Iterator[tuple[StepInfo, StepResult[AdaptiveStepDetails]]]:
        assert isinstance(method, _AdaptiveMethod)
        self.prepare(method, request)
        index = 0
        while method.solver.status == 'running':
            start: float = float(method.solver.t)
            before: np.ndarray = method.current_state.values.copy()
            result: StepResult[AdaptiveStepDetails] = method.advance(start, before, method.max_step)
            end: float = float(method.solver.t)
            left: int = 0 if index == 0 else int(np.searchsorted(request.output_times, start, side='right'))
            right: int = int(np.searchsorted(request.output_times, end, side='right'))
            if right > left and not method.dense_output:
                # Observer-only interpolation is excluded from numerical work.
                # Count it when these same stages are needed for requested output.
                details: AdaptiveStepDetails = result.details
                if not details.has_interpolant:
                    details = method._capture_dense_output()
                    method.previous_counts = np.asarray(
                        (method.solver.nfev, method.solver.njev, method.solver.nlu), dtype=int)
                statistics: dict[str, StepValue] = dict(result.statistics)
                for key, extra in zip(
                    ('function_evaluations', 'jacobian_evaluations', 'lu_decompositions'),
                    details.interpolation_work,
                ):
                    statistics[key] = int(statistics[key]) + extra
                result = replace(result, details=details, statistics=statistics)
            yield StepInfo(index, start, end - start, end, before), result
            index += 1

    def sample(
        self, method: IntegrationMethod[AdaptiveStepDetails], info: StepInfo,
        result: StepResult[AdaptiveStepDetails], time: float,
    ) -> IntegrationState:
        """Evaluate a frozen interpolant without advancing the live solver."""
        return IntegrationState(time, result.details.dense_state(time), result.quadrature)
