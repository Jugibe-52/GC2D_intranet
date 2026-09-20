"""Prepare SciPy DOP853/Radau runs without duplicating global integration.

Each method run owns
one live SciPy solver, preserving its rejected trials, stage reuse and step-size
history. Accepted intervals and their dense interpolants use the same common
collector and coordinator as fixed methods.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from scipy.integrate import DOP853 as ScipyDOP853, Radau as ScipyRadau

from dynamics import DynamicalSystem, ExtendedHamiltonianSystem

from ..._result import DiagnosticValue
from ...formulations.base import generalized_energy_error
from ...integration import IntegrationMethod, StepInfo, StepResult
from ...observation import AdaptiveIntegrationStep, AdaptiveStepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest


@dataclass(frozen=True, slots=True)
class _AdaptiveDetails:
    """Dense output of one accepted step; never retained by the collector."""

    dense_state: Callable[[float | np.ndarray], np.ndarray]


class ScipyAdaptiveController:
    """Report the live method's accepted intervals and dense output samples."""

    sample_endpoints = True
    sampling_tolerance = 0.0

    def steps(self, method: IntegrationMethod[_AdaptiveDetails], request: SimulationRequest) -> Iterator[tuple[StepInfo, StepResult[_AdaptiveDetails]]]:
        assert isinstance(method, _AdaptiveMethod)
        index = 0
        while method.solver.status == 'running':
            start = float(method.solver.t)
            before = np.asarray(method.solver.y).copy()
            result = method.advance(start, before, request.max_step)
            end = float(method.solver.t)
            yield StepInfo(index, start, end - start, end, before), result
            index += 1

    def sample(self, method: IntegrationMethod[_AdaptiveDetails], info: StepInfo, result: StepResult[_AdaptiveDetails], time: float) -> np.ndarray:
        """Evaluate accepted dense output without advancing the solver."""
        return result.details.dense_state(time)


@dataclass(slots=True)
class _AdaptiveMethod(IntegrationMethod[_AdaptiveDetails]):
    """Shared controls and run operations; the concrete class selects the solver."""

    relative_tolerance: float = 1e-10
    absolute_tolerance: float = 1e-12
    first_step: float | None = None
    dense_output: bool = False
    track_energy: bool = False
    progress: bool = False
    step_observer: AdaptiveStepObserver | None = None
    solver_type: ClassVar[Any]
    order: ClassVar[int]

    # Resources owned by one run; excluded from constructor options.
    solver: Any = field(init=False, repr=False, compare=False)
    previous_counts: np.ndarray = field(init=False, repr=False, compare=False)
    accepted_steps: int = field(init=False, repr=False, compare=False)
    dynamics: DynamicalSystem = field(init=False, repr=False, compare=False)
    physical_size: int = field(init=False, repr=False, compare=False)
    particle_count: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ('relative_tolerance', 'absolute_tolerance', 'first_step'):
            value = getattr(self, name)
            if value is None and name == 'first_step':
                continue
            value = float(value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive.')
            setattr(self, name, value)

    def _backend_options(self) -> dict[str, Any]:
        """Provide solver-specific controls without branching in the coordinator."""
        return {}

    def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
        """Initialize this run's vector field, internal state and live solver."""
        self.dynamics = problem.dynamics
        if not isinstance(self.dynamics, DynamicalSystem):
            raise TypeError(f'{type(self).__name__} requires DynamicalSystem.')
        if self.track_energy and not isinstance(self.dynamics, ExtendedHamiltonianSystem):
            raise TypeError('Energy tracking requires ExtendedHamiltonianSystem.')
        self.physical_size = problem.initial_state.size
        self.particle_count = problem.particle_count
        initial = problem.initial_state
        if self.track_energy:
            initial = np.concatenate((initial, np.zeros(self.particle_count)))
        self.initial_state = initial
        self.metadata = {
            'step_control': 'adaptive', 'method_order': self.order,
            'relative_tolerance': self.relative_tolerance,
            'absolute_tolerance': self.absolute_tolerance,
            'track_energy': self.track_energy, 'dense_output': self.dense_output,
            'first_step': self.first_step if self.first_step is not None else 'automatic',
            'backend': 'scipy',
        }
        self.solver = self._make_solver(
            request.t_span[0], self.initial_state.copy(), request.t_span[1], request.max_step,
        )
        self.previous_counts = np.zeros(3, dtype=int)
        self.accepted_steps = 0

    def controller(self) -> ScipyAdaptiveController:
        """Use the live solver's accepted times and its dense interpolants."""
        return ScipyAdaptiveController()

    def _derivative(self, time: float, state: np.ndarray) -> np.ndarray:
        physical = state[:self.physical_size]
        field = np.asarray(self.dynamics.vector_field(time, physical), dtype=float)
        if field.shape != physical.shape or not np.all(np.isfinite(field)):
            raise ValueError('The adaptive vector field changed shape or became non-finite.')
        if not self.track_energy:
            return field
        assert isinstance(self.dynamics, ExtendedHamiltonianSystem)
        momentum = np.asarray(self.dynamics.extended_momentum_derivative(time, physical), dtype=float)
        if momentum.shape != (self.particle_count,) or not np.all(np.isfinite(momentum)):
            raise ValueError('The momentum derivative must be finite with one value per particle.')
        return np.concatenate((field, momentum))

    def _make_solver(self, time: float, state: np.ndarray, end: float, max_step: float) -> Any:
        return self.solver_type(self._derivative, time, state, end, rtol=self.relative_tolerance,
                                atol=self.absolute_tolerance, max_step=max_step, first_step=self.first_step,
                                **self._backend_options())

    def advance(self, time: float, state: np.ndarray, step: float) -> StepResult[_AdaptiveDetails]:
        """Accept one adaptive step, with ``step`` as its upper size bound.

        The live solver retains rejected trials and stage reuse. The accepted
        end time is solver.t and can be earlier than time + step. Output-only
        interpolation work is included; extra observer work is excluded.
        """
        solver = self.solver
        if time != solver.t or not np.array_equal(state, solver.y):
            raise ValueError('Adaptive advance must start at the live solver state.')
        if not np.isfinite(step) or step <= 0:
            raise ValueError('The adaptive step bound must be finite and positive.')
        if solver.status != 'running':
            raise RuntimeError('This adaptive run has no remaining accepted steps.')
        solver.max_step = step
        message = solver.step()
        if solver.status == 'failed':
            raise RuntimeError(f'{type(solver).__name__} integration failed: {message}')
        counts = np.asarray((solver.nfev, solver.njev, solver.nlu), dtype=int)
        work = counts - self.previous_counts
        end = float(solver.t)
        left = 0 if self.accepted_steps == 0 else np.searchsorted(self.request.output_times, time, side='right')
        right = np.searchsorted(self.request.output_times, end, side='right')
        interpolant = None
        if self.dense_output or right > left or self.step_observer is not None:
            interpolant = solver.dense_output()
            updated = np.asarray((solver.nfev, solver.njev, solver.nlu), dtype=int)
            if self.dense_output or right > left:
                work += updated - counts
            counts = updated
        self.previous_counts = counts
        self.accepted_steps += 1

        def dense_state(query: float | np.ndarray) -> np.ndarray:
            if interpolant is None:
                raise RuntimeError('No dense output was requested for this interval.')
            return np.asarray(interpolant(query)).copy()

        return StepResult(np.asarray(solver.y).copy(), {
            'function_evaluations': int(work[0]),
            'jacobian_evaluations': int(work[1]),
            'lu_decompositions': int(work[2]),
        }, _AdaptiveDetails(dense_state))

    def build_observation(self, info: StepInfo, result: StepResult[_AdaptiveDetails]) -> AdaptiveIntegrationStep:
        return AdaptiveIntegrationStep(
            dynamics_name=type(self.dynamics).__name__, method_name=type(self).__name__,
            step_index=info.index, start_time=info.time, time=info.end_time, duration=info.duration,
            state_before=info.state_before.copy(), state_after=result.state.copy(),
            dense_state=result.details.dense_state,
            function_evaluations=int(result.statistics['function_evaluations']),
            jacobian_evaluations=int(result.statistics['jacobian_evaluations']),
            lu_decompositions=int(result.statistics['lu_decompositions']),
        )

    def export_history(self, times: np.ndarray, history: np.ndarray) -> tuple[np.ndarray, dict[str, DiagnosticValue]]:
        states = history[:self.physical_size]
        auxiliary: dict[str, DiagnosticValue] = {}
        if self.track_energy:
            momentum = history[self.physical_size:]
            auxiliary['extended_momentum'] = momentum
            auxiliary['energy_error'] = generalized_energy_error(times, states, momentum, self.dynamics)
        return states, auxiliary


@dataclass(slots=True)
class DOP853(_AdaptiveMethod):
    """Explicit order-eight adaptive Runge--Kutta with SciPy dense output."""

    solver_type: ClassVar[Any] = ScipyDOP853
    order: ClassVar[int] = 8


@dataclass(slots=True)
class Radau(_AdaptiveMethod):
    """Implicit order-five Radau IIA with SciPy's Newton and adaptive control."""

    solver_type: ClassVar[Any] = ScipyRadau
    order: ClassVar[int] = 5
    # The Jacobian must cover the complete integrated state, including auxiliary
    # momentum when enabled. None selects SciPy's finite-difference Jacobian.
    jacobian: Callable[[float, np.ndarray], np.ndarray] | None = None

    def _backend_options(self) -> dict[str, Any]:
        """Forward the optional full-state Jacobian to the implicit solver."""
        return {'jac': self.jacobian}


__all__ = ['DOP853', 'Radau']
