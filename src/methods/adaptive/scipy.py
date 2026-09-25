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

from dynamics import DynamicalSystem

from formulations.state import PhysicalFormulation
from integration.core import IntegrationMethod
from contracts.step import StepInfo, StepResult
from contracts.observation import AdaptiveIntegrationStep, AdaptiveStepObserver
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest


# Fixed eight-point Gauss quadrature of -partial_t H along accepted dense output.
# This is diagnostic work and never contributes to physical solver statistics.
_nodes, _weights = np.polynomial.legendre.leggauss(8)
_ENERGY_NODES = (_nodes + 1.) / 2.
_ENERGY_WEIGHTS = _weights / 2.


@dataclass(frozen=True, slots=True)
class _AdaptiveDetails:
    """Dense output of one accepted step; never retained by the collector."""

    dense_state: Callable[[float | np.ndarray], np.ndarray]


class ScipyAdaptiveController:
    """Report the live method's accepted intervals and dense output samples."""

    sampling_tolerance = 0.0

    def steps(self, method: IntegrationMethod[_AdaptiveDetails], request: SimulationRequest) -> Iterator[tuple[StepInfo, StepResult[_AdaptiveDetails]]]:
        assert isinstance(method, _AdaptiveMethod)
        index = 0
        while method.solver.status == 'running':
            start = float(method.solver.t)
            before = method.current_state.copy()
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
    state_formulation: PhysicalFormulation = field(init=False, repr=False, compare=False)
    current_state: np.ndarray = field(init=False, repr=False, compare=False)
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
        self.physical_size = problem.initial_state.size
        self.particle_count = problem.particle_count
        self.state_formulation = PhysicalFormulation(problem, request.t_span[0], self.track_energy)
        self.initial_state = self.state_formulation.initial_state
        self.current_state = self.initial_state.copy()
        self.metadata = {
            'step_control': 'adaptive', 'method_order': self.order,
            'relative_tolerance': self.relative_tolerance,
            'absolute_tolerance': self.absolute_tolerance,
            'track_energy': self.track_energy, 'dense_output': self.dense_output,
            'first_step': self.first_step if self.first_step is not None else 'automatic',
            'backend': 'scipy', 'energy_quadrature_nodes': 8 if self.track_energy else 0,
        }
        self.solver = self._make_solver(
            request.t_span[0], problem.initial_state, request.t_span[1], request.max_step,
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
        return field

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
        if time != solver.t or not np.array_equal(self.state_formulation.physical(state), solver.y):
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
        if self.dense_output or right > left or self.step_observer is not None or self.track_energy:
            interpolant = solver.dense_output()
            updated = np.asarray((solver.nfev, solver.njev, solver.nlu), dtype=int)
            if self.dense_output or right > left:
                work += updated - counts
            counts = updated
        self.previous_counts = counts
        self.accepted_steps += 1

        # Diagnostic quadrature is evaluated only after acceptance. SciPy owns
        # a purely physical state, so energy cannot affect its error norm or Newton.
        momentum_before = self.state_formulation.momentum(state)
        if momentum_before is not None:
            momentum_before = momentum_before.copy()
        momentum_after: np.ndarray | None = None

        def momentum_at(query: float | np.ndarray) -> np.ndarray | None:
            """Integrate interior queries while reusing the two endpoint momenta."""
            if momentum_before is None:
                return None
            assert interpolant is not None
            queries = np.asarray(query, dtype=float)
            values = []
            for endpoint in queries.reshape(-1):
                # Endpoint sampling and observers must not repeat accepted quadrature.
                if endpoint == time:
                    values.append(momentum_before)
                    continue
                if endpoint == end and momentum_after is not None:
                    values.append(momentum_after)
                    continue
                duration = float(endpoint - time)
                rates = [self.state_formulation.momentum_rate(time + duration * node,
                             np.asarray(interpolant(time + duration * node))) for node in _ENERGY_NODES]
                values.append(momentum_before + duration * sum(w * rate for w, rate in zip(_ENERGY_WEIGHTS, rates)))
            if not values:
                return np.empty((self.particle_count, *queries.shape))
            return np.asarray(np.stack(values, axis=-1).reshape(self.particle_count, *queries.shape))

        def dense_state(query: float | np.ndarray) -> np.ndarray:
            if interpolant is None:
                raise RuntimeError('No dense output was requested for this interval.')
            return self.state_formulation.pack(np.asarray(interpolant(query)), query, momentum_at(query))

        momentum_after = momentum_at(end)
        after = self.state_formulation.pack(np.asarray(solver.y), end, momentum_after)
        self.current_state = after.copy()
        return StepResult(after, {
            'function_evaluations': int(work[0]),
            'jacobian_evaluations': int(work[1]),
            'lu_decompositions': int(work[2]),
        }, _AdaptiveDetails(dense_state))

    def build_observation(self, info: StepInfo, result: StepResult[_AdaptiveDetails]) -> AdaptiveIntegrationStep:
        return AdaptiveIntegrationStep(
            dynamics_name=type(self.dynamics).__name__, method_name=type(self).__name__,
            step_index=info.index, start_time=info.time, time=info.end_time, duration=info.duration,
            state_before=self.state_formulation.physical(info.state_before).copy(), state_after=self.state_formulation.physical(result.state).copy(),
            dense_state=lambda query: self.state_formulation.physical(result.details.dense_state(query)),
            function_evaluations=int(result.statistics['function_evaluations']),
            jacobian_evaluations=int(result.statistics['jacobian_evaluations']),
            lu_decompositions=int(result.statistics['lu_decompositions']),
        )


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
    # The Jacobian covers physical coordinates only, also with energy tracking.
    # None selects SciPy's finite-difference Jacobian.
    jacobian: Callable[[float, np.ndarray], np.ndarray] | None = None

    def _backend_options(self) -> dict[str, Any]:
        """Forward the optional physical-state Jacobian to the implicit solver."""
        return {'jac': self.jacobian}


__all__ = ['DOP853', 'Radau']
