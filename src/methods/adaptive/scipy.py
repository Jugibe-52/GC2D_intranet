"""Prepare SciPy DOP853/Radau runs without duplicating global integration.

Each method run owns
one live SciPy solver, preserving its rejected trials, stage reuse and step-size
history. Accepted intervals and their dense interpolants use the same common
States history and coordinator as fixed methods.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np
from scipy.integrate import DOP853 as ScipyDOP853, Radau as ScipyRadau

from contracts.methods.quadrature import InterpolantQuadrature
from methods.base import IntegrationMethod
from contracts.methods.step import StepInfo, StepResult
from contracts.methods.adaptive.scipy import AdaptiveIntegrationStep
from request import DEFAULT_INTEGRATION_STEP, SimulationRequest
from methods._validation import _positive_finite


if TYPE_CHECKING:
    from execution.state import IntegrationState


@dataclass(frozen=True, slots=True)
class _AdaptiveDetails:
    """Dense output of one accepted step; never retained by States."""

    dense_state: Callable[[float | np.ndarray], np.ndarray]


class ScipyAdaptiveController:
    """Report the live method's accepted intervals and dense output samples."""

    sampling_tolerance = 0.0

    def steps(self, method: IntegrationMethod[_AdaptiveDetails], request: SimulationRequest) -> Iterator[tuple[StepInfo, StepResult[_AdaptiveDetails]]]:
        assert isinstance(method, _AdaptiveMethod)
        index = 0
        while method.solver.status == 'running':
            start: float = float(method.solver.t)
            before: np.ndarray = method.current_state.values.copy()
            result: StepResult[_AdaptiveDetails] = method.advance(start, before, method.max_step)
            end: float = float(method.solver.t)
            yield StepInfo(index, start, end - start, end, before), result
            index += 1

    def sample(self, method: IntegrationMethod[_AdaptiveDetails], info: StepInfo, result: StepResult[_AdaptiveDetails], time: float) -> IntegrationState:
        """Evaluate accepted dense output without advancing the solver."""
        from execution.state import IntegrationState
        return IntegrationState(time, result.details.dense_state(time), result.quadrature)


@dataclass(slots=True)
class _AdaptiveMethod(IntegrationMethod[_AdaptiveDetails]):
    """Shared controls and run operations; the concrete class selects the solver."""

    jax_mode = "hybrid_scipy_controller"

    relative_tolerance: float = 1e-10
    absolute_tolerance: float = 1e-12
    max_step: float = field(default=DEFAULT_INTEGRATION_STEP, kw_only=True)
    first_step: float | None = None
    dense_output: bool = False
    progress: bool = False
    solver_type: ClassVar[Any]
    order: ClassVar[int]

    # Resources owned by one run; excluded from constructor options.
    current_state: IntegrationState = field(init=False, repr=False, compare=False)
    solver: Any = field(init=False, repr=False, compare=False)
    previous_counts: np.ndarray = field(init=False, repr=False, compare=False)
    accepted_steps: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize tolerance controls and an optional first-step bound."""
        self.max_step = _positive_finite(self.max_step, "max_step")
        self.relative_tolerance = _positive_finite(
            self.relative_tolerance, 'relative_tolerance', allow_boolean=True,
            message='relative_tolerance must be finite and positive.',
        )
        self.absolute_tolerance = _positive_finite(
            self.absolute_tolerance, 'absolute_tolerance', allow_boolean=True,
            message='absolute_tolerance must be finite and positive.',
        )
        if self.first_step is not None:
            self.first_step = _positive_finite(
                self.first_step, 'first_step', allow_boolean=True,
                message='first_step must be finite and positive.',
            )

    def _backend_options(self) -> dict[str, Any]:
        """Provide solver-specific controls without branching in the coordinator."""
        return {}

    def initialize(self, request: SimulationRequest) -> None:
        """Initialize this run's vector field, physical snapshot and live solver."""
        from execution.state import IntegrationState

        self.current_state = IntegrationState(request.t_span[0], self.initial_state)
        self.metadata = {
            'step_control': 'adaptive', 'method_order': self.order,
            'relative_tolerance': self.relative_tolerance,
            'absolute_tolerance': self.absolute_tolerance,
            'dense_output': self.dense_output,
            'first_step': self.first_step if self.first_step is not None else 'automatic',
            'backend': 'scipy',
        }
        self.solver = self._make_solver(
            request.t_span[0], self.initial_state.copy(), request.t_span[1], self.max_step,
        )
        self.previous_counts = np.zeros(3, dtype=int)
        self.accepted_steps = 0

    def controller(self) -> ScipyAdaptiveController:
        """Use the live solver's accepted times and its dense interpolants."""
        return ScipyAdaptiveController()

    def _derivative(self, time: float, state: np.ndarray) -> np.ndarray:
        physical = state[:self.physical_size]
        field: np.ndarray = np.asarray(self.f(time, physical), dtype=float)
        if field.shape != physical.shape or not np.all(np.isfinite(field)):
            raise ValueError('The adaptive vector field changed shape or became non-finite.')
        return field

    def _make_solver(self, time: float, state: np.ndarray, end: float, max_step: float) -> Any:
        return self.solver_type(self._derivative, time, state, end, rtol=self.relative_tolerance,
                                atol=self.absolute_tolerance, max_step=max_step, first_step=self.first_step,
                                **self._backend_options())

    def advance(self, time: float, state: np.ndarray, step: float, *, observe: bool = True) -> StepResult[_AdaptiveDetails]:
        """Accept one adaptive step, with ``step`` as its upper size bound.

        The live solver retains rejected trials and stage reuse. The accepted
        end time is solver.t and can be earlier than time + step. Output-only
        interpolation work is included; extra observer work is excluded.
        """
        from execution.state import IntegrationState

        solver = self.solver
        if time != solver.t or not np.array_equal(self._checked_state(state), solver.y):
            raise ValueError('Adaptive advance must start at the live solver state.')
        if not np.isfinite(step) or step <= 0:
            raise ValueError('The adaptive step bound must be finite and positive.')
        if solver.status != 'running':
            raise RuntimeError('This adaptive run has no remaining accepted steps.')
        solver.max_step = step
        message: Any = solver.step()
        if solver.status == 'failed':
            raise RuntimeError(f'{type(solver).__name__} integration failed: {message}')
        counts: np.ndarray = np.asarray((solver.nfev, solver.njev, solver.nlu), dtype=int)
        work = counts - self.previous_counts
        end: float = float(solver.t)
        left: int | np.intp = 0 if self.accepted_steps == 0 else np.searchsorted(self.request.output_times, time, side='right')
        right: np.intp = np.searchsorted(self.request.output_times, end, side='right')
        interpolant: Any | None = None
        if self.dense_output or right > left or self.observe or self.capture_quadrature:
            interpolant = solver.dense_output()
            updated: np.ndarray = np.asarray((solver.nfev, solver.njev, solver.nlu), dtype=int)
            if self.dense_output or right > left:
                work += updated - counts
            counts = updated
        self.previous_counts = counts
        self.accepted_steps += 1

        def dense_state(query: float | np.ndarray) -> np.ndarray:
            if interpolant is None:
                raise RuntimeError('No dense output was requested for this interval.')
            return np.asarray(interpolant(query))

        after: np.ndarray = np.array(solver.y, copy=True)
        self.current_state = IntegrationState(float(self.solver.t), after)
        return StepResult(after, {
            'function_evaluations': int(work[0]),
            'jacobian_evaluations': int(work[1]),
            'lu_decompositions': int(work[2]),
        }, _AdaptiveDetails(dense_state),
            InterpolantQuadrature(time, end, dense_state) if self.capture_quadrature else None)

    def build_observation(self, info: StepInfo, result: StepResult[_AdaptiveDetails]) -> AdaptiveIntegrationStep:
        return AdaptiveIntegrationStep(
            dynamics_name="", method_name=type(self).__name__,
            step_index=info.index, start_time=info.time, time=info.end_time, duration=info.duration,
            state_before=info.state_before.copy(), state_after=result.state.copy(),
            dense_state=result.details.dense_state,
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
    # Optional physical derivative; None preserves SciPy finite differences.
    jacobian: Callable[[float, np.ndarray], np.ndarray] | None = None

    def _backend_options(self) -> dict[str, Any]:
        """Forward the optional physical-state Jacobian to the implicit solver."""
        if self.jacobian is not None and self.jacobian_structure == 'particle':
            from solvers.derivatives import packed_particle_jacobian
            derivative = self.jacobian
            return {'jac': lambda time, state: packed_particle_jacobian(np.asarray(derivative(time, state)))}
        return {'jac': self.jacobian}


__all__ = ['DOP853', 'Radau']
