"""Shared spatial Hairer equations for palindromic extended-space maps."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING
import numpy as np
from typing import cast
from contracts.arrays import BackendArray
from solvers.linear import solve_linear
from types import ModuleType
from contracts.arrays import array_namespace, checked_finite

from methods.extended.maps import GCDoubledMaps
from solvers.linear import _solve_particle_systems
from solvers.nonlinear import SolveStats, NewtonSolver, BroydenSolver, AlgebraicSolver
from solvers.derivatives import ResolvedJacobianMethod, centered_difference
from methods.extended.core.composition import Composition, compose
from methods.projection import ProjectionChoice, resolve_projection
from methods.extended.core.jacobians import composition_jacobian
from methods.extended.core.records import CompositionTrace, ProjectedMapResult

if TYPE_CHECKING:
    from contracts.methods.projection import ProjectionFormulation
    from solvers.nonlinear import _NonlinearResult
    from typing import Any


@dataclass(frozen=True, slots=True)
class ProjectionProblem:
    """Complete spatial root equations with the current step bound as context.

    ``residual(x)`` returns a vector with the same packing as x: mu (2N), or
    corrected final copies and mu (6N). ``jacobian(x)`` differentiates that
    complete function, including the dependence of both computed copies on mu.
    Analytic independent-particle derivatives use (N,2,2) or (N,6,6) blocks;
    otherwise the returned matrix is dense in component-major ordering.
    """

    maps: GCDoubledMaps
    recipe: Composition
    equations: ProjectionFormulation
    time: float
    state: BackendArray
    duration: float
    derivative: ResolvedJacobianMethod
    jacobian_relative_step: float
    # Only short-lived solver adapters bind this pair, never a shared cache.
    _evaluation: tuple[BackendArray, CompositionTrace] | None = field(default=None, repr=False, compare=False)

    @property
    def particle_blocks(self) -> bool:
        return self.derivative == 'analytic' and self.maps.jacobian_structure == 'particle'

    def _trace(self, unknown: BackendArray) -> CompositionTrace:
        expected = self.equations.unknown_components * self.maps.particle_count
        if unknown.ndim != 1 or unknown.size != expected:
            raise ValueError(f'The projection unknown must be a vector of length {expected}.')
        if self._evaluation is not None and unknown is self._evaluation[0]:
            return self._evaluation[1]
        mu: Any = self.equations.multiplier(unknown, self.state.size)
        candidate: BackendArray = self.maps.state_formulation.join_copies(self.state + mu, self.state - mu)
        return compose(self.maps, self.recipe, self.time, candidate, self.duration)

    def _at(self, unknown: BackendArray, trace: CompositionTrace) -> ProjectionProblem:
        """Bind an exact evaluation for one immediate, immutable adapter call.

        A different candidate always recomputes its own composition. The base
        problem keeps no candidate or trace, including during JAX tracing.
        """
        return replace(self, _evaluation=(unknown, trace))

    def residual(self, unknown: BackendArray) -> BackendArray:
        """Compute the copies for x and return only the complete residual R(x)."""
        trace: CompositionTrace = self._trace(unknown)
        return cast(BackendArray, self.equations.assemble_residual(unknown, trace.state, self.state.size))

    def jacobian(self, unknown: BackendArray) -> BackendArray:
        """Return D_x R(x), retaining independent blocks when available."""
        trace: CompositionTrace = self._trace(unknown)
        base: BackendArray
        if self.derivative == 'analytic':
            base = composition_jacobian(self.maps, trace)
        else:
            base = centered_difference(
                lambda candidate: compose(self.maps, self.recipe, self.time, candidate, self.duration).state,
                trace.state_before, relative_step=self.jacobian_relative_step)
        if self.particle_blocks:
            return cast(BackendArray, self.equations.particle_jacobian(base))
        return cast(BackendArray, self.equations.dense_jacobian(base, self.state.size))

    def evaluate(self, unknown: BackendArray) -> tuple[BackendArray, CompositionTrace]:
        """Adapt residual(x) to the solver's retained-composition payload."""
        trace: CompositionTrace = self._trace(unknown)
        return self._at(unknown, trace).residual(unknown), trace


def solve_projection(
    maps: GCDoubledMaps, recipe: Composition, solver: AlgebraicSolver,
    formulation: ProjectionChoice, t: float, state: BackendArray, h: float, *,
    jacobian_method: ResolvedJacobianMethod | None = None, observe: bool = False,
    composition_name: str = 'custom',
) -> ProjectedMapResult:
    """Assemble one exterior projection; the solver owns iteration and controls."""
    xp: ModuleType = array_namespace(state)
    value: BackendArray = xp.asarray(state, dtype=float)
    value = checked_finite(value, 'The projection requires a finite packed physical state.', ValueError)
    if value.ndim != 1 or value.size != maps.physical_size:
        raise ValueError('The projection requires a finite packed physical state.')
    size = value.size
    equations: ProjectionFormulation = resolve_projection(formulation)
    tolerance: float = solver.tolerance(value)
    context = f'{composition_name} {equations.name} at t={t} with step={h}'
    derivative = jacobian_method or ('analytic' if solver.jacobian_method == 'auto' and maps.jacobian is not None
                                    else 'finite_difference' if solver.jacobian_method == 'auto'
                                    else solver.jacobian_method)

    problem: ProjectionProblem = ProjectionProblem(maps, recipe, equations, t, value, h, derivative, solver.jacobian_relative_step)

    def newton_update(unknown: BackendArray, residual: BackendArray, trace: CompositionTrace) -> BackendArray:
        """Use jacobian(x) with the same composition that produced residual(x)."""
        matrix: BackendArray = problem._at(unknown, trace).jacobian(unknown)
        if problem.particle_blocks:
            rhs: BackendArray = residual.reshape(-1, maps.particle_count).T
            correction: BackendArray = _solve_particle_systems(matrix, rhs,
                singular_message=f'The projection Jacobian is singular for {context}.').T.reshape(-1)
        else:
            correction = solve_linear(matrix, residual, singular_message=f'The projection Jacobian is singular for {context}.')
        return cast(BackendArray, xp.asarray(unknown - correction))

    def initial_map(candidate: BackendArray) -> tuple[BackendArray, CompositionTrace]:
        trace: CompositionTrace = compose(maps, recipe, t, candidate, h)
        return trace.state, trace

    initial: Any
    cached: Any
    initial, cached = equations.prepare(value, initial_map)
    if isinstance(solver, BroydenSolver):
        result: _NonlinearResult[Any] = solver.solve(problem.evaluate, initial, equations.initial_jacobian(value),
            scale=value, context=context, initial_evaluation=cached)
    elif isinstance(solver, NewtonSolver):
        result = solver.solve(problem.evaluate, initial, newton_update, scale=value,
            context=context, initial_evaluation=cached, time=t, duration=h,
            multiplier_slice=equations.multiplier_slice(size), observe=observe)
    else:
        raise ValueError('Unknown nonlinear solver for spatial projection.')
    mu = equations.multiplier(result.unknown, size)
    physical: Any = equations.reconstruct(result.unknown, result.payload.state, size)
    return ProjectedMapResult(t, h, value.copy(), physical, mu.copy(),
        SolveStats(solver.name, result.iterations, result.residual_evaluations,
            xp.linalg.norm(result.residual, ord=np.inf), tolerance, result.converged), result.payload)
