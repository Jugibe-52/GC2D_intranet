"""Shared spatial Hairer equations for palindromic extended-space maps."""
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

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


def solve_projection(
    maps: GCDoubledMaps, recipe: Composition, solver: AlgebraicSolver,
    formulation: ProjectionChoice, t: float, state: np.ndarray, h: float, *,
    jacobian_method: ResolvedJacobianMethod | None = None, observe: bool = False,
) -> ProjectedMapResult:
    """Assemble one exterior projection; the solver owns iteration and controls."""
    value: np.ndarray = np.asarray(state, dtype=float)
    if value.ndim != 1 or value.size != maps.physical_size or not np.all(np.isfinite(value)):
        raise ValueError('The projection requires a finite packed physical state.')
    size = value.size
    equations: ProjectionFormulation = resolve_projection(formulation)
    tolerance: float = solver.tolerance(value)
    context = f'{recipe.name} {equations.name} at t={t:.16g} with step={h:.16g}'
    derivative = jacobian_method or ('analytic' if solver.jacobian_method == 'auto' and maps.jacobian is not None
                                    else 'finite_difference' if solver.jacobian_method == 'auto'
                                    else solver.jacobian_method)

    def evaluate(unknown: np.ndarray) -> tuple[np.ndarray, CompositionTrace]:
        """Evaluate a spatial residual and retain that exact map's trace."""
        mu: Any = equations.multiplier(unknown, size)
        candidate: np.ndarray = maps.state_formulation.join_copies(value + mu, value - mu)
        trace: CompositionTrace = compose(maps, recipe, t, candidate, h)
        residual: Any = equations.residual(unknown, trace.state, size)
        return residual, trace

    def newton_update(unknown: np.ndarray, residual: np.ndarray, trace: CompositionTrace) -> np.ndarray:
        """Assemble the residual Jacobian from its complete-map derivative."""
        if derivative == 'analytic':
            base: np.ndarray = composition_jacobian(maps, trace)
        else:
            base = centered_difference(lambda candidate: compose(maps, recipe, t, candidate, h).state,
                                       trace.state_before, relative_step=solver.jacobian_relative_step)
        if derivative == 'analytic' and maps.jacobian_structure == 'particle':
            matrix: Any = equations.particle_jacobian(base)
            rhs: np.ndarray = residual.reshape(-1, maps.particle_count).T
            correction: np.ndarray = _solve_particle_systems(matrix, rhs,
                singular_message=f'The projection Jacobian is singular for {context}.').T.reshape(-1)
        else:
            matrix = equations.dense_jacobian(base, size)
            try:
                correction = np.linalg.solve(matrix, residual)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(f'The projection Jacobian is singular for {context}.') from exc
        return np.asarray(unknown - correction)

    def initial_map(candidate: np.ndarray) -> tuple[np.ndarray, CompositionTrace]:
        trace: CompositionTrace = compose(maps, recipe, t, candidate, h)
        return trace.state, trace

    initial: Any
    cached: Any
    initial, cached = equations.prepare(value, initial_map)
    if isinstance(solver, BroydenSolver):
        result: _NonlinearResult[Any] = solver.solve(evaluate, initial, equations.initial_jacobian(value),
            scale=value, context=context, initial_evaluation=cached)
    elif isinstance(solver, NewtonSolver):
        result = solver.solve(evaluate, initial, newton_update, scale=value,
            context=context, initial_evaluation=cached, time=t, duration=h,
            multiplier_slice=equations.multiplier_slice(size), observe=observe)
    else:
        raise ValueError('Unknown nonlinear solver for spatial projection.')
    mu = equations.multiplier(result.unknown, size)
    physical: Any = equations.reconstruct(result.unknown, result.payload.state, size)
    return ProjectedMapResult(t, h, value.copy(), physical, mu.copy(),
        SolveStats(solver.name, result.iterations, result.residual_evaluations,
            float(np.linalg.norm(result.residual, ord=np.inf)), tolerance), result.payload)
