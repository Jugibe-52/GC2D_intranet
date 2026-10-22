"""Shared spatial Hairer equations for palindromic extended-space maps."""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING
import numpy as np

from contracts.nonlinear import NewtonObserver
from formulations.gc import GCDoubledMaps
from solvers.linear import _solve_particle_systems
from solvers.nonlinear import SolverOptions, SolveStats, _bind_newton_observer, configured_solver, NewtonSolver, BroydenSolver, AlgebraicSolver
from methods.extended.core.composition import Composition, compose
from formulations.projection import ProjectionChoice, resolve_projection
from methods.extended.core.jacobians import central_difference_jacobian, particle_jacobians
from methods.extended.core.records import CompositionTrace, ProjectedMapResult


if TYPE_CHECKING:
    from formulations.projection import ProjectionFormulation
    from solvers.nonlinear import _NonlinearResult
    from typing import Any


def solve_projection(
    maps: GCDoubledMaps, recipe: Composition, options: SolverOptions,
    formulation: ProjectionChoice,
    t: float, state: np.ndarray, h: float, *,
    jacobian_method: Literal['analytic', 'finite_difference'] = 'analytic',
    jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps)),
    retain_energy_points: bool = False,
    newton_observer: NewtonObserver | None = None,
    solver: AlgebraicSolver | None = None,
) -> ProjectedMapResult:
    """Solve the selected equation around one complete unprojected recipe.

    Reduced unknowns are the physical multiplier ``mu`` (2N components).
    Simultaneous unknowns are ``(u_out, v_out, mu)`` (6N components). Time and
    passive momentum never enter either root or its state-scaled tolerance.
    """
    value: np.ndarray = np.asarray(state, dtype=float)
    if value.ndim != 1 or value.size != maps.physical_size or not np.all(np.isfinite(value)):
        raise ValueError("The projection requires a finite packed physical state.")
    size = value.size
    equations: ProjectionFormulation = resolve_projection(formulation)
    selected_solver = solver or configured_solver(options.solver, options.absolute_tolerance,
        options.relative_tolerance, options.max_iterations)
    tolerance: float = options.tolerance(value)
    context = f'{recipe.name} {equations.name} at t={t:.16g} with step={h:.16g}'
    if newton_observer is not None and options.solver != 'newton':
        raise ValueError('A newton_observer requires nonlinear_solver="newton".')

    def evaluate(unknown: np.ndarray) -> tuple[np.ndarray, CompositionTrace]:
        """Evaluate a spatial residual and retain that exact map's trace."""
        mu: Any = equations.multiplier(unknown, size)
        trace: CompositionTrace = compose(maps, recipe, t, np.concatenate((value + mu, value - mu)), h)
        residual: Any = equations.residual(unknown, trace.state, size)
        return residual, trace

    def newton_update(unknown: np.ndarray, residual: np.ndarray, trace: CompositionTrace) -> np.ndarray:
        """Solve independent analytic particle blocks, or the dense FD system."""
        if jacobian_method == 'analytic':
            base: np.ndarray = particle_jacobians(maps, trace)
            count = maps.particle_count
            matrix: Any = equations.particle_jacobian(base)
            rhs = residual.reshape(-1, count).T
            correction: np.ndarray = _solve_particle_systems(
                matrix, rhs, singular_message=f'The projection Jacobian is singular for {context}.',
            ).T.reshape(-1)
        elif jacobian_method == 'finite_difference':
            base_dense: np.ndarray = central_difference_jacobian(
                lambda candidate: compose(maps, recipe, t, candidate, h).state,
                trace.state_before, relative_step=jacobian_relative_step,
            )
            matrix_dense: Any = equations.dense_jacobian(base_dense, size)
            try:
                correction = np.linalg.solve(matrix_dense, residual)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(f'The projection Jacobian is singular for {context}.') from exc
        else:
            raise ValueError('Unknown projection Jacobian method.')
        return np.asarray(unknown - correction)

    def initial_map(candidate: np.ndarray) -> tuple[np.ndarray, CompositionTrace]:
        trace: CompositionTrace = compose(maps, recipe, t, candidate, h)
        return trace.state, trace

    initial: Any
    cached: Any
    initial, cached = equations.prepare(value, initial_map)
    if isinstance(selected_solver, BroydenSolver):
        result: _NonlinearResult[Any] = selected_solver.solve(evaluate, initial, equations.initial_jacobian(value),
            scale=value, context=context, initial_evaluation=cached)
    elif isinstance(selected_solver, NewtonSolver):
        result = selected_solver.solve(evaluate, initial, newton_update,
            scale=value,
            context=context, initial_evaluation=cached,
            iteration_observer=_bind_newton_observer(
                newton_observer, time=t, duration=h, tolerance=tolerance,
                multiplier_slice=equations.multiplier_slice(size),
            ))
    else:
        raise ValueError('Unknown nonlinear solver for spatial projection.')
    mu: Any = equations.multiplier(result.unknown, size)
    physical: Any = equations.reconstruct(result.unknown, result.payload.state, size)
    return ProjectedMapResult(t, h, value.copy(), physical, mu.copy(),
        SolveStats(options.solver, result.iterations, result.residual_evaluations,
            float(np.linalg.norm(result.residual, ord=np.inf)), tolerance), result.payload,
        retain_energy_points)
