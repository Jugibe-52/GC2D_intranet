"""Shared spatial Hairer equations for palindromic extended-space maps."""
from __future__ import annotations

from typing import Literal
import numpy as np

from formulations.gc import GCDoubledMaps
from methods._nonlinear import SolverOptions, SolveStats, _solve_broyden, _solve_newton
from methods.extended.composition import Composition, compose
from methods.extended.configuration import ProjectionFormulation
from methods.extended.jacobians import central_difference_jacobian, particle_jacobians
from methods.extended.records import CompositionTrace, ProjectedMapResult


def solve_projection(
    maps: GCDoubledMaps, recipe: Composition, options: SolverOptions,
    formulation: ProjectionFormulation,
    t: float, state: np.ndarray, h: float, *,
    jacobian_method: Literal['analytic', 'finite_difference'] = 'analytic',
    jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps)),
    retain_energy_points: bool = False,
) -> ProjectedMapResult:
    """Solve the selected equation around one complete unprojected recipe.

    Reduced unknowns are the physical multiplier ``mu`` (2N components).
    Simultaneous unknowns are ``(u_out, v_out, mu)`` (6N components). Time and
    passive momentum never enter either root or its state-scaled tolerance.
    """
    value = np.asarray(state, dtype=float)
    if value.ndim != 1 or value.size != maps.physical_size or not np.all(np.isfinite(value)):
        raise ValueError("The projection requires a finite packed physical state.")
    size = value.size
    simultaneous = formulation == 'simultaneous_state_multiplier'
    if formulation not in ('reduced_multiplier', 'simultaneous_state_multiplier'):
        raise ValueError("Unknown projection formulation.")
    tolerance = options.tolerance(value)
    context = f'{recipe.name} {formulation} at t={t:.16g} with step={h:.16g}'

    def evaluate(unknown: np.ndarray) -> tuple[np.ndarray, CompositionTrace]:
        """Evaluate a spatial residual and retain that exact map's trace."""
        mu = unknown[2 * size:] if simultaneous else unknown
        trace = compose(maps, recipe, t, np.concatenate((value + mu, value - mu)), h)
        first, second = trace.state[:size], trace.state[size:]
        if simultaneous:
            u, v = unknown[:size], unknown[size:2 * size]
            residual = np.concatenate((u - mu - first, v + mu - second, u - v))
        else:
            residual = first - second + 2 * mu
        return residual, trace

    def newton_update(unknown: np.ndarray, residual: np.ndarray, trace: CompositionTrace) -> np.ndarray:
        """Solve independent analytic particle blocks, or the dense FD system."""
        if jacobian_method == 'analytic':
            base = particle_jacobians(maps, trace)
            count = maps.particle_count
            identity = np.broadcast_to(np.eye(2), (count, 2, 2))
            if simultaneous:
                normal = np.concatenate((identity, -identity), axis=1)
                constraint = np.concatenate((identity, -identity), axis=2)
                identity4 = np.broadcast_to(np.eye(4), (count, 4, 4))
                matrix = np.concatenate((
                    np.concatenate((identity4, -(identity4 + base) @ normal), axis=2),
                    np.concatenate((constraint, np.zeros((count, 2, 2))), axis=2),
                ), axis=1)
            else:
                matrix = base[:, :2, :2] - base[:, :2, 2:] - base[:, 2:, :2] + base[:, 2:, 2:] + 2 * identity
            rhs = residual.reshape(-1, count).T
            try:
                correction = np.linalg.solve(matrix, rhs[..., None])[..., 0].T.reshape(-1)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(f'The projection Jacobian is singular for {context}.') from exc
        elif jacobian_method == 'finite_difference':
            base_dense = central_difference_jacobian(
                lambda candidate: compose(maps, recipe, t, candidate, h).state,
                trace.state_before, relative_step=jacobian_relative_step,
            )
            identity_dense = np.eye(size)
            normal_dense = np.concatenate((identity_dense, -identity_dense), axis=0)
            constraint_dense = normal_dense.T
            if simultaneous:
                matrix_dense = np.block([
                    [np.eye(2 * size), -(np.eye(2 * size) + base_dense) @ normal_dense],
                    [constraint_dense, np.zeros((size, size))],
                ])
            else:
                matrix_dense = constraint_dense @ base_dense @ normal_dense + 2 * identity_dense
            try:
                correction = np.linalg.solve(matrix_dense, residual)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(f'The projection Jacobian is singular for {context}.') from exc
        else:
            raise ValueError('Unknown projection Jacobian method.')
        return np.asarray(unknown - correction)

    initial = np.zeros(size)
    cached = None
    if simultaneous:
        trace = compose(maps, recipe, t, np.concatenate((value, value)), h)
        initial = np.concatenate((trace.state, initial))
        cached = (np.concatenate((np.zeros(2 * size), trace.state[:size] - trace.state[size:])), trace)
    if options.solver == 'broyden':
        if simultaneous:
            identity = np.eye(size)
            normal = np.concatenate((identity, -identity), axis=0)
            initial_jacobian = np.block([
                [np.eye(2 * size), -2 * normal],
                [normal.T, np.zeros((size, size))],
            ])
        else:
            # D r(0) = G I N + 2I = 4I at zero step for every recipe.
            initial_jacobian = 4 * np.eye(size)
        result = _solve_broyden(evaluate, initial, initial_jacobian,
            tolerance=tolerance, max_iterations=options.max_iterations,
            context=context, initial_evaluation=cached)
    elif options.solver == 'newton':
        result = _solve_newton(evaluate, initial, newton_update,
            tolerance=tolerance, max_iterations=options.max_iterations,
            context=context, initial_evaluation=cached)
    else:
        raise ValueError('Unknown nonlinear solver for spatial projection.')
    if simultaneous:
        first, second = result.unknown[:size], result.unknown[size:2 * size]
        mu = result.unknown[2 * size:]
    else:
        mu = result.unknown
        first, second = result.payload.state[:size] + mu, result.payload.state[size:] - mu
    return ProjectedMapResult(t, h, value.copy(), (first + second) / 2, mu.copy(),
        SolveStats(options.solver, result.iterations, result.residual_evaluations,
            float(np.linalg.norm(result.residual, ord=np.inf)), tolerance), result.payload,
        retain_energy_points)
