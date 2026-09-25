"""Compatibility step views for existing ABBA diagnostic callers."""

from __future__ import annotations

import numpy as np
from dynamics import GuidingCenterJacobianSystem
from methods._nonlinear import SolverOptions
from methods.extended.configuration import ProjectionFormulation
from methods.extended.composition import ABBA2, ABBA4
from methods.extended.projection import solve_projection
from methods.extended.abba_maps import uncoupled_maps
from methods.extended.records import ProjectedMap, StepSolver, ProjectedMapResult


def solve_physical_projection(
	dynamics: GuidingCenterJacobianSystem, options: SolverOptions,
	formulation: ProjectionFormulation, t: float, state: np.ndarray, h: float,
	*, outer: bool,
) -> ProjectedMapResult:
    """Compatibility binding for diagnostic callers; runtime binds once."""
    return solve_projection(uncoupled_maps(dynamics, state), ABBA4 if outer else ABBA2,
                            options, formulation, t, state, h)


def bind_physical_projection(
	dynamics: GuidingCenterJacobianSystem, options: SolverOptions,
	formulation: ProjectionFormulation, *, outer: bool,
) -> ProjectedMap:
	"""Compatibility adapter for existing diagnostic composition helpers."""
	from functools import partial
	return partial(solve_physical_projection, dynamics, options, formulation, outer=outer)


def solve_single_map_step(
	project: ProjectedMap, t: float, state: np.ndarray, h: float
) -> tuple[ProjectedMapResult, ...]:
	"""ABBA2: one complete projected map, with one accepted projection."""
	return (project(t, state, h),)


def solve_projected_composition_step(
	project: ProjectedMap,
	coefficients: tuple[float, ...],
	t: float,
	state: np.ndarray,
	h: float,
) -> tuple[ProjectedMapResult, ...]:
	"""Compose accepted projected maps in signed coefficient order."""
	time = float(t)
	current = state
	accepted = []
	for coefficient in coefficients:
		duration = float(coefficient * h)
		result = project(time, current, duration)
		accepted.append(result)
		current = result.state
		time += duration
	expected = float(t + h)
	tolerance = float(64.0 * np.finfo(float).eps * max(1.0, abs(t), abs(expected), abs(h)))
	if not np.isclose(time, expected, rtol=0.0, atol=tolerance):
		raise RuntimeError("The ABBA composition coefficients do not sum to one.")
	return tuple(accepted)


def solve_outer_projection_step(
	project: ProjectedMap, t: float, state: np.ndarray, h: float
) -> tuple[ProjectedMapResult, ...]:
	"""ABBA4 exterior placement: one projection around its continuous maps."""
	return (project(t, state, h),)


__all__: list[str] = []
