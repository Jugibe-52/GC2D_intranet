"""Explicit numerical step recipes bound to the common ABBA runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np

from dynamics import GuidingCenterJacobianSystem

from .._nonlinear import SolveStats, SolverOptions
from ._coefficients import _ABBA4_COEFFICIENTS
from ._configuration import ProjectionFormulation
from ._projection_common import _ProjectedStep
from .projection_outer import (
	_ABBA4SingleProjectionStep,
	_solve_reduced_abba4_single_projection_step,
	_solve_simultaneous_abba4_single_projection_step,
)
from .projection_reduced import _solve_reduced_multiplier_step
from .projection_simultaneous import _solve_simultaneous_state_multiplier_step
from .records import (
	PhysicalBaseMapTrace, PhysicalProjectionTrace,
	ProjectedMapResult,
)


ProjectedMap: TypeAlias = Callable[[float, np.ndarray, float], ProjectedMapResult]
StepSolver: TypeAlias = Callable[
	[float, np.ndarray, float], tuple[ProjectedMapResult, ...]
]


def solve_physical_projection(
	dynamics: GuidingCenterJacobianSystem, options: SolverOptions,
	formulation: ProjectionFormulation, t: float, state: np.ndarray, h: float,
	*, outer: bool,
) -> ProjectedMapResult:
	"""Solve the spatial equation and retain its accepted map traces."""
	single_solver = (
		_solve_reduced_multiplier_step if formulation == "reduced_multiplier"
		else _solve_simultaneous_state_multiplier_step
	)
	outer_solver = (
		_solve_reduced_abba4_single_projection_step
		if formulation == "reduced_multiplier"
		else _solve_simultaneous_abba4_single_projection_step
	)

	result: _ProjectedStep | _ABBA4SingleProjectionStep
	if outer:
		result = outer_solver(
			dynamics, t, state, h,
			absolute_tolerance=options.absolute_tolerance,
			relative_tolerance=options.relative_tolerance,
			max_iterations=options.max_iterations, nonlinear_solver=options.solver,
		)
		stages = result.substeps
		coefficients = _ABBA4_COEFFICIENTS
	else:
		single = single_solver(
			dynamics, t, state, h,
			absolute_tolerance=options.absolute_tolerance,
			relative_tolerance=options.relative_tolerance,
			max_iterations=options.max_iterations, nonlinear_solver=options.solver,
		)
		result = single
		stages = (single.stages,)
		coefficients = np.asarray((1.0,))
	time = float(t)
	maps: list[PhysicalBaseMapTrace] = []
	for coefficient, stage in zip(coefficients, stages, strict=True):
		duration = float(coefficient * h)
		maps.append(PhysicalBaseMapTrace(time, duration, stage))
		time += duration
	return ProjectedMapResult(
		t, h, state.copy(), result.state, result.multiplier,
		SolveStats(options.solver, result.iterations, result.residual_evaluations,
			result.residual_norm, options.tolerance(state)),
		PhysicalProjectionTrace(tuple(maps)),
	)


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
