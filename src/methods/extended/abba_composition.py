"""Compatibility step views built from the shared projected-composition recipe.

Existing diagnostic helpers use these private state/substeps views. Integration
itself consumes the common ProjectedMapResult tuple directly.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from dynamics import GuidingCenterJacobianSystem
from methods._nonlinear import NonlinearSolver, SolverOptions
from methods.extended.coefficients import _ABBA6_COEFFICIENTS
from methods.extended.configuration import ProjectionFormulation
from methods.extended.abba_maps import _ProjectedStep
from methods.extended.records import PhysicalProjectionTrace
from methods.extended.abba_steps import bind_physical_projection, solve_projected_composition_step

@dataclass(frozen=True, slots=True)
class _AcceptedSubstep:
	"""One signed projected-ABBA solve inside an accepted outer step."""

	start_time: float
	duration: float
	state_before: np.ndarray
	result: _ProjectedStep


@dataclass(frozen=True, slots=True)
class _ComposedABBAStep:
	"""Final physical state and the accepted nonlinear composition substeps."""

	state: np.ndarray
	substeps: tuple[_AcceptedSubstep, ...]


def _solve_composed_abba_step(
    dynamics: GuidingCenterJacobianSystem, t: float, state: np.ndarray, step: float, *,
    coefficients: np.ndarray, method_name: str, absolute_tolerance: float,
    relative_tolerance: float, max_iterations: int, nonlinear_solver: NonlinearSolver,
    projection_formulation: ProjectionFormulation = "reduced_multiplier",
) -> _ComposedABBAStep:
    """Adapt common accepted records to the existing diagnostic step view."""
    value = np.asarray(state, dtype=float)
    if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError(f"The {method_name} physical state must be a finite, non-empty vector.")
    composition = np.asarray(coefficients, dtype=float)
    if composition.ndim != 1 or composition.size == 0 or not np.all(np.isfinite(composition)):
        raise ValueError("ABBA composition coefficients must be finite and non-empty.")
    project = bind_physical_projection(
        dynamics, SolverOptions(nonlinear_solver, absolute_tolerance, relative_tolerance, max_iterations),
        projection_formulation, outer=False,
    )
    projections = solve_projected_composition_step(project, tuple(composition), t, value, step)
    accepted = []
    for p in projections:
        assert isinstance(p.trace, PhysicalProjectionTrace)
        kernel_result = _ProjectedStep(
            p.state, p.multiplier, p.trace.maps[0].stages,
            p.stats.iterations, p.stats.residual_evaluations, p.stats.residual_norm,
        )
        accepted.append(_AcceptedSubstep(p.start_time, p.duration, p.state_before, kernel_result))
    return _ComposedABBAStep(projections[-1].state, tuple(accepted))


def _solve_abba6_step(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver,
	projection_formulation: ProjectionFormulation = "reduced_multiplier",
) -> _ComposedABBAStep:
	"""Compose the seven signed projected ABBA maps of ABBA6Implicit."""
	return _solve_composed_abba_step(
		dynamics,
		t,
		state,
		step,
		coefficients=_ABBA6_COEFFICIENTS,
		method_name="ABBA6Implicit",
		absolute_tolerance=absolute_tolerance,
		relative_tolerance=relative_tolerance,
		max_iterations=max_iterations,
		nonlinear_solver=nonlinear_solver,
		projection_formulation=projection_formulation,
	)


__all__: list[str] = []
