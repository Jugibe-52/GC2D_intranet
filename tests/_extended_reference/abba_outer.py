"""One physical projection around three continuous signed A-B-B-A maps."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from methods.extended.observations import abba_pairs
from methods.extended.core.projection import solve_projection
from methods.extended.core.composition import ABBA4
from tests._extended_reference.abba_maps import uncoupled_maps
from methods._nonlinear import SolverOptions
from dynamics import GuidingCenterJacobianSystem
from methods._nonlinear import NonlinearSolver, _solve_broyden, _solve_newton
from methods.extended.core.composition import _ABBA4_COEFFICIENTS
from methods.extended.configuration import ProjectionFormulation
from tests._extended_reference.abba_maps import _ABBAStages, _evaluate_unprojected_stages, _differentiate_stages

@dataclass(frozen=True, slots=True)
class _SingleProjectionBaseEvaluation:
	"""One complete unprojected ABBA4 map at a candidate multiplier."""

	u_final: np.ndarray
	v_final: np.ndarray
	residual: np.ndarray
	substeps: tuple[_ABBAStages, ...]


@dataclass(frozen=True, slots=True)
class _SingleProjectionResidualEvaluation:
	"""ABBA4 residual and its exact independent-particle Newton blocks."""

	u_final: np.ndarray
	v_final: np.ndarray
	residual: np.ndarray
	jacobian: np.ndarray
	base_jacobian: np.ndarray
	substeps: tuple[_ABBAStages, ...]


@dataclass(frozen=True, slots=True)
class _ABBA4SingleProjectionStep:
	"""Accepted physical state and one outer projection solve's diagnostics."""

	state: np.ndarray
	multiplier: np.ndarray
	substeps: tuple[_ABBAStages, ...]
	iterations: int
	residual_evaluations: int
	residual_norm: float


def _validated_state(
	dynamics: GuidingCenterJacobianSystem,
	state: np.ndarray,
) -> np.ndarray:
	"""Return one finite packed planar GC state."""
	value = np.asarray(state, dtype=float)
	if (
		value.ndim != 1
		or value.size == 0
		or value.size % dynamics.state_dimension
		or not np.all(np.isfinite(value))
	):
		raise ValueError(
			"The ABBA4 single-projection state must be a finite, non-empty "
			"packed GC vector."
		)
	return value


def _evaluate_single_projection_base(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	multiplier: np.ndarray,
) -> _SingleProjectionBaseEvaluation:
	"""Apply three continuous signed ABBA maps with no intermediate projection."""
	value = _validated_state(dynamics, state)
	mu = np.asarray(multiplier, dtype=float)
	if mu.shape != value.shape or not np.all(np.isfinite(mu)):
		raise ValueError(
			"The ABBA4 single-projection multiplier must match the physical state."
		)
	if not np.isfinite(t) or not np.isfinite(step):
		raise ValueError("The ABBA4 single-projection time and step must be finite.")

	# Hairer's normal embedding N=G^T displaces the two physical copies once,
	# before the whole fourth-order base composition.
	u_current = value + mu
	v_current = value - mu
	current_time = float(t)
	substeps: list[_ABBAStages] = []
	for coefficient in _ABBA4_COEFFICIENTS:
		duration = float(coefficient * step)
		stages = _evaluate_unprojected_stages(
			dynamics,
			current_time,
			u_current,
			v_current,
			duration,
		)
		substeps.append(stages)
		u_current = stages.u_final
		v_current = stages.v_final
		current_time += duration

	expected_time = float(t + step)
	time_tolerance = float(
		64.0
		* np.finfo(float).eps
		* max(1.0, abs(float(t)), abs(expected_time), abs(float(step)))
	)
	if not np.isclose(
		current_time,
		expected_time,
		rtol=0.0,
		atol=time_tolerance,
	):
		raise RuntimeError("The ABBA4 composition coefficients do not sum to one.")

	# The final normal correction is (+mu, -mu), so diagonal equality is
	# u_final - v_final + 2 mu = 0.
	residual = u_current - v_current + 2.0 * mu
	return _SingleProjectionBaseEvaluation(
		u_final=np.asarray(u_current),
		v_final=np.asarray(v_current),
		residual=np.asarray(residual),
		substeps=tuple(substeps),
	)


def _differentiate_single_projection_base(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	base: _SingleProjectionBaseEvaluation,
) -> _SingleProjectionResidualEvaluation:
	"""Form ``K = G M_3 M_2 M_1 N + 2 I`` from exact ABBA tangents."""
	value = _validated_state(dynamics, state)
	particle_count = value.size // dynamics.state_dimension
	identity_2 = np.broadcast_to(
		np.eye(dynamics.state_dimension),
		(particle_count, dynamics.state_dimension, dynamics.state_dimension),
	)
	base_jacobian = np.broadcast_to(
		np.eye(2 * dynamics.state_dimension),
		(
			particle_count,
			2 * dynamics.state_dimension,
			2 * dynamics.state_dimension,
		),
	).copy()
	current_time = float(t)
	for coefficient, stages in zip(
		_ABBA4_COEFFICIENTS,
		base.substeps,
		strict=True,
	):
		duration = float(coefficient * step)
		factor = _differentiate_stages(
			dynamics,
			current_time,
			value,
			duration,
			stages,
		).abba_jacobian
		base_jacobian = factor @ base_jacobian
		current_time += duration

	dimension = dynamics.state_dimension
	top_left = base_jacobian[:, :dimension, :dimension]
	top_right = base_jacobian[:, :dimension, dimension:]
	bottom_left = base_jacobian[:, dimension:, :dimension]
	bottom_right = base_jacobian[:, dimension:, dimension:]
	jacobian = (
		top_left
		- top_right
		- bottom_left
		+ bottom_right
		+ 2.0 * identity_2
	)
	if not np.all(np.isfinite(jacobian)):
		raise ValueError(
			"The ABBA4 single-projection residual Jacobian is non-finite."
		)
	return _SingleProjectionResidualEvaluation(
		u_final=base.u_final,
		v_final=base.v_final,
		residual=base.residual,
		jacobian=np.asarray(jacobian),
		base_jacobian=np.asarray(base_jacobian),
		substeps=base.substeps,
	)


def _evaluate_single_projection_residual(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	multiplier: np.ndarray,
) -> _SingleProjectionResidualEvaluation:
	"""Evaluate the reduced outer residual and its exact analytic Jacobian."""
	base = _evaluate_single_projection_base(
		dynamics,
		t,
		state,
		step,
		multiplier,
	)
	return _differentiate_single_projection_base(
		dynamics,
		t,
		state,
		step,
		base,
	)


def _projected_state(
	base: _SingleProjectionBaseEvaluation,
	multiplier: np.ndarray,
) -> np.ndarray:
	"""Return the neutral mean of the two symmetrically corrected copies."""
	first_copy = base.u_final + multiplier
	second_copy = base.v_final - multiplier
	return np.asarray((first_copy + second_copy) / 2.0)


def _solve_reduced_abba4_single_projection_step(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver = "newton",
) -> _ABBA4SingleProjectionStep:
    """Adapt the common outer projection to the existing ABBA4 diagnostic view."""
    result = solve_projection(uncoupled_maps(dynamics, state), ABBA4,
        SolverOptions(nonlinear_solver, absolute_tolerance, relative_tolerance, max_iterations),
        'reduced_multiplier', t, state, step)
    return _ABBA4SingleProjectionStep(result.state, result.multiplier,
        tuple(m.stages for m in abba_pairs(result.trace)), result.iterations,
        result.residual_evaluations, result.residual_norm)


def _simultaneous_residual_blocks(
	base: _SingleProjectionBaseEvaluation,
	multiplier: np.ndarray,
	first_output: np.ndarray,
	second_output: np.ndarray,
	state_dimension: int,
) -> np.ndarray:
	"""Return particle-major output, multiplier, and diagonal defects."""
	first_defect = first_output - multiplier - base.u_final
	second_defect = second_output + multiplier - base.v_final
	constraint = first_output - second_output
	particle_blocks = lambda vector: vector.reshape(state_dimension, -1).T
	return np.concatenate(
		(
			particle_blocks(first_defect),
			particle_blocks(second_defect),
			particle_blocks(constraint),
		),
		axis=-1,
	)


def _simultaneous_newton_jacobian(
	evaluation: _SingleProjectionResidualEvaluation,
) -> np.ndarray:
	"""Assemble exact simultaneous blocks around the complete ABBA4 base map."""
	particle_count = evaluation.base_jacobian.shape[0]
	identity_2 = np.broadcast_to(np.eye(2), (particle_count, 2, 2))
	identity_4 = np.broadcast_to(np.eye(4), (particle_count, 4, 4))
	zero_2 = np.zeros((particle_count, 2, 2), dtype=float)
	normal = np.concatenate((identity_2, -identity_2), axis=-2)
	constraint = np.concatenate((identity_2, -identity_2), axis=-1)
	top_right = -(identity_4 + evaluation.base_jacobian) @ normal
	return np.concatenate(
		(
			np.concatenate((identity_4, top_right), axis=-1),
			np.concatenate((constraint, zero_2), axis=-1),
		),
		axis=-2,
	)


def _solve_simultaneous_abba4_single_projection_step(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver = "newton",
) -> _ABBA4SingleProjectionStep:
    """Adapt the common outer projection to the existing ABBA4 diagnostic view."""
    result = solve_projection(uncoupled_maps(dynamics, state), ABBA4,
        SolverOptions(nonlinear_solver, absolute_tolerance, relative_tolerance, max_iterations),
        'simultaneous_state_multiplier', t, state, step)
    return _ABBA4SingleProjectionStep(result.state, result.multiplier,
        tuple(m.stages for m in abba_pairs(result.trace)), result.iterations,
        result.residual_evaluations, result.residual_norm)


def _solve_abba4_single_projection_step(
	dynamics: GuidingCenterJacobianSystem,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	nonlinear_solver: NonlinearSolver = "newton",
	projection_formulation: ProjectionFormulation = "reduced_multiplier",
) -> _ABBA4SingleProjectionStep:
	"""Select one formulation around the complete unprojected ABBA4 map."""
	solver = (
		_solve_reduced_abba4_single_projection_step
		if projection_formulation == "reduced_multiplier"
		else _solve_simultaneous_abba4_single_projection_step
	)
	return solver(
		dynamics,
		t,
		state,
		step,
		absolute_tolerance=absolute_tolerance,
		relative_tolerance=relative_tolerance,
		max_iterations=max_iterations,
		nonlinear_solver=nonlinear_solver,
	)


__all__: list[str] = []
