"""Resolve model choices once into the shared implicit ABBA execution plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Literal

import numpy as np

from dynamics import GuidingCenterDynamics, GuidingCenterJacobianSystem

from ..._result import DiagnosticValue
from ...observation import StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from .._nonlinear import SolverOptions
from ._coefficients import _ABBA4_COEFFICIENTS, _ABBA6_COEFFICIENTS
from ._configuration import _state_dimension_diagnostics
from ._energy import _validate_energy_tracking
from ._implicit import _ABBAImplicitConfig
from .maps.extended import _extended_vector_field_jacobian
from .maps.physical import _checked_vector_field_jacobian
from .observations import EventBuilder, bind_event_builder
from .state import StatePolicy, extended_state_policy, physical_state_policy
from .steps import (
	StepSolver, bind_extended_projection, bind_physical_projection,
	solve_outer_projection_step, solve_projected_composition_step, solve_single_map_step,
)


@dataclass(frozen=True, slots=True)
class PreparedABBA:
	"""Validated inputs and bound operations consumed by the shared runtime."""

	initial_workspace: np.ndarray
	solve_step: StepSolver
	state_ops: StatePolicy
	build_event: EventBuilder | None
	solver_options: SolverOptions
	method_metadata: Mapping[str, DiagnosticValue]
	include_substep_metrics: bool
	step_observer: StepObserver | None
	progress: bool
	method_name: str


def prepare_abba(
	problem: InitialValueProblem,
	method: _ABBAImplicitConfig,
	request: SimulationRequest,
	*,
	order: Literal[2, 4, 6],
	projection_placement: str = "after_each_abba_map",
) -> PreparedABBA:
	"""Bind state, step recipe, projection equation and event adaptation."""
	if projection_placement not in ("after_each_abba_map", "around_complete_composition"):
		raise ValueError("Unknown ABBA projection placement.")
	outer = projection_placement == "around_complete_composition"
	if outer and order != 4:
		raise ValueError("Exterior composition projection requires ABBA4.")
	coefficients = (
		(1.0,) if order == 2 else tuple(float(c) for c in (
			_ABBA4_COEFFICIENTS if order == 4 else _ABBA6_COEFFICIENTS
		))
	)
	dynamics = problem.dynamics
	name = type(method).__name__
	fully_extended = method.state_extension == "fully_extended"
	options = SolverOptions(
		method.nonlinear_solver, method.newton_absolute_tolerance,
		method.newton_relative_tolerance, method.newton_max_iterations,
	)
	z0 = problem.initial_state
	if fully_extended:
		if not isinstance(dynamics, GuidingCenterDynamics):
			raise TypeError(f"{name} requires GuidingCenterDynamics.")
		if z0.shape != (2,):
			raise ValueError(f"{name} requires exactly one GC particle.")
		state_ops = extended_state_policy(dynamics)
		project = bind_extended_projection(
			dynamics, options, method.projection_formulation,
			outer=outer, method_name=name,
		)
	else:
		if not isinstance(dynamics, GuidingCenterJacobianSystem):
			raise TypeError(f"{name} requires GuidingCenterJacobianSystem.")
		if dynamics.state_dimension != 2:
			raise TypeError(f"{name} requires planar two-component dynamics.")
		_validate_energy_tracking(dynamics, enabled=method.track_energy, method_name=name)
		state_ops = physical_state_policy(
			dynamics, physical_size=z0.size,
			particle_count=z0.size // 2, track_energy=method.track_energy,
		)
		project = bind_physical_projection(
			dynamics, options, method.projection_formulation, outer=outer,
		)
	initial = state_ops.initialize(z0, request.t_span[0])
	# Validate Newton capabilities before a zero residual can mask their absence.
	if options.solver == "newton":
		if fully_extended:
			assert isinstance(dynamics, GuidingCenterDynamics)
			_extended_vector_field_jacobian(dynamics, initial)
		else:
			assert isinstance(dynamics, GuidingCenterJacobianSystem)
			_checked_vector_field_jacobian(dynamics, request.t_span[0], z0)
	if outer:
		solve_step: StepSolver = partial(solve_outer_projection_step, project)
	elif order == 2:
		solve_step = partial(solve_single_map_step, project)
	else:
		solve_step = partial(solve_projected_composition_step, project, coefficients)
	builder = None
	if method.step_observer is not None:
		builder = bind_event_builder(
			dynamics, name, method.projection_formulation, order=order,
			fully_extended=fully_extended, outer=outer, coefficients=coefficients,
			solve_step=solve_step, project=project,
		)
	projection_count = 1 if outer else len(coefficients)
	metadata: dict[str, DiagnosticValue] = {
		"nonlinear_solves_per_step": projection_count,
		"nonlinear_solver": options.solver,
		"nonlinear_absolute_tolerance": options.absolute_tolerance,
		"nonlinear_relative_tolerance": options.relative_tolerance,
		"nonlinear_max_iterations": options.max_iterations,
		"projection_formulation": method.projection_formulation,
		"state_extension": method.state_extension,
		"track_energy": method.track_energy,
	}
	metadata.update(_state_dimension_diagnostics(
		method.state_extension, method.projection_formulation, particle_count=z0.size // 2,
	))
	include_substeps = fully_extended or order != 2
	if include_substeps:
		coefficient_array = np.asarray(coefficients)
		coefficient_array.setflags(write=False)
		metadata.update({
			"implicit_substeps_per_step": projection_count,
			"projection_placement": projection_placement,
			"composition_coefficients": coefficient_array,
		})
	if fully_extended:
		assert isinstance(dynamics, GuidingCenterDynamics)
		metadata.update({
			"unprojected_abba_maps_per_step": len(coefficients),
			"unprojected_abba_maps_per_residual_evaluation": len(coefficients) if outer else 1,
			"projection_jacobian": (
				"analytic_stage_product" if options.solver == "newton"
				or (method.step_observer is not None and dynamics.effective_potential.interpolation_order >= 3)
				else "centered_difference_observer_fallback" if method.step_observer is not None
				else "not_evaluated"
			),
		})
	elif outer:
		metadata.update({
			"unprojected_abba_maps_per_step": len(coefficients),
			"unprojected_abba_maps_per_residual_evaluation": len(coefficients),
			"base_composition": "unprojected_abba4_triple_jump",
		})
	elif order != 2:
		metadata.update({
			"substep_projection_formulation": method.projection_formulation,
			"composition_policy": "project_each_abba_substep",
		})
	initial = initial.copy()
	initial.setflags(write=False)
	return PreparedABBA(
		initial, solve_step, state_ops, builder, options, MappingProxyType(metadata),
		include_substeps, method.step_observer, bool(method.progress), name,
	)


__all__: list[str] = []
