"""On-demand adapters from common ABBA records to existing observer events."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import numpy as np
from dynamics import DynamicalSystem

from ...observation import (
	ABBA2ImplicitIntegrationStep,
	ABBA4ImplicitSingleProjectionIntegrationStep, ABBA6ImplicitIntegrationStep,
	FullyExtendedBaseMap, FullyExtendedImplicitIntegrationStep,
	IntegrationStep, StepObserver, UnprojectedABBAIntegrationStep,
)
from ._configuration import ProjectionFormulation
from .maps.extended import (
	_ANTIDIAGONAL_EMBEDDING, _COPY_DIFFERENCE, _IDENTITY_4,
	_checked_extended_state, _synchronized_extended_time,
)
from .projection_extended import _base_map_jacobian, _projected_substep_jacobian
from .records import (
	ExtendedProjectionTrace, PhysicalProjectionTrace, ProjectedMapResult, StepResult,
)
from .steps import ProjectedMap, StepSolver


EventBuilder: TypeAlias = Callable[
	[float, float, int, np.ndarray, StepResult], IntegrationStep
]


def _solve_fields(projections: tuple[ProjectedMapResult, ...]) -> dict[str, Any]:
	"""Read one consistent aggregate from the accepted numerical records."""
	worst = max(projections, key=lambda p: p.stats.residual_norm / p.stats.tolerance)
	return {
		"nonlinear_solver": worst.stats.solver,
		"newton_iterations": sum(p.stats.iterations for p in projections),
		"residual_evaluations": sum(p.stats.residual_evaluations for p in projections),
		"newton_residual_norm": worst.stats.residual_norm,
		"newton_tolerance": worst.stats.tolerance,
		"projection_multiplier_norm": max(
			float(np.linalg.norm(p.multiplier, ord=np.inf)) for p in projections
		),
	}


def _extended_base_event(
	projection: ProjectedMapResult, *, map_name: str
) -> FullyExtendedBaseMap:
	"""Copy one accepted full base map and lazily form its diagnostic tangent."""
	trace = projection.trace
	assert isinstance(trace, ExtendedProjectionTrace)
	result = trace.result

	def jacobian_state(state: np.ndarray) -> np.ndarray:
		return _base_map_jacobian(result.base_map, state)

	residual_jacobian = result.residual_jacobian
	if residual_jacobian is None:
		residual_jacobian = np.asarray(
			_COPY_DIFFERENCE @ jacobian_state(result.internal_input)
			@ _ANTIDIAGONAL_EMBEDDING + 2.0 * _IDENTITY_4
		)
	return FullyExtendedBaseMap(
		map_name=map_name, start_time=projection.start_time, duration=projection.duration,
		state_before=result.internal_input.copy(), state_after=result.mapped.copy(),
		map_state=result.base_map.map_state, jacobian_state=jacobian_state,
		projection_multiplier=result.multiplier.copy(),
		residual_jacobian=residual_jacobian.copy(),
	)


def bind_event_builder(
	dynamics: DynamicalSystem,
	method_name: str,
	formulation: ProjectionFormulation,
	*,
	order: Literal[2, 4, 6],
	fully_extended: bool,
	outer: bool,
	coefficients: tuple[float, ...],
	solve_step: StepSolver,
	project: ProjectedMap,
) -> EventBuilder:
	"""Choose the public event adapter once, outside the integration loop."""
	if order == 4 and not outer:
		raise ValueError("ABBA4 observation requires the outer-projection recipe.")
	def physical_single(
		projection: ProjectedMapResult, step_index: int
	) -> ABBA2ImplicitIntegrationStep:
		trace = projection.trace
		assert isinstance(trace, PhysicalProjectionTrace) and len(trace.maps) == 1
		stages = trace.maps[0].stages

		def map_state(candidate: np.ndarray) -> np.ndarray:
			return project(projection.start_time, candidate, projection.duration).state

		return ABBA2ImplicitIntegrationStep(
			dynamics_name=type(dynamics).__name__, method_name=method_name,
			step_index=step_index, start_time=projection.start_time,
			time=projection.start_time + projection.duration, duration=projection.duration,
			state_before=projection.state_before.copy(), state_after=projection.state.copy(),
			map_state=map_state, formulation_name=formulation, dynamics=dynamics,
			multiplier=projection.multiplier.copy(),
			u_initial=stages.u_initial.copy(), v_initial=stages.v_initial.copy(),
			u_first=stages.u_first.copy(), v_final=stages.v_final.copy(),
			u_final=stages.u_final.copy(), **_solve_fields((projection,)),
		)

	def build(
		t: float, h: float, step_index: int, state_before: np.ndarray, result: StepResult
	) -> IntegrationStep:
		projections = result.projections
		if not fully_extended and order == 2:
			return physical_single(projections[0], step_index)

		def map_state(candidate: np.ndarray) -> np.ndarray:
			if fully_extended:
				value = _checked_extended_state(candidate, duplicated=False)
				mapped = solve_step(float(value[2]), value, h)[-1].state
				return _synchronized_extended_time(
					mapped, float(value[2] + h), context="The fully extended ABBA map",
				)
			return solve_step(t, candidate, h)[-1].state

		fields: dict[str, Any] = dict(
			dynamics_name=type(dynamics).__name__, method_name=method_name,
			step_index=step_index, start_time=t, time=t + h, duration=h,
			state_before=state_before.copy(),
			state_after=(result.next_workspace if fully_extended else projections[-1].state).copy(),
			map_state=map_state, dynamics=dynamics, formulation_name=formulation,
			**_solve_fields(projections),
		)
		if fully_extended:
			map_name = (
				"fully_extended_abba4_unprojected_composition" if outer
				else "fully_extended_abba_map"
			)
			base_maps = tuple(_extended_base_event(p, map_name=map_name) for p in projections)
			jacobian = _IDENTITY_4.copy()
			for projection in projections:
				assert isinstance(projection.trace, ExtendedProjectionTrace)
				jacobian = _projected_substep_jacobian(projection.trace.result) @ jacobian
			return FullyExtendedImplicitIntegrationStep(
				**fields, multiplier=projections[-1].multiplier.copy(),
				jacobian=jacobian, base_maps=base_maps,
			)
		if outer:
			trace = projections[0].trace
			assert isinstance(trace, PhysicalProjectionTrace)
			substeps = tuple(
				UnprojectedABBAIntegrationStep(
					start_time=m.start_time, time=m.start_time + m.duration,
					duration=m.duration, u_initial=m.stages.u_initial.copy(),
					v_initial=m.stages.v_initial.copy(), u_first=m.stages.u_first.copy(),
					v_final=m.stages.v_final.copy(), u_final=m.stages.u_final.copy(),
				)
				for m in trace.maps
			)
			return ABBA4ImplicitSingleProjectionIntegrationStep(
				**fields, multiplier=projections[0].multiplier.copy(),
				composition_coefficients=np.asarray(coefficients), substeps=substeps,
			)
		return ABBA6ImplicitIntegrationStep(
			**fields, composition_coefficients=np.asarray(coefficients),
			substeps=tuple(physical_single(p, step_index) for p in projections),
		)

	return build


def emit_observation(
	builder: EventBuilder | None,
	observer: StepObserver | None,
	t: float,
	h: float,
	step_index: int,
	state_before: np.ndarray,
	result: StepResult,
) -> None:
	"""Construct and publish an event only for a configured main-step observer."""
	if builder is not None and observer is not None:
		observer(builder(t, h, step_index, state_before, result))


__all__: list[str] = []
