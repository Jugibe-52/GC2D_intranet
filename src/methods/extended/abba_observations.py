"""On-demand adapters from common ABBA records to existing observer events."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import numpy as np
from dynamics import DynamicalSystem

from contracts.observation import (
	ABBA2ImplicitIntegrationStep,
	ABBA4ImplicitIntegrationStep, ABBA6ImplicitIntegrationStep,
	IntegrationStep, UnprojectedABBAIntegrationStep,
)
from methods.extended.configuration import ProjectionFormulation
from methods.extended.records import (
	PhysicalProjectionTrace, ProjectedMapResult, StepResult,
)
from methods.extended.records import ProjectedMap, StepSolver


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


def bind_event_builder(
	dynamics: DynamicalSystem,
	method_name: str,
	formulation: ProjectionFormulation,
	*,
	order: Literal[2, 4, 6],
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
		if order == 2:
			return physical_single(projections[0], step_index)

		def map_state(candidate: np.ndarray) -> np.ndarray:
			return solve_step(t, candidate, h)[-1].state

		fields: dict[str, Any] = dict(
			dynamics_name=type(dynamics).__name__, method_name=method_name,
			step_index=step_index, start_time=t, time=t + h, duration=h,
			state_before=state_before.copy(),
			state_after=projections[-1].state.copy(),
			map_state=map_state, dynamics=dynamics, formulation_name=formulation,
			**_solve_fields(projections),
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
			return ABBA4ImplicitIntegrationStep(
				**fields, multiplier=projections[0].multiplier.copy(),
				composition_coefficients=np.asarray(coefficients), substeps=substeps,
			)
		return ABBA6ImplicitIntegrationStep(
			**fields, composition_coefficients=np.asarray(coefficients),
			substeps=tuple(physical_single(p, step_index) for p in projections),
		)

	return build


__all__: list[str] = []
