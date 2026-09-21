"""One integration coordinator for the prepared implicit ABBA family."""

from __future__ import annotations

import numpy as np

from ..._fixed import integrate_fixed_grid
from ..._result import IntegrationData
from ...request import SimulationRequest
from .observations import emit_observation
from .preparation import PreparedABBA
from .records import StepMetrics, StepResult, record_completed_step


def integrate_abba(prepared: PreparedABBA, request: SimulationRequest) -> IntegrationData:
	"""Advance the selected map, record main steps and extract physical output."""
	metrics = StepMetrics()
	state_ops = prepared.state_ops

	def advance(
		t: float, workspace: np.ndarray, h: float, step_index: int, observe: bool
	) -> np.ndarray:
		"""Use identical numerical work for main steps and shadow samples."""
		state_before = state_ops.unpack(t, workspace)
		projections = prepared.solve_step(t, state_before, h)
		next_workspace = state_ops.finish_step(t, h, workspace, projections)
		if observe:
			result = StepResult(next_workspace, projections)
			record_completed_step(metrics, result)
			emit_observation(
				prepared.build_event, prepared.step_observer,
				t, h, step_index, state_before, result,
			)
		return next_workspace

	history, step_count = integrate_fixed_grid(
		prepared.initial_workspace, request, advance,
		progress=prepared.progress, label=prepared.method_name,
	)
	states, energy = state_ops.extract(request.output_times, history)
	diagnostics = dict(prepared.method_metadata)
	diagnostics["step_count"] = step_count
	diagnostics.update(metrics.finalize(include_substeps=prepared.include_substep_metrics))
	diagnostics.update(energy)
	# Legacy names are output aliases only; the numerical records use solver-neutral names.
	for legacy, canonical in (
		("newton_iterations", "nonlinear_iterations"),
		("newton_residual_norms", "nonlinear_residual_norms"),
		("newton_absolute_tolerance", "nonlinear_absolute_tolerance"),
		("newton_relative_tolerance", "nonlinear_relative_tolerance"),
		("newton_max_iterations", "nonlinear_max_iterations"),
	):
		diagnostics[legacy] = diagnostics[canonical]
	return IntegrationData(t=request.output_times, states=states, diagnostics=diagnostics)


__all__: list[str] = []
