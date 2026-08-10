"""Optional observations emitted by numerical stages and complete steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, TypeAlias

import numpy as np

from dynamics import GuidingCenterJacobianSystem


StateMap: TypeAlias = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class IntegrationStage:
	"""Describe one direct or adjoint map inside a composed integration step.

	``state_before`` and ``state_after`` are independent snapshots of the packed
	internal state. ``map_state`` evaluates this exact stage—with its duration and
	evaluation time already fixed—on another state of the same shape. Diagnostic
	code can therefore differentiate a stage without duplicating integrator logic.
	For a stage-projected method, every stage map includes that projection.
	"""

	dynamics_name: str
	formulation_name: str
	method_name: str
	flow_name: Literal["flow", "adjoint_flow"]
	step_index: int
	stage_index: int
	time: float
	duration: float
	state_before: np.ndarray
	state_after: np.ndarray
	map_state: StateMap = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class IntegrationStep:
	"""Describe one complete numerical step on the method's internal state.

	``map_state`` evaluates the same fixed-time, fixed-duration numerical map on
	another state. Shadow advances used only for output interpolation do not emit
	step observations.
	"""

	dynamics_name: str
	method_name: str
	step_index: int
	time: float
	duration: float
	state_before: np.ndarray
	state_after: np.ndarray
	map_state: StateMap = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ImplicitABBAIntegrationStep(IntegrationStep):
	"""Expose converged ABBA stages without performing diagnostic analysis.

	The numerical method owns the nonlinear projection solve. Analytic tangent
	diagnostics can use these snapshots and ``dynamics`` to evaluate the four
	vector-field Jacobians without importing private solver helpers.
	"""

	formulation_name: str
	start_time: float
	dynamics: GuidingCenterJacobianSystem = field(repr=False, compare=False)
	multiplier: np.ndarray = field(repr=False, compare=False)
	u_initial: np.ndarray = field(repr=False, compare=False)
	v_initial: np.ndarray = field(repr=False, compare=False)
	u_first: np.ndarray = field(repr=False, compare=False)
	v_final: np.ndarray = field(repr=False, compare=False)
	u_final: np.ndarray = field(repr=False, compare=False)


StageObserver: TypeAlias = Callable[[IntegrationStage], None]
StepObserver: TypeAlias = Callable[[IntegrationStep], None]


__all__ = [
	"ImplicitABBAIntegrationStep",
	"IntegrationStage",
	"IntegrationStep",
	"StageObserver",
	"StateMap",
	"StepObserver",
]
