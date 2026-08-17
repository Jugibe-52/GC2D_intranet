"""Optional observations emitted by numerical stages and complete steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, TypeAlias

import numpy as np

from dynamics import DynamicalSystem, GuidingCenterJacobianSystem


StateMap: TypeAlias = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class IntegrationStage:
	"""Describe one direct or adjoint map inside a composed integration step.

	``state_before`` and ``state_after`` are independent snapshots of the packed
	internal state. ``map_state`` evaluates this exact stage—with its duration and
	evaluation time already fixed—on another state of the same shape. Diagnostic
	code can therefore differentiate a stage without duplicating integrator logic.
	For a stage-projected method, every stage map includes that projection.
	``dynamics`` identifies the exact system instance that generated the snapshots;
	exact analytic observers use it to reject accidentally mismatched systems.
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
	dynamics: DynamicalSystem | None = field(
		default=None,
		repr=False,
		compare=False,
	)


@dataclass(frozen=True, slots=True)
class IntegrationStep:
	"""Describe one complete numerical step on the method's internal state.

	``map_state`` evaluates the same fixed-time, fixed-duration numerical map on
	another state. Shadow advances used only for output interpolation do not emit
	step observations. ``dynamics`` retains the exact system instance so analytic
	observers cannot accidentally evaluate derivatives for another potential.
	"""

	dynamics_name: str
	method_name: str
	step_index: int
	time: float
	duration: float
	state_before: np.ndarray
	state_after: np.ndarray
	map_state: StateMap = field(repr=False, compare=False)
	start_time: float = field(default=float("nan"), kw_only=True)
	dynamics: DynamicalSystem | None = field(
		default=None,
		repr=False,
		compare=False,
		kw_only=True,
	)


@dataclass(frozen=True, slots=True)
class ImplicitIntegrationStep(IntegrationStep):
	"""Expose accepted nonlinear-solver metrics for one implicit step."""

	formulation_name: str
	nonlinear_solver: Literal["newton", "broyden"]
	newton_iterations: int
	residual_evaluations: int
	newton_residual_norm: float
	newton_tolerance: float
	projection_multiplier_norm: float


@dataclass(frozen=True, slots=True)
class ImplicitABBAIntegrationStep(ImplicitIntegrationStep):
	"""Expose converged ABBA stages without performing diagnostic analysis.

	Analytic tangent diagnostics use these snapshots and ``dynamics`` to evaluate
	the four vector-field Jacobians without importing private solver helpers.
	"""

	multiplier: np.ndarray = field(repr=False, compare=False)
	u_initial: np.ndarray = field(repr=False, compare=False)
	v_initial: np.ndarray = field(repr=False, compare=False)
	u_first: np.ndarray = field(repr=False, compare=False)
	v_final: np.ndarray = field(repr=False, compare=False)
	u_final: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ImplicitABBA4IntegrationStep(ImplicitIntegrationStep):
	"""Expose the three accepted implicit-ABBA maps in one fourth-order step.

	``substeps`` follows composition order and contains the signed
	``(gamma, delta, gamma)`` durations. Each entry owns its converged multiplier
	and ABBA stage snapshots, allowing diagnostics to form the exact physical
	Jacobian product without differentiating nonlinear iterations. The inherited
	nonlinear counts are sums over all three solves; the residual and tolerance
	identify the accepted substep with the largest residual-to-tolerance ratio.
	"""

	composition_coefficients: np.ndarray = field(repr=False, compare=False)
	substeps: tuple[ImplicitABBAIntegrationStep, ...] = field(
		repr=False,
		compare=False,
	)


@dataclass(frozen=True, slots=True)
class ImplicitBM4IntegrationStep(ImplicitIntegrationStep):
	"""Expose the converged projected-BM4 base cycle to exact observers."""

	coupling_frequency: float
	multiplier: np.ndarray = field(repr=False, compare=False)
	base_stages: tuple[IntegrationStage, ...] = field(repr=False, compare=False)


StageObserver: TypeAlias = Callable[[IntegrationStage], None]
StepObserver: TypeAlias = Callable[[IntegrationStep], None]


__all__ = [
	"ImplicitABBA4IntegrationStep",
	"ImplicitABBAIntegrationStep",
	"ImplicitBM4IntegrationStep",
	"ImplicitIntegrationStep",
	"IntegrationStage",
	"IntegrationStep",
	"StageObserver",
	"StateMap",
	"StepObserver",
]
