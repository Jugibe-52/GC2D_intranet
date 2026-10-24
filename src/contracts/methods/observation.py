"""methods.observation observation records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, TypeAlias
import numpy as np
from contracts.dynamics.protocols import DynamicalSystem
from contracts.methods.quadrature import StageQuadrature


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
	"""Describe one complete numerical step on its observer-facing map domain.

	``map_state`` evaluates the same fixed-time, fixed-duration numerical map on
	another state. This is normally the accepted internal state. A triangular
	state extension may instead expose its closed physical subsystem so existing
	physical stage observers remain reusable; solution diagnostics identify the
	observer state dimension and kind. Shadow advances used only for output
	interpolation do not emit observations. ``dynamics`` retains the exact system
	instance so analytic observers cannot use derivatives from another potential.
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
	quadrature: StageQuadrature | None = field(default=None, repr=False, compare=False, kw_only=True)
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


StageObserver: TypeAlias = Callable[[IntegrationStage], None]


StepObserver: TypeAlias = Callable[[IntegrationStep], None]


__all__ = ['StateMap', 'IntegrationStage', 'IntegrationStep', 'ImplicitIntegrationStep', 'StageObserver', 'StepObserver']
