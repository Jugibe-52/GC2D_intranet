"""Optional observations emitted by numerical stages and complete steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, TypeAlias

import numpy as np

from dynamics import DynamicalSystem, GuidingCenterJacobianSystem


StateMap: TypeAlias = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class AdaptiveIntegrationStep:
	"""One accepted adaptive interval, with its dense output and actual work.

	The interpolant evaluates this accepted trajectory only. A variable-step
	solver retains history and may reject trials, so this event intentionally
	exposes no fixed-duration state map for geometric differentiation.
	"""

	dynamics_name: str
	method_name: str
	step_index: int
	start_time: float
	time: float
	duration: float
	state_before: np.ndarray
	state_after: np.ndarray
	dense_state: Callable[[float | np.ndarray], np.ndarray] = field(repr=False, compare=False)
	function_evaluations: int
	jacobian_evaluations: int
	lu_decompositions: int


AdaptiveStepObserver: TypeAlias = Callable[[AdaptiveIntegrationStep], None]


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
	dynamics: DynamicalSystem | None = field(
		default=None,
		repr=False,
		compare=False,
		kw_only=True,
	)


@dataclass(frozen=True, slots=True)
class GaussLegendre4IntegrationStep(IntegrationStep):
	"""Expose the converged stages of one two-stage Gauss collocation step.

	The inherited state map is the physical fixed-time map, even when the
	integrator also advances the time-conjugate momentum for energy diagnostics.
	The two stage snapshots allow analytic observers to differentiate the ideal
	converged collocation equations without differentiating Newton iterations.
	"""

	first_stage_time: float
	second_stage_time: float
	newton_iterations: int
	residual_evaluations: int
	newton_residual_norm: float
	newton_tolerance: float
	first_stage_state: np.ndarray = field(repr=False, compare=False)
	second_stage_state: np.ndarray = field(repr=False, compare=False)


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
class ABBA2ImplicitIntegrationStep(ImplicitIntegrationStep):
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
class UnprojectedABBAIntegrationStep:
	"""Expose one signed ABBA map inside an unprojected composition.

	The two copies remain independent between consecutive entries. Stage arrays
	use the same component-major physical layout as an implicit ABBA snapshot,
	but this record has no multiplier or nonlinear solve of its own.
	"""

	start_time: float
	time: float
	duration: float
	u_initial: np.ndarray = field(repr=False, compare=False)
	v_initial: np.ndarray = field(repr=False, compare=False)
	u_first: np.ndarray = field(repr=False, compare=False)
	v_final: np.ndarray = field(repr=False, compare=False)
	u_final: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ABBAImplicitCompositionIntegrationStep(ImplicitIntegrationStep):
	"""Expose one outer projection around a continuous ABBA composition.

	Each signed substep retains both spatial copies and has no independent
	multiplier or nonlinear solve. The complete step owns those quantities.
	"""

	multiplier: np.ndarray = field(repr=False, compare=False)
	composition_coefficients: np.ndarray = field(repr=False, compare=False)
	substeps: tuple[UnprojectedABBAIntegrationStep, ...] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ABBA4ImplicitIntegrationStep(ABBAImplicitCompositionIntegrationStep):
	"""Three unprojected ABBA pairs and their one outer projection."""


@dataclass(frozen=True, slots=True)
class ABBA6ImplicitIntegrationStep(ABBAImplicitCompositionIntegrationStep):
	"""Seven unprojected ABBA pairs and their one outer projection."""


@dataclass(frozen=True, slots=True)
class ImplicitBM4IntegrationStep(ImplicitIntegrationStep):
	"""Expose the converged projected-BM4 base cycle to exact observers."""

	coupling_frequency: float
	multiplier: np.ndarray = field(repr=False, compare=False)
	base_stages: tuple[IntegrationStage, ...] = field(repr=False, compare=False)


StageObserver: TypeAlias = Callable[[IntegrationStage], None]
StepObserver: TypeAlias = Callable[[IntegrationStep], None]


__all__ = [
	"AdaptiveIntegrationStep",
	"AdaptiveStepObserver",
	"ABBA2ImplicitIntegrationStep",
	"ABBA4ImplicitIntegrationStep",
	"ABBA6ImplicitIntegrationStep",
	"ABBAImplicitCompositionIntegrationStep",
	"ImplicitBM4IntegrationStep",
	"GaussLegendre4IntegrationStep",
	"ImplicitIntegrationStep",
	"IntegrationStage",
	"IntegrationStep",
	"StageObserver",
	"StateMap",
	"StepObserver",
	"UnprojectedABBAIntegrationStep",
]
