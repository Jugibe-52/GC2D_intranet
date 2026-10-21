"""BM4 presets and observation adapters on the common palindromic method."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias
import numpy as np

from contracts.nonlinear import NewtonObserver
from contracts.observation import ImplicitBM4IntegrationStep, StepObserver
from contracts.problem import InitialValueProblem
from contracts.result import DiagnosticValue
from contracts.step import StepInfo, StepResult
from formulations.gc import GCDoubledMaps
from formulations.state import DoubledFormulation
from request import SimulationRequest
from solvers.nonlinear import AlgebraicSolver, NonlinearSolver
from methods._validation import _positive_finite
from methods.extended.configuration import StateExtension
from methods.extended.core.composition import BM4, Composition
from methods.extended.core.records import ProjectedMap, ProjectedMapResult as _ProjectedBM4Step
from methods.extended.observations import stage_events
from methods.extended.palindromic import ProjectedPalindromicMethod, MidpointPalindromicMethod

NewtonJacobianMethod: TypeAlias = Literal['analytic', 'finite_difference']
NEWTON_JACOBIAN_METHODS: tuple[NewtonJacobianMethod, ...] = ('analytic', 'finite_difference')


@dataclass(slots=True)
class BM4Implicit(ProjectedPalindromicMethod):
	"""Configure uncoupled BM4 with one reduced Hairer projection per cycle.

	The class has no projection-placement or state-extension modes.  Every call
	accepts and returns the physical guiding-centre vector in ``R^(2p)`` and uses
	a doubled ``R^(4p)`` value only while evaluating one complete BM4 cycle,
	where ``p`` is the particle count.

	Parameters
	----------
	newton_absolute_tolerance, newton_relative_tolerance:
		Positive terms in the reduced-residual stopping threshold
		``atol + rtol * max(1, ||z_n||_inf)``.  These controls also apply when
		``nonlinear_solver="broyden"``; their historical names are retained as
		part of the public API.
	newton_max_iterations:
		Maximum number of nonlinear corrections before rejecting a step.
	newton_jacobian_relative_step:
		Positive scale factor for centered differences.  It is used only by
		finite-difference Newton; analytic Newton and Broyden do not consult it.
	newton_jacobian_method:
		``"analytic"`` for the exact ordered GC stage product or
		``"finite_difference"`` for differentiation of the complete base map.
		This selection is consulted only by Newton.  Broyden starts from ``4 I``
		and updates that reduced residual-Jacobian approximation by secants.
	nonlinear_solver:
		``"newton"`` or ``"broyden"`` for the same reduced Hairer equation.
	progress:
		Whether the shared fixed-grid driver reports integration progress.
	step_observer:
		Optional callback receiving one :class:`ImplicitBM4IntegrationStep` per
		accepted main-grid step.  Output-only shadow steps are not observed.
	"""

	recipe: Composition = field(default=BM4, init=False)

	solver: AlgebraicSolver | None = field(default=None, kw_only=True)

	# np.ndarray-scaled stopping threshold and common nonlinear correction limit.
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	# Cube-root epsilon balances truncation and roundoff for centered differences.
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	newton_jacobian_method: NewtonJacobianMethod = "analytic"
	nonlinear_solver: NonlinearSolver = "newton"
	track_energy: bool = False
	# Fixed-grid presentation and optional accepted-step instrumentation.
	progress: bool = False
	step_observer: StepObserver | None = None
	newton_observer: NewtonObserver | None = field(default=None, kw_only=True)

	# Resources owned by one run; excluded from constructor options.
	state_formulation: DoubledFormulation = field(init=False, repr=False, compare=False)
	formulation: GCDoubledMaps = field(init=False, repr=False, compare=False)
	project: ProjectedMap = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Validate and normalize all numeric and enumerated solver controls."""
		self._validate_solver_controls()
		self.newton_jacobian_relative_step = _positive_finite(self.newton_jacobian_relative_step, 'newton_jacobian_relative_step')
		if self.newton_jacobian_method not in NEWTON_JACOBIAN_METHODS:
			raise ValueError(
				"`newton_jacobian_method` must be 'analytic' or 'finite_difference'."
			)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Bind the physical BM4 map, accepted event adapter and output metadata.
		Only two physical copies enter the temporary formulation. Per-run metric
		lists, scheduling and observer dispatch belong to the common coordinator.
		"""
		self._prepare_projection(problem, request, 'reduced_multiplier',
			jacobian=self.newton_jacobian_method, relative_step=self.newton_jacobian_relative_step)
		metadata: dict[str, DiagnosticValue] = {
			"nonlinear_solver": self.nonlinear_solver,
			"nonlinear_absolute_tolerance": self.newton_absolute_tolerance,
			"nonlinear_relative_tolerance": self.newton_relative_tolerance,
			"nonlinear_max_iterations": self.newton_max_iterations,
			"newton_jacobian_relative_step": self.newton_jacobian_relative_step,
			"newton_jacobian_method": self.newton_jacobian_method,
			"coupling_frequency": 0.0,
			"projection_solver_formulation": "bm4_implicit_reduced",
			"nonlinear_unknown_dimension": problem.initial_state.size,
			"nonlinear_solves_per_step": 1,
		}
		self.initial_state = self.state_formulation.initial_state
		self.metadata = {**metadata, **self._operation_metadata()}


	def build_observation(self, info: StepInfo, step: StepResult[_ProjectedBM4Step]) -> ImplicitBM4IntegrationStep:
		"""Reconstruct converged stage snapshots only for a requested event."""
		result = step.details
		base_stages = stage_events(self.formulation, result.trace,
		    step_index=info.index, formulation_name="GCExtendedFormulation",
		    method_name=type(self).__name__)

		def map_state(candidate: np.ndarray) -> np.ndarray:
			"""Evaluate the same physical map without collecting or observing."""
			return self.project(info.time, candidate, info.duration).state

		return ImplicitBM4IntegrationStep(
			dynamics_name=self.formulation.dynamics_name, method_name=type(self).__name__,
			step_index=info.index, start_time=info.time,
			time=info.time + info.duration, duration=info.duration,
			state_before=self.state_formulation.physical(info.state_before).copy(), state_after=result.state.copy(),
			map_state=map_state, dynamics=self.formulation.dynamics,
			formulation_name="bm4_implicit_reduced", nonlinear_solver=self.nonlinear_solver,
			newton_iterations=result.iterations, residual_evaluations=result.residual_evaluations,
			newton_residual_norm=result.residual_norm,
			newton_tolerance=float(step.statistics["nonlinear_tolerances"]),
			projection_multiplier_norm=float(step.statistics["projection_multiplier_norms"]),
			coupling_frequency=0.0,
			multiplier=result.multiplier.copy(), base_stages=base_stages,
		)

@dataclass(slots=True)
class BM4Midpoint(MidpointPalindromicMethod):
	"""Uncoupled fourth-order BM4 palindrome with one final arithmetic mean.

	The common base owns execution, optional passive energy and observations.
	Arithmetic projection does not guarantee reversibility or symplecticity.
	"""

	state_extension: StateExtension = "physical"
	progress: bool = False
	step_observer: StepObserver | None = None
	track_energy: bool = False
	recipe: Composition = field(default=BM4, init=False)

	def _preset_metadata(self) -> dict[str, DiagnosticValue]:
		return {"coupling_frequency": 0.0, "composition_stage_count": 12}

__all__ = ["BM4Implicit", "BM4Midpoint"]
