"""BM4 configuration and observation adapters on the shared extended engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, TypeAlias
import numpy as np
from formulations.state import DoubledFormulation
from formulations.gc import GCDoubledMaps, _EnergyQuadraturePoint
from integration.core import IntegrationMethod, NEWTON_ALIASES
from contracts.step import StepInfo, StepResult
from contracts.result import DiagnosticValue
from contracts.observation import ImplicitBM4IntegrationStep, IntegrationStage, StepObserver
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from methods._nonlinear import NonlinearSolver, SolverOptions, _validate_nonlinear_solver
from methods.extended.configuration import _positive_finite, _positive_integer, _nonnegative_finite
from methods.extended.composition import BM4, compose
from methods.extended.projection import solve_projection
from methods.extended.records import ProjectedMapResult as _ProjectedBM4Step
from methods.extended.records import ProjectedMap
from methods.extended.energy import momentum_increment
from methods.extended.jacobians import central_difference_jacobian as _central_difference_jacobian, particle_jacobians, packed_jacobian
from methods.extended.bm4_composition import _advance_composition, stage_events

NewtonJacobianMethod: TypeAlias = Literal['analytic', 'finite_difference']
NEWTON_JACOBIAN_METHODS: tuple[NewtonJacobianMethod, ...] = ('analytic', 'finite_difference')


def _bm4_map(prepared: GCDoubledMaps, t: float, internal_state: np.ndarray, step: float,
             *, energy_points: list[_EnergyQuadraturePoint] | None = None) -> np.ndarray:
    """Compatibility view of the shared BM4 traversal."""
    trace = compose(prepared, BM4, t, internal_state, step)
    if energy_points is not None:
        energy_points.extend(trace.energy_points)
    return trace.state


def _analytic_bm4_map_jacobian(prepared: GCDoubledMaps, t: float, internal_input: np.ndarray, step: float) -> np.ndarray:
    """Expose the shared exact tangent in the historical dense layout."""
    return packed_jacobian(particle_jacobians(prepared, compose(prepared, BM4, t, internal_input, step)))


def _bm4_map_jacobian(prepared: GCDoubledMaps, t: float, internal_input: np.ndarray, step: float,
                      *, relative_step: float, method: NewtonJacobianMethod) -> np.ndarray:
    """Differentiate the complete BM4 recipe for existing diagnostic callers."""
    if method == 'analytic':
        return _analytic_bm4_map_jacobian(prepared, t, internal_input, step)
    return _central_difference_jacobian(lambda state: _bm4_map(prepared, t, state, step),
                                       internal_input, relative_step=relative_step)


def _solve_reduced_projected_bm4_step(
    prepared: GCDoubledMaps, t: float, state: np.ndarray, step: float, *,
    absolute_tolerance: float, relative_tolerance: float, max_iterations: int,
    jacobian_relative_step: float, jacobian_method: NewtonJacobianMethod,
    nonlinear_solver: NonlinearSolver = 'newton', retain_energy_points: bool = False,
) -> _ProjectedBM4Step:
    """Compatibility entry point; integrations bind the common solver directly."""
    return solve_projection(prepared, BM4,
        SolverOptions(nonlinear_solver, absolute_tolerance, relative_tolerance, max_iterations),
        'reduced_multiplier', t, state, step, jacobian_method=jacobian_method,
        jacobian_relative_step=jacobian_relative_step, retain_energy_points=retain_energy_points)


@dataclass(slots=True)
class BM4Implicit(IntegrationMethod[_ProjectedBM4Step]):
	"""Configure physical BM4 with one reduced Hairer projection per cycle.

	The class has no projection-placement or state-extension modes.  Every call
	accepts and returns the physical guiding-centre vector in ``R^(2p)`` and uses
	a doubled ``R^(4p)`` value only while evaluating one complete BM4 cycle,
	where ``p`` is the particle count.

	Parameters
	----------
	coupling_frequency:
		Non-negative harmonic mixing frequency for the two internal GC copies.
		Defaults to zero (no mixing); the reduced Hairer projection remains active.
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

	# Harmonic coupling of the two temporary physical copies.
	coupling_frequency: float = 0.0
	# State-scaled stopping threshold and common nonlinear correction limit.
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

	# Resources owned by one run; excluded from constructor options.
	state_formulation: DoubledFormulation = field(init=False, repr=False, compare=False)
	formulation: GCDoubledMaps = field(init=False, repr=False, compare=False)
	project: ProjectedMap = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Validate and normalize all numeric and enumerated solver controls."""
		# Normalize controls before they are copied into individual runs.
		self.coupling_frequency = _nonnegative_finite(self.coupling_frequency, 'coupling_frequency')
		self.newton_absolute_tolerance = _positive_finite(self.newton_absolute_tolerance, 'newton_absolute_tolerance')
		self.newton_relative_tolerance = _positive_finite(self.newton_relative_tolerance, 'newton_relative_tolerance')
		self.newton_max_iterations = _positive_integer(self.newton_max_iterations, 'newton_max_iterations')
		self.newton_jacobian_relative_step = _positive_finite(self.newton_jacobian_relative_step, 'newton_jacobian_relative_step')
		if self.newton_jacobian_method not in NEWTON_JACOBIAN_METHODS:
			raise ValueError(
				"`newton_jacobian_method` must be 'analytic' or 'finite_difference'."
			)
		self.nonlinear_solver = _validate_nonlinear_solver(self.nonlinear_solver)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Bind the physical BM4 map, accepted event adapter and output metadata.
		Only two physical copies enter the temporary formulation. Per-run metric
		lists, scheduling and observer dispatch belong to the common coordinator.
		"""
		self.state_formulation = DoubledFormulation(problem, request.t_span[0], self.track_energy)
		self.formulation = GCDoubledMaps(problem, self.coupling_frequency)
		self.project = partial(solve_projection, self.formulation, BM4,
		    SolverOptions(self.nonlinear_solver, self.newton_absolute_tolerance,
		                  self.newton_relative_tolerance, self.newton_max_iterations),
		    'reduced_multiplier', jacobian_method=self.newton_jacobian_method,
		    jacobian_relative_step=self.newton_jacobian_relative_step,
		    retain_energy_points=self.track_energy)
		metadata: dict[str, DiagnosticValue] = {
			"nonlinear_solver": self.nonlinear_solver,
			"nonlinear_absolute_tolerance": self.newton_absolute_tolerance,
			"nonlinear_relative_tolerance": self.newton_relative_tolerance,
			"nonlinear_max_iterations": self.newton_max_iterations,
			"newton_jacobian_relative_step": self.newton_jacobian_relative_step,
			"newton_jacobian_method": self.newton_jacobian_method,
			"coupling_frequency": self.coupling_frequency,
			"projection_solver_formulation": "bm4_implicit_reduced",
			"nonlinear_unknown_dimension": problem.initial_state.size,
			"nonlinear_solves_per_step": 1,
		}
		self.initial_state = self.state_formulation.initial_state
		self.metadata = metadata
		self.diagnostic_aliases = NEWTON_ALIASES

	def advance(self, t: float, state: np.ndarray, h: float) -> StepResult[_ProjectedBM4Step]:
		"""Return one projected state and the already computed solve metrics."""
		physical = self.state_formulation.physical(state)
		result = self.project(t, physical, h)
		tolerance = self.newton_absolute_tolerance + self.newton_relative_tolerance * max(
			1.0, float(np.linalg.norm(physical, ord=np.inf))
		)
		increment = None
		if self.track_energy:
			increment = momentum_increment(self.state_formulation, result.energy_points)
		after = self.state_formulation.finish(state, result.state, t + h, increment)
		return StepResult(after, {
			"nonlinear_iterations": result.iterations,
			"residual_evaluations": result.residual_evaluations,
			"nonlinear_residual_norms": result.residual_norm,
			"nonlinear_tolerances": tolerance,
			"projection_multiplier_norms": float(np.linalg.norm(result.multiplier, ord=np.inf)),
		}, result)

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
			coupling_frequency=self.coupling_frequency,
			multiplier=result.multiplier.copy(), base_stages=tuple(base_stages),
		)



__all__ = ["BM4Implicit"]
