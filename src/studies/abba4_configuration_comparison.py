"""Sixteen-configuration ABBA4 comparison on separate particle trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from itertools import product
from multiprocessing import get_context
from pathlib import Path
import sys
from time import perf_counter
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import numpy as np
from threadpoolctl import ThreadpoolController

from diagnostics import StoredReferenceTrajectory
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import GC2DH5Potential, Potential
from simulation import (
	ABBA4Implicit,
	ABBA4ImplicitSingleProjection,
	InitialValueProblem,
	NonlinearSolver,
	NumericalMethod,
	ProjectionFormulation,
	SimulationRequest,
	Solution,
	StateExtension,
	simulate,
)

from ._trajectory_accuracy import (
	reference_distance_convention,
	reference_indices_for_times,
)
from ._validation import (
	integer_ratio,
	nonnegative_finite,
	positive_finite,
	positive_integer,
)
from .reference_trajectory import potential_fingerprint


ABBA4ConfigurationMethod: TypeAlias = Literal[
	"ABBA4Implicit",
	"ABBA4ImplicitSingleProjection",
]

ABBA4_CONFIGURATION_PARTICLE_COUNT = 10
_METHOD_NAMES: tuple[ABBA4ConfigurationMethod, ...] = (
	"ABBA4Implicit",
	"ABBA4ImplicitSingleProjection",
)
_STATE_EXTENSIONS: tuple[StateExtension, ...] = (
	"shared_time",
	"fully_extended",
)
_PROJECTION_FORMULATIONS: tuple[ProjectionFormulation, ...] = (
	"reduced_multiplier",
	"simultaneous_state_multiplier",
)
_NONLINEAR_SOLVERS: tuple[NonlinearSolver, ...] = ("newton", "broyden")

_METHOD_SLUGS: Mapping[ABBA4ConfigurationMethod, str] = MappingProxyType(
	{
		"ABBA4Implicit": "abba4_implicit",
		"ABBA4ImplicitSingleProjection": (
			"abba4_implicit_single_projection"
		),
	}
)
_METHOD_LABELS: Mapping[ABBA4ConfigurationMethod, str] = MappingProxyType(
	{
		"ABBA4Implicit": "ABBA4 (three projections)",
		"ABBA4ImplicitSingleProjection": "SP-ABBA4 (single projection)",
	}
)
_EXTENSION_LABELS: Mapping[StateExtension, str] = MappingProxyType(
	{
		"shared_time": "shared time R6",
		"fully_extended": "fully extended R8",
		"physical": "physical R4",
	}
)
_FORMULATION_LABELS: Mapping[ProjectionFormulation, str] = MappingProxyType(
	{
		"reduced_multiplier": "reduced multiplier",
		"simultaneous_state_multiplier": "simultaneous state-multiplier",
	}
)


@dataclass(frozen=True, slots=True)
class ABBA4ConfigurationVariant:
	"""Stable identity for one point of the requested configuration cube."""

	key: str
	label: str
	method_name: ABBA4ConfigurationMethod
	state_extension: StateExtension
	projection_formulation: ProjectionFormulation
	nonlinear_solver: NonlinearSolver


def _configuration_variant(
	method_name: ABBA4ConfigurationMethod,
	state_extension: StateExtension,
	projection_formulation: ProjectionFormulation,
	nonlinear_solver: NonlinearSolver,
) -> ABBA4ConfigurationVariant:
	"""Construct one public descriptor from the four orthogonal axes."""
	key = "__".join(
		(
			_METHOD_SLUGS[method_name],
			state_extension,
			projection_formulation,
			nonlinear_solver,
		)
	)
	label = " | ".join(
		(
			_METHOD_LABELS[method_name],
			_EXTENSION_LABELS[state_extension],
			_FORMULATION_LABELS[projection_formulation],
			nonlinear_solver.title(),
		)
	)
	return ABBA4ConfigurationVariant(
		key=key,
		label=label,
		method_name=method_name,
		state_extension=state_extension,
		projection_formulation=projection_formulation,
		nonlinear_solver=nonlinear_solver,
	)


ABBA4_CONFIGURATION_VARIANTS: tuple[ABBA4ConfigurationVariant, ...] = tuple(
	_configuration_variant(method_name, extension, formulation, solver)
	for method_name, extension, formulation, solver in product(
		_METHOD_NAMES,
		_STATE_EXTENSIONS,
		_PROJECTION_FORMULATIONS,
		_NONLINEAR_SOLVERS,
	)
)
ABBA4_CONFIGURATION_KEYS: tuple[str, ...] = tuple(
	variant.key for variant in ABBA4_CONFIGURATION_VARIANTS
)
ABBA4_CONFIGURATION_LABELS: Mapping[str, str] = MappingProxyType(
	{variant.key: variant.label for variant in ABBA4_CONFIGURATION_VARIANTS}
)
_ABBA4_CONFIGURATION_VARIANTS_BY_KEY: Mapping[
	str,
	ABBA4ConfigurationVariant,
] = MappingProxyType(
	{variant.key: variant for variant in ABBA4_CONFIGURATION_VARIANTS}
)
_PARALLEL_PROGRESS_BAR_WIDTH = 24
_PARALLEL_PROGRESS_HEARTBEAT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class ABBA4ConfigurationComparisonConfig:
	"""Physical, numerical, and process-level execution controls.

	``worker_count=1`` preserves the serial execution and isolated timing
	semantics. Larger values distribute complete one-particle trajectories over
	spawned worker processes; the time steps within each trajectory remain
	strictly sequential.
	"""

	t_span: tuple[float, float] = (0.0, 4.0)
	integration_step: float = 0.0025
	save_interval: float = 0.01
	rho: float = 0.0
	absolute_tolerance: float = 1e-14
	relative_tolerance: float = 1e-13
	max_iterations: int = 40
	progress: bool = False
	particle_count: int = ABBA4_CONFIGURATION_PARTICLE_COUNT
	worker_count: int = 1

	def __post_init__(self) -> None:
		"""Normalize all reproducibility controls and require aligned grids."""
		span = np.asarray(self.t_span, dtype=float)
		if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
			raise ValueError("`t_span` must contain two finite, increasing times.")
		object.__setattr__(self, "t_span", (float(span[0]), float(span[1])))
		for name in (
			"integration_step",
			"save_interval",
			"absolute_tolerance",
			"relative_tolerance",
		):
			object.__setattr__(
				self,
				name,
				positive_finite(getattr(self, name), name),
			)
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		object.__setattr__(
			self,
			"max_iterations",
			positive_integer(self.max_iterations, "max_iterations"),
		)
		object.__setattr__(self, "progress", bool(self.progress))
		object.__setattr__(
			self,
			"particle_count",
			positive_integer(self.particle_count, "particle_count"),
		)
		object.__setattr__(
			self,
			"worker_count",
			positive_integer(self.worker_count, "worker_count"),
		)
		integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.integration_step,
			"duration / integration_step",
		)
		integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		)
		integer_ratio(
			self.save_interval,
			self.integration_step,
			"save_interval / integration_step",
		)

	@property
	def step_count(self) -> int:
		"""Return the number of accepted complete steps in every simulation."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.integration_step,
			"duration / integration_step",
		)

	@property
	def output_sample_count(self) -> int:
		"""Return the number of aligned saved nodes, including both endpoints."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		) + 1


@dataclass(frozen=True, slots=True)
class ABBA4ConfigurationComparisonSummary:
	"""The five requested metrics plus the stable configuration identity.

	When ``worker_count > 1``, ``total_runtime_seconds`` is the sum of contended
	per-worker wall times, not the overall process-pool elapsed time. Use serial
	execution for isolated method-runtime comparisons.
	"""

	key: str
	label: str
	method_name: ABBA4ConfigurationMethod
	state_extension: StateExtension
	projection_formulation: ProjectionFormulation
	nonlinear_solver: NonlinearSolver
	mean_trajectory_error: float
	final_trajectory_error: float
	total_runtime_seconds: float
	mean_iterations_per_solve: float
	mean_relative_energy_error: float


def _readonly_runtime_array(
	value: np.ndarray,
	*,
	particle_count: int,
) -> np.ndarray:
	"""Own and freeze one positive wall-clock measurement per particle."""
	array = np.array(value, dtype=float, copy=True)
	if array.shape != (particle_count,):
		raise ValueError(
			"Every configuration must contain one runtime sample per particle."
		)
	if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
		raise ValueError("Every trajectory runtime must be positive and finite.")
	array.setflags(write=False)
	return array


def _expected_dimensions(variant: ABBA4ConfigurationVariant) -> tuple[int, int, int]:
	"""Return accepted, base-map, and nonlinear workspace dimensions."""
	if variant.state_extension == "shared_time":
		return (
			4,
			6,
			2 if variant.projection_formulation == "reduced_multiplier" else 6,
		)
	return (
		4,
		8,
		4 if variant.projection_formulation == "reduced_multiplier" else 12,
	)


def _generalized_energy_history(
	dynamics: GuidingCenterDynamics,
	variant: ABBA4ConfigurationVariant,
	solution: Solution,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return aligned generalized energy and drift for one trajectory.

	The shared-time solver stores the conjugate momentum but deliberately does
	not duplicate the fully-extended solver's energy diagnostics.  Evaluating
	the Hamiltonian here gives both extensions the same energy-accounting work
	inside the wall-clock interval used by the comparison.
	"""
	diagnostics = solution.diagnostics
	if variant.state_extension == "shared_time":
		hamiltonian = np.asarray(
			dynamics.hamiltonian(solution.t, solution.states),
			dtype=float,
		).reshape(-1)
		kappa = np.asarray(
			diagnostics["extended_kappa"],
			dtype=float,
		).reshape(-1)
		if hamiltonian.shape != solution.t.shape or kappa.shape != solution.t.shape:
			raise ValueError("Shared-time energy histories are not aligned.")
		generalized = hamiltonian + kappa
		error = generalized - generalized[0]
	else:
		generalized = np.asarray(
			diagnostics["generalized_energy"],
			dtype=float,
		).reshape(-1)
		error = np.asarray(
			diagnostics["generalized_energy_error"],
			dtype=float,
		).reshape(-1)
		if generalized.shape != solution.t.shape or error.shape != solution.t.shape:
			raise ValueError("Fully extended energy histories are not aligned.")
		if not np.allclose(
			error,
			generalized - generalized[0],
			rtol=float(64.0 * np.finfo(float).eps),
			atol=float(64.0 * np.finfo(float).eps),
		):
			raise ValueError("Fully extended energy diagnostics are inconsistent.")
	if not np.all(np.isfinite(generalized)) or not np.all(np.isfinite(error)):
		raise ValueError("Generalized-energy histories must be finite.")
	return generalized, error


@dataclass(frozen=True, slots=True)
class ABBA4ConfigurationComparisonResult:
	"""Aligned one-particle solutions and task timings for all configurations."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: GCInitialConfiguration
	reference: StoredReferenceTrajectory
	config: ABBA4ConfigurationComparisonConfig
	reference_sample_indices: np.ndarray
	solutions: Mapping[str, tuple[Solution, ...]]
	runtimes: Mapping[str, np.ndarray]

	def __post_init__(self) -> None:
		"""Freeze nested values and enforce complete alignment and diagnostics."""
		if not isinstance(self.potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		if not isinstance(self.dynamics, GuidingCenterDynamics):
			raise TypeError("`dynamics` must be GuidingCenterDynamics.")
		if not isinstance(self.initial_configuration, GCInitialConfiguration):
			raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
		if not isinstance(self.reference, StoredReferenceTrajectory):
			raise TypeError("`reference` must be a StoredReferenceTrajectory.")
		if not isinstance(self.config, ABBA4ConfigurationComparisonConfig):
			raise TypeError("`config` must be ABBA4ConfigurationComparisonConfig.")
		if tuple(self.solutions) != ABBA4_CONFIGURATION_KEYS:
			raise ValueError("Solutions must follow all sixteen stable configuration keys.")
		if tuple(self.runtimes) != ABBA4_CONFIGURATION_KEYS:
			raise ValueError("Runtimes must follow all sixteen stable configuration keys.")

		initial_state = self.initial_configuration.initial_state
		if initial_state is None:
			raise ValueError("The initial configuration must contain a state.")
		particle_count = self.config.particle_count
		if (
			self.initial_configuration.layout.particle_count(initial_state)
			!= particle_count
		):
			raise ValueError(
				"The initial configuration must contain exactly "
				f"particle_count={particle_count} initial conditions."
			)
		initial_x, initial_y = self.initial_configuration.positions(initial_state)
		common_times: np.ndarray | None = None
		frozen_solutions: dict[str, tuple[Solution, ...]] = {}
		frozen_runtimes: dict[str, np.ndarray] = {}
		for variant in ABBA4_CONFIGURATION_VARIANTS:
			trajectory_solutions = tuple(self.solutions[variant.key])
			if len(trajectory_solutions) != particle_count:
				raise ValueError(
					"Every configuration must contain one solution per particle."
				)
			expected_dimensions = _expected_dimensions(variant)
			for particle, solution in enumerate(trajectory_solutions):
				if not isinstance(solution, Solution):
					raise TypeError("Every configuration trajectory must be a Solution.")
				if not isinstance(solution.source, GCInitialConfiguration):
					raise TypeError("Every solution must use a GC initial configuration.")
				expected_initial = np.asarray(
					(initial_x[particle], initial_y[particle]),
					dtype=float,
				)
				if not np.array_equal(solution.states[:, 0], expected_initial):
					raise ValueError("A trajectory used the wrong initial condition.")
				if common_times is None:
					common_times = solution.t
				elif not np.array_equal(solution.t, common_times):
					raise ValueError(
						"All configuration trajectories must share one saved-time grid."
					)
				diagnostics = solution.diagnostics
				if int(diagnostics.get("step_count", -1)) != self.config.step_count:
					raise ValueError("A trajectory used an inconsistent integration grid.")
				if diagnostics.get("state_extension") != variant.state_extension:
					raise ValueError("A trajectory used the wrong state extension.")
				if diagnostics.get("projection_formulation") != (
					variant.projection_formulation
				):
					raise ValueError("A trajectory used the wrong projection formulation.")
				if diagnostics.get("nonlinear_solver") != variant.nonlinear_solver:
					raise ValueError("A trajectory used the wrong nonlinear solver.")
				actual_dimensions = tuple(
					int(diagnostics[name])
					for name in (
						"accepted_internal_state_dimension",
						"base_splitting_state_dimension",
						"nonlinear_unknown_dimension",
					)
				)
				if actual_dimensions != expected_dimensions:
					raise ValueError("A trajectory reported inconsistent state dimensions.")
			frozen_solutions[variant.key] = trajectory_solutions
			frozen_runtimes[variant.key] = _readonly_runtime_array(
				self.runtimes[variant.key],
				particle_count=particle_count,
			)

		assert common_times is not None
		indices = np.array(self.reference_sample_indices, dtype=np.int64, copy=True)
		if (
			indices.shape != common_times.shape
			or np.any(indices < 0)
			or np.any(indices >= self.reference.times.size)
			or not np.array_equal(self.reference.times[indices], common_times)
		):
			raise ValueError("Saved times do not align with the certified reference.")
		indices.setflags(write=False)
		object.__setattr__(self, "reference_sample_indices", indices)
		object.__setattr__(
			self,
			"solutions",
			MappingProxyType(frozen_solutions),
		)
		object.__setattr__(self, "runtimes", MappingProxyType(frozen_runtimes))

	@property
	def variants(self) -> tuple[ABBA4ConfigurationVariant, ...]:
		"""Return all configuration descriptors in stable table order."""
		return ABBA4_CONFIGURATION_VARIANTS

	@property
	def times(self) -> np.ndarray:
		"""Return the common saved-time grid."""
		return self.solutions[ABBA4_CONFIGURATION_KEYS[0]][0].t

	def _trajectory_distances(self, key: str) -> np.ndarray:
		"""Return Euclidean errors with shape ``(particles, saved_times)``."""
		particle_count = self.config.particle_count
		reference = self.reference.states[:, self.reference_sample_indices]
		distances = np.empty(
			(particle_count, self.times.size),
			dtype=float,
		)
		for particle, solution in enumerate(self.solutions[key]):
			delta_x = solution.states[0] - reference[particle]
			delta_y = solution.states[1] - reference[particle_count + particle]
			distances[particle] = np.hypot(delta_x, delta_y)
		return distances

	def _relative_energy_errors(
		self,
		variant: ABBA4ConfigurationVariant,
	) -> np.ndarray:
		"""Return normalized absolute generalized-energy errors for all paths."""
		values = np.empty(
			(self.config.particle_count, self.times.size),
			dtype=float,
		)
		for particle, solution in enumerate(self.solutions[variant.key]):
			generalized, error = _generalized_energy_history(
				self.dynamics,
				variant,
				solution,
			)
			scale = max(abs(float(generalized[0])), float(np.finfo(float).eps))
			values[particle] = np.abs(error) / scale
		return values

	def summaries(self) -> tuple[ABBA4ConfigurationComparisonSummary, ...]:
		"""Return sixteen rows, including aggregate trajectory-task runtime."""
		rows: list[ABBA4ConfigurationComparisonSummary] = []
		for variant in ABBA4_CONFIGURATION_VARIANTS:
			distances = self._trajectory_distances(variant.key)
			total_iterations = 0
			total_solves = 0
			for solution in self.solutions[variant.key]:
				diagnostics = solution.diagnostics
				step_count = int(diagnostics["step_count"])
				iterations = np.asarray(
					diagnostics["nonlinear_iterations"],
					dtype=int,
				)
				if iterations.shape != (step_count,) or np.any(iterations < 0):
					raise ValueError("Nonlinear-iteration histories are not aligned.")
				solves_per_step = int(diagnostics["nonlinear_solves_per_step"])
				if solves_per_step < 1:
					raise ValueError("Every implicit step must contain a nonlinear solve.")
				total_iterations += int(np.sum(iterations))
				total_solves += step_count * solves_per_step
			energy_errors = self._relative_energy_errors(variant)
			rows.append(
				ABBA4ConfigurationComparisonSummary(
					key=variant.key,
					label=variant.label,
					method_name=variant.method_name,
					state_extension=variant.state_extension,
					projection_formulation=variant.projection_formulation,
					nonlinear_solver=variant.nonlinear_solver,
					mean_trajectory_error=float(np.sqrt(np.mean(distances**2))),
					final_trajectory_error=float(
						np.sqrt(np.mean(distances[:, -1] ** 2))
					),
					total_runtime_seconds=float(
						np.sum(self.runtimes[variant.key])
					),
					mean_iterations_per_solve=(
						float(total_iterations) / float(total_solves)
					),
					mean_relative_energy_error=float(np.mean(energy_errors)),
				)
			)
		return tuple(rows)


def _method_for_variant(
	variant: ABBA4ConfigurationVariant,
	config: ABBA4ConfigurationComparisonConfig,
) -> NumericalMethod:
	"""Construct one numerical method with common nonlinear controls."""
	method_type: type[ABBA4Implicit] | type[ABBA4ImplicitSingleProjection]
	if variant.method_name == "ABBA4Implicit":
		method_type = ABBA4Implicit
	else:
		method_type = ABBA4ImplicitSingleProjection
	return method_type(
		state_extension=variant.state_extension,
		projection_formulation=variant.projection_formulation,
		nonlinear_solver=variant.nonlinear_solver,
		newton_absolute_tolerance=config.absolute_tolerance,
		newton_relative_tolerance=config.relative_tolerance,
		newton_max_iterations=config.max_iterations,
		progress=config.progress,
	)


def _alternating_particle_order(particle_count: int) -> tuple[int, ...]:
	"""Interleave low and high indices to avoid monotone trajectory ordering."""
	order: list[int] = []
	left = 0
	right = particle_count - 1
	while left <= right:
		order.append(left)
		left += 1
		if left <= right:
			order.append(right)
			right -= 1
	return tuple(order)


@dataclass(frozen=True, slots=True)
class _GC2DH5PotentialSnapshot:
	"""Pickle-safe processed HDF5 field used to initialize spawned workers."""

	x: np.ndarray
	y: np.ndarray
	mean_value: np.ndarray | None
	fluctuations: np.ndarray | None
	frequencies: np.ndarray
	source_field_indices: np.ndarray
	source_x: np.ndarray
	source_y: np.ndarray
	source_frequencies: np.ndarray
	characteristic_length: float | None
	characteristic_period: float | None
	normalization_factor: float
	attributes: dict[str, np.ndarray]
	interpolation_order: int
	source_path: str | None

	@classmethod
	def from_potential(
		cls,
		potential: GC2DH5Potential,
	) -> _GC2DH5PotentialSnapshot:
		"""Capture the selected and resampled fields without the source HDF5."""
		return cls(
			x=potential.x,
			y=potential.y,
			mean_value=potential.mean_value,
			fluctuations=potential.fluctuations,
			frequencies=potential.frequencies,
			source_field_indices=potential.source_field_indices,
			source_x=potential.source_x,
			source_y=potential.source_y,
			source_frequencies=potential.source_frequencies,
			characteristic_length=potential.characteristic_length,
			characteristic_period=potential.characteristic_period,
			normalization_factor=potential.normalization_factor,
			attributes=dict(potential.attributes),
			interpolation_order=potential.interpolation_order,
			source_path=(
				None
				if potential.source_path is None
				else str(potential.source_path)
			),
		)

	def restore(self) -> GC2DH5Potential:
		"""Rebuild runtime splines once inside one worker process."""
		return GC2DH5Potential(
			self.x,
			self.y,
			self.mean_value,
			self.fluctuations,
			self.frequencies,
			source_field_indices=self.source_field_indices,
			source_x=self.source_x,
			source_y=self.source_y,
			source_frequencies=self.source_frequencies,
			characteristic_length=self.characteristic_length,
			characteristic_period=self.characteristic_period,
			normalization_factor=self.normalization_factor,
			attributes=self.attributes,
			interpolation_order=self.interpolation_order,
			source_path=(
				None if self.source_path is None else Path(self.source_path)
			),
		)


_WorkerPotentialPayload: TypeAlias = Potential | _GC2DH5PotentialSnapshot


@dataclass(frozen=True, slots=True)
class _ABBA4TrajectoryTask:
	"""One independent configuration-particle trajectory."""

	variant_key: str
	particle: int


@dataclass(frozen=True, slots=True)
class _ABBA4TrajectoryPayload:
	"""Pickle-safe trajectory data returned by a worker process."""

	variant_key: str
	particle: int
	times: np.ndarray
	states: np.ndarray
	diagnostics: dict[str, Any]
	runtime_seconds: float


@dataclass(frozen=True, slots=True)
class _ABBA4WorkerContext:
	"""Objects constructed once and reused by consecutive tasks in one worker."""

	dynamics: GuidingCenterDynamics
	request: SimulationRequest
	problems: tuple[InitialValueProblem, ...]
	methods: Mapping[str, NumericalMethod]


_ABBA4_WORKER_CONTEXT: _ABBA4WorkerContext | None = None
_ABBA4_WORKER_THREADPOOL_LIMITER: Any = None


def _worker_potential_payload(potential: Potential) -> _WorkerPotentialPayload:
	"""Return a spawn-safe potential representation without rereading HDF5."""
	if isinstance(potential, GC2DH5Potential):
		return _GC2DH5PotentialSnapshot.from_potential(potential)
	return potential


def _initialize_abba4_worker(
	potential_payload: _WorkerPotentialPayload,
	particle_coordinates: tuple[tuple[float, float], ...],
	config: ABBA4ConfigurationComparisonConfig,
) -> None:
	"""Build process-local dynamics, methods, and single-particle problems."""
	global _ABBA4_WORKER_CONTEXT, _ABBA4_WORKER_THREADPOOL_LIMITER
	# NumPy is imported before the initializer under ``spawn``. Runtime limits
	# therefore prevent process-by-BLAS oversubscription more reliably than
	# environment variables set by the calling notebook.
	_ABBA4_WORKER_THREADPOOL_LIMITER = ThreadpoolController().limit(limits=1)
	potential = (
		potential_payload.restore()
		if isinstance(potential_payload, _GC2DH5PotentialSnapshot)
		else potential_payload
	)
	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.integration_step,
		sample_count=config.output_sample_count,
	)
	particle_configurations = tuple(
		GCInitialConfiguration.from_components(
			x=np.asarray([x], dtype=float),
			y=np.asarray([y], dtype=float),
		)
		for x, y in particle_coordinates
	)
	worker_config = replace(config, progress=False)
	_ABBA4_WORKER_CONTEXT = _ABBA4WorkerContext(
		dynamics=dynamics,
		request=request,
		problems=tuple(
			InitialValueProblem(dynamics, particle_configuration)
			for particle_configuration in particle_configurations
		),
		methods={
			variant.key: _method_for_variant(variant, worker_config)
			for variant in ABBA4_CONFIGURATION_VARIANTS
		},
	)


def _run_abba4_trajectory_task(
	task: _ABBA4TrajectoryTask,
) -> _ABBA4TrajectoryPayload:
	"""Integrate and time one complete trajectory inside a worker process."""
	context = _ABBA4_WORKER_CONTEXT
	if context is None:
		raise RuntimeError("The ABBA4 worker process was not initialized.")
	variant = _ABBA4_CONFIGURATION_VARIANTS_BY_KEY[task.variant_key]
	started = perf_counter()
	solution = simulate(
		context.problems[task.particle],
		context.methods[task.variant_key],
		context.request,
	)
	_generalized_energy_history(context.dynamics, variant, solution)
	runtime_seconds = perf_counter() - started
	return _ABBA4TrajectoryPayload(
		variant_key=task.variant_key,
		particle=task.particle,
		times=solution.t,
		states=solution.states,
		diagnostics=dict(solution.diagnostics),
		runtime_seconds=runtime_seconds,
	)


def _validated_reference_samples(
	potential: Potential,
	dynamics: GuidingCenterDynamics,
	initial_configuration: GCInitialConfiguration,
	reference: StoredReferenceTrajectory,
	config: ABBA4ConfigurationComparisonConfig,
	request: SimulationRequest,
) -> np.ndarray:
	"""Validate reference identity and locate this study's aligned subinterval."""
	initial_state = initial_configuration.initial_state
	if initial_state is None or not np.array_equal(initial_state, reference.initial_state):
		raise ValueError("The comparison initial state differs from the reference.")
	reference_particle_count = reference.metadata.get("particle_count")
	if reference_particle_count is not None:
		try:
			stored_particle_count = positive_integer(
				reference_particle_count,
				"reference particle_count",
			)
		except ValueError as exc:
			raise ValueError("Reference particle-count metadata is invalid.") from exc
		if stored_particle_count != config.particle_count:
			raise ValueError(
				"The reference particle count differs from the configured particle_count."
			)
	if reference_distance_convention(reference) != "euclidean":
		raise ValueError("The real HDF5 comparison requires Euclidean reference errors.")
	reference_config = reference.metadata.get("config")
	if not isinstance(reference_config, Mapping):
		raise ValueError("Reference numerical configuration is missing.")
	if "rho" not in reference_config or float(reference_config["rho"]) != config.rho:
		raise ValueError("The comparison rho differs from the reference.")
	if config.t_span[0] != float(reference.times[0]):
		raise ValueError(
			"The comparison interval must start at the reference initial time; "
			"only aligned reference prefixes are supported."
		)
	indices = reference_indices_for_times(reference, request.output_times)
	if "dynamics_fingerprint_sha256" in reference.metadata:
		stored_fingerprint = reference.metadata["dynamics_fingerprint_sha256"]
		actual_fingerprint = potential_fingerprint(dynamics.effective_potential)
		if stored_fingerprint != actual_fingerprint:
			raise ValueError("The interpolated comparison dynamics differs from the reference.")
	if potential is not dynamics.potential:
		raise ValueError("The comparison dynamics must use the supplied potential.")
	return indices


def _empty_trajectory_rows(
	particle_count: int,
) -> tuple[dict[str, dict[int, Solution]], dict[str, np.ndarray]]:
	"""Allocate stable variant rows for solutions and per-particle timings."""
	return (
		{key: {} for key in ABBA4_CONFIGURATION_KEYS},
		{
			key: np.empty(particle_count, dtype=float)
			for key in ABBA4_CONFIGURATION_KEYS
		},
	)


def _sequential_task_schedule(
	particle_count: int,
) -> tuple[_ABBA4TrajectoryTask, ...]:
	"""Retain the historical alternating order used for isolated timings."""
	tasks: list[_ABBA4TrajectoryTask] = []
	for schedule_index, particle in enumerate(
		_alternating_particle_order(particle_count)
	):
		variant_order = (
			ABBA4_CONFIGURATION_VARIANTS
			if schedule_index % 2 == 0
			else tuple(reversed(ABBA4_CONFIGURATION_VARIANTS))
		)
		tasks.extend(
			_ABBA4TrajectoryTask(variant.key, particle)
			for variant in variant_order
		)
	return tuple(tasks)


def _parallel_task_schedule(
	particle_count: int,
) -> tuple[_ABBA4TrajectoryTask, ...]:
	"""Submit empirically heavier R8/Newton trajectories before short work."""
	variant_order = tuple(
		sorted(
			ABBA4_CONFIGURATION_VARIANTS,
			key=lambda variant: (
				variant.state_extension == "fully_extended",
				variant.nonlinear_solver == "newton",
				variant.method_name == "ABBA4Implicit",
				variant.projection_formulation
				== "simultaneous_state_multiplier",
			),
			reverse=True,
		)
	)
	particle_order = _alternating_particle_order(particle_count)
	return tuple(
		_ABBA4TrajectoryTask(variant.key, particle)
		for variant in variant_order
		for particle in particle_order
	)


def _trajectory_progress_bar(
	completed_run_count: int,
	total_run_count: int,
) -> str:
	"""Return the fixed-width global trajectory progress bar."""
	completion_fraction = completed_run_count / total_run_count
	filled_width = min(
		_PARALLEL_PROGRESS_BAR_WIDTH,
		int(completion_fraction * _PARALLEL_PROGRESS_BAR_WIDTH),
	)
	return (
		"#" * filled_width
		+ "-" * (_PARALLEL_PROGRESS_BAR_WIDTH - filled_width)
	)


def _report_trajectory_completion(
	*,
	task: _ABBA4TrajectoryTask,
	runtime_seconds: float,
	completed_run_count: int,
	total_run_count: int,
	particle_count: int,
	study_started: float,
) -> None:
	"""Print one parent-owned completion record and campaign ETA."""
	elapsed_seconds = perf_counter() - study_started
	remaining_seconds = (
		elapsed_seconds
		* (total_run_count - completed_run_count)
		/ completed_run_count
	)
	progress_bar = _trajectory_progress_bar(
		completed_run_count,
		total_run_count,
	)
	variant = _ABBA4_CONFIGURATION_VARIANTS_BY_KEY[task.variant_key]
	print(
		f"[{completed_run_count}/{total_run_count}] Completed in "
		f"{runtime_seconds:.2f} s; particle "
		f"{task.particle + 1}/{particle_count}: {variant.label}; "
		f"progress [{progress_bar}]; "
		f"total {completed_run_count / total_run_count:.1%}; "
		f"elapsed {elapsed_seconds:.1f} s; ETA {remaining_seconds:.1f} s.",
		file=sys.stderr,
		flush=True,
	)


def _report_parallel_progress(
	*,
	completed_run_count: int,
	total_run_count: int,
	unfinished_run_count: int,
	worker_count: int,
	study_started: float,
) -> None:
	"""Print a periodic parent-owned heartbeat for a parallel campaign."""
	elapsed_seconds = perf_counter() - study_started
	completion_fraction = completed_run_count / total_run_count
	progress_bar = _trajectory_progress_bar(
		completed_run_count,
		total_run_count,
	)
	eta_text = (
		"ETA after first completion"
		if completed_run_count == 0
		else (
			f"ETA {elapsed_seconds * unfinished_run_count / completed_run_count:.1f} s"
		)
	)
	print(
		f"Parallel progress [{progress_bar}] "
		f"{completed_run_count}/{total_run_count} "
		f"({completion_fraction:.1%}); unfinished {unfinished_run_count}; "
		f"workers {worker_count}; elapsed {elapsed_seconds:.1f} s; {eta_text}.",
		file=sys.stderr,
		flush=True,
	)


def _run_sequential_trajectories(
	dynamics: GuidingCenterDynamics,
	particle_configurations: tuple[GCInitialConfiguration, ...],
	request: SimulationRequest,
	config: ABBA4ConfigurationComparisonConfig,
	*,
	study_started: float,
) -> tuple[dict[str, dict[int, Solution]], dict[str, np.ndarray]]:
	"""Run the compatibility path with one process and per-method progress."""
	problems = tuple(
		InitialValueProblem(dynamics, particle_configuration)
		for particle_configuration in particle_configurations
	)
	methods = {
		variant.key: _method_for_variant(variant, config)
		for variant in ABBA4_CONFIGURATION_VARIANTS
	}
	solution_rows, runtime_rows = _empty_trajectory_rows(config.particle_count)
	tasks = _sequential_task_schedule(config.particle_count)
	for completed_before, task in enumerate(tasks):
		variant = _ABBA4_CONFIGURATION_VARIANTS_BY_KEY[task.variant_key]
		if config.progress:
			print(
				f"[{completed_before + 1}/{len(tasks)}] Starting particle "
				f"{task.particle + 1}/{config.particle_count}: "
				f"{variant.label}",
				file=sys.stderr,
				flush=True,
			)
		started = perf_counter()
		solution = simulate(
			problems[task.particle],
			methods[task.variant_key],
			request,
		)
		_generalized_energy_history(dynamics, variant, solution)
		runtime_seconds = perf_counter() - started
		runtime_rows[task.variant_key][task.particle] = runtime_seconds
		solution_rows[task.variant_key][task.particle] = solution
		if config.progress:
			_report_trajectory_completion(
				task=task,
				runtime_seconds=runtime_seconds,
				completed_run_count=completed_before + 1,
				total_run_count=len(tasks),
				particle_count=config.particle_count,
				study_started=study_started,
			)
	return solution_rows, runtime_rows


def _run_parallel_trajectories(
	potential: Potential,
	particle_configurations: tuple[GCInitialConfiguration, ...],
	config: ABBA4ConfigurationComparisonConfig,
	*,
	study_started: float,
) -> tuple[dict[str, dict[int, Solution]], dict[str, np.ndarray]]:
	"""Run complete trajectories in spawn-safe, single-threaded workers."""
	tasks = _parallel_task_schedule(config.particle_count)
	worker_count = min(config.worker_count, len(tasks))
	particle_coordinates: list[tuple[float, float]] = []
	for particle_configuration in particle_configurations:
		initial_state = particle_configuration.initial_state
		assert initial_state is not None
		x, y = particle_configuration.positions(initial_state)
		particle_coordinates.append((float(x[0]), float(y[0])))

	solution_rows, runtime_rows = _empty_trajectory_rows(config.particle_count)
	with ProcessPoolExecutor(
		max_workers=worker_count,
		mp_context=get_context("spawn"),
		initializer=_initialize_abba4_worker,
		initargs=(
			_worker_potential_payload(potential),
			tuple(particle_coordinates),
			config,
		),
	) as executor:
		future_tasks: dict[
			Future[_ABBA4TrajectoryPayload],
			_ABBA4TrajectoryTask,
		] = {}
		try:
			for task in tasks:
				future_tasks[
					executor.submit(_run_abba4_trajectory_task, task)
				] = task
			pending_futures = set(future_tasks)
			completed_run_count = 0
			if config.progress:
				_report_parallel_progress(
					completed_run_count=completed_run_count,
					total_run_count=len(tasks),
					unfinished_run_count=len(pending_futures),
					worker_count=worker_count,
					study_started=study_started,
				)
			while pending_futures:
				completed_futures, pending_futures = wait(
					pending_futures,
					timeout=_PARALLEL_PROGRESS_HEARTBEAT_SECONDS,
					return_when=FIRST_COMPLETED,
				)
				if not completed_futures:
					if config.progress:
						_report_parallel_progress(
							completed_run_count=completed_run_count,
							total_run_count=len(tasks),
							unfinished_run_count=len(pending_futures),
							worker_count=worker_count,
							study_started=study_started,
						)
					continue
				for future in completed_futures:
					task = future_tasks[future]
					variant = _ABBA4_CONFIGURATION_VARIANTS_BY_KEY[
						task.variant_key
					]
					try:
						payload = future.result()
					except Exception as exc:
						raise RuntimeError(
							"ABBA4 comparison failed for particle "
							f"{task.particle + 1}/{config.particle_count}, "
							f"configuration {task.variant_key} ({variant.label})."
						) from exc
					if (
						payload.variant_key != task.variant_key
						or payload.particle != task.particle
					):
						raise RuntimeError(
							"An ABBA4 worker returned a trajectory for the wrong task."
						)
					solution_rows[task.variant_key][task.particle] = Solution(
						t=payload.times,
						states=payload.states,
						source=particle_configurations[task.particle],
						diagnostics=payload.diagnostics,
					)
					runtime_rows[task.variant_key][task.particle] = (
						payload.runtime_seconds
					)
					completed_run_count += 1
					if config.progress:
						_report_trajectory_completion(
							task=task,
							runtime_seconds=payload.runtime_seconds,
							completed_run_count=completed_run_count,
							total_run_count=len(tasks),
							particle_count=config.particle_count,
							study_started=study_started,
						)
		except BaseException:
			for pending in future_tasks:
				pending.cancel()
			raise
	return solution_rows, runtime_rows


def run_abba4_configuration_comparison(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	reference: StoredReferenceTrajectory,
	*,
	config: ABBA4ConfigurationComparisonConfig,
) -> ABBA4ConfigurationComparisonResult:
	"""Run sixteen configurations for each configured initial condition."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(reference, StoredReferenceTrajectory):
		raise TypeError("`reference` must be a StoredReferenceTrajectory.")
	if not isinstance(config, ABBA4ConfigurationComparisonConfig):
		raise TypeError("`config` must be ABBA4ConfigurationComparisonConfig.")
	initial_state = initial_configuration.initial_state
	if initial_state is None or initial_configuration.layout.particle_count(
		initial_state
	) != config.particle_count:
		raise ValueError(
			"The initial configuration must contain exactly "
			f"particle_count={config.particle_count} initial conditions."
		)

	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.integration_step,
		sample_count=config.output_sample_count,
	)
	reference_indices = _validated_reference_samples(
		potential,
		dynamics,
		initial_configuration,
		reference,
		config,
		request,
	)
	initial_x, initial_y = initial_configuration.positions(initial_state)
	particle_configurations = tuple(
		GCInitialConfiguration.from_components(
			x=np.asarray([initial_x[particle]], dtype=float),
			y=np.asarray([initial_y[particle]], dtype=float),
		)
		for particle in range(config.particle_count)
	)
	total_run_count = len(ABBA4_CONFIGURATION_VARIANTS) * config.particle_count
	effective_worker_count = min(config.worker_count, total_run_count)
	study_started = perf_counter()
	if config.progress:
		worker_label = (
			"worker process"
			if effective_worker_count == 1
			else "worker processes"
		)
		print(
			"ABBA4 comparison: starting "
			f"{total_run_count} independent integrations "
			f"({config.step_count} steps each) with "
			f"{effective_worker_count} {worker_label}.",
			file=sys.stderr,
			flush=True,
		)
	if config.worker_count == 1:
		solution_rows, runtime_rows = _run_sequential_trajectories(
			dynamics,
			particle_configurations,
			request,
			config,
			study_started=study_started,
		)
	else:
		solution_rows, runtime_rows = _run_parallel_trajectories(
			potential,
			particle_configurations,
			config,
			study_started=study_started,
		)
	if config.progress:
		print(
			f"ABBA4 comparison: all integrations completed in "
			f"{perf_counter() - study_started:.1f} s.",
			file=sys.stderr,
			flush=True,
		)

	return ABBA4ConfigurationComparisonResult(
		potential=potential,
		dynamics=dynamics,
		initial_configuration=initial_configuration,
		reference=reference,
		config=config,
		reference_sample_indices=reference_indices,
		solutions={
			key: tuple(
				solution_rows[key][particle]
				for particle in range(config.particle_count)
			)
			for key in ABBA4_CONFIGURATION_KEYS
		},
		runtimes=runtime_rows,
	)


__all__ = [
	"ABBA4_CONFIGURATION_KEYS",
	"ABBA4_CONFIGURATION_LABELS",
	"ABBA4_CONFIGURATION_PARTICLE_COUNT",
	"ABBA4_CONFIGURATION_VARIANTS",
	"ABBA4ConfigurationComparisonConfig",
	"ABBA4ConfigurationComparisonResult",
	"ABBA4ConfigurationComparisonSummary",
	"ABBA4ConfigurationMethod",
	"ABBA4ConfigurationVariant",
	"run_abba4_configuration_comparison",
]
