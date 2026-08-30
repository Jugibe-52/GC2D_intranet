"""Sixteen-configuration ABBA4 comparison on ten one-particle trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from time import perf_counter
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np

from diagnostics import StoredReferenceTrajectory
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
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


@dataclass(frozen=True, slots=True)
class ABBA4ConfigurationComparisonConfig:
	"""Physical interval, fixed grid, and common nonlinear controls."""

	t_span: tuple[float, float] = (0.0, 4.0)
	integration_step: float = 0.0025
	save_interval: float = 0.01
	rho: float = 0.0
	absolute_tolerance: float = 1e-14
	relative_tolerance: float = 1e-13
	max_iterations: int = 40
	progress: bool = False

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
	"""The five requested metrics plus the stable configuration identity."""

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


def _readonly_runtime_array(value: np.ndarray) -> np.ndarray:
	"""Own and freeze ten positive wall-clock measurements."""
	array = np.array(value, dtype=float, copy=True)
	if array.shape != (ABBA4_CONFIGURATION_PARTICLE_COUNT,):
		raise ValueError("Every configuration must contain ten runtime samples.")
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
	"""Aligned one-particle solutions for all sixteen configurations."""

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
		if self.initial_configuration.layout.particle_count(initial_state) != (
			ABBA4_CONFIGURATION_PARTICLE_COUNT
		):
			raise ValueError("The comparison requires exactly ten initial conditions.")
		initial_x, initial_y = self.initial_configuration.positions(initial_state)
		common_times: np.ndarray | None = None
		frozen_solutions: dict[str, tuple[Solution, ...]] = {}
		frozen_runtimes: dict[str, np.ndarray] = {}
		for variant in ABBA4_CONFIGURATION_VARIANTS:
			trajectory_solutions = tuple(self.solutions[variant.key])
			if len(trajectory_solutions) != ABBA4_CONFIGURATION_PARTICLE_COUNT:
				raise ValueError("Every configuration must contain ten solutions.")
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
					raise ValueError("All 160 trajectories must share one saved-time grid.")
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
				self.runtimes[variant.key]
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
		"""Return direct Euclidean errors with shape ``(10, saved_times)``."""
		reference = self.reference.states[:, self.reference_sample_indices]
		distances = np.empty(
			(ABBA4_CONFIGURATION_PARTICLE_COUNT, self.times.size),
			dtype=float,
		)
		for particle, solution in enumerate(self.solutions[key]):
			delta_x = solution.states[0] - reference[particle]
			delta_y = solution.states[1] - reference[
				ABBA4_CONFIGURATION_PARTICLE_COUNT + particle
			]
			distances[particle] = np.hypot(delta_x, delta_y)
		return distances

	def _relative_energy_errors(
		self,
		variant: ABBA4ConfigurationVariant,
	) -> np.ndarray:
		"""Return normalized absolute generalized-energy errors for ten paths."""
		values = np.empty(
			(ABBA4_CONFIGURATION_PARTICLE_COUNT, self.times.size),
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
		"""Return exactly sixteen rows containing the five requested metrics."""
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


def _alternating_particle_order() -> tuple[int, ...]:
	"""Interleave low and high indices to avoid monotone trajectory ordering."""
	order: list[int] = []
	left = 0
	right = ABBA4_CONFIGURATION_PARTICLE_COUNT - 1
	while left <= right:
		order.append(left)
		left += 1
		if left <= right:
			order.append(right)
			right -= 1
	return tuple(order)


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


def run_abba4_configuration_comparison(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	reference: StoredReferenceTrajectory,
	*,
	config: ABBA4ConfigurationComparisonConfig,
) -> ABBA4ConfigurationComparisonResult:
	"""Run sixteen configurations for ten separate initial conditions."""
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
	) != ABBA4_CONFIGURATION_PARTICLE_COUNT:
		raise ValueError("The comparison requires exactly ten initial conditions.")

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
		for particle in range(ABBA4_CONFIGURATION_PARTICLE_COUNT)
	)
	problems = tuple(
		InitialValueProblem(dynamics, particle_configuration)
		for particle_configuration in particle_configurations
	)
	methods = {
		variant.key: _method_for_variant(variant, config)
		for variant in ABBA4_CONFIGURATION_VARIANTS
	}
	solution_rows: dict[str, dict[int, Solution]] = {
		key: {} for key in ABBA4_CONFIGURATION_KEYS
	}
	runtime_rows: dict[str, np.ndarray] = {
		key: np.empty(ABBA4_CONFIGURATION_PARTICLE_COUNT, dtype=float)
		for key in ABBA4_CONFIGURATION_KEYS
	}
	for schedule_index, particle in enumerate(_alternating_particle_order()):
		variant_order = (
			ABBA4_CONFIGURATION_VARIANTS
			if schedule_index % 2 == 0
			else tuple(reversed(ABBA4_CONFIGURATION_VARIANTS))
		)
		for variant in variant_order:
			started = perf_counter()
			solution = simulate(problems[particle], methods[variant.key], request)
			_generalized_energy_history(dynamics, variant, solution)
			runtime_rows[variant.key][particle] = perf_counter() - started
			solution_rows[variant.key][particle] = solution

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
				for particle in range(ABBA4_CONFIGURATION_PARTICLE_COUNT)
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
