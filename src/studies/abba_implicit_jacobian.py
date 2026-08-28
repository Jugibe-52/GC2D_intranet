"""Reusable local-Jacobian study for implicit ABBA guiding-centre steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

import numpy as np

from diagnostics.abba_jacobian import (
	ImplicitABBAJacobianMethod,
	ImplicitABBAJacobianObserver,
	ImplicitABBAJacobianOutputBlock,
	ImplicitABBAJacobianRecord,
	ImplicitABBAJacobianSample,
)
from dynamics import GuidingCenterDynamics
from potential import Potential
from simulation import (
	ImplicitABBA1,
	ImplicitABBA2,
	InitialConfiguration,
	InitialValueProblem,
	SimulationRequest,
	Solution,
	simulate,
)

from ._validation import nonnegative_finite, positive_finite, positive_integer


ABBAJacobianFormulation: TypeAlias = Literal["implicit_1", "implicit_2"]
ABBA_JACOBIAN_FORMULATIONS: tuple[ABBAJacobianFormulation, ...] = (
	"implicit_1",
	"implicit_2",
)


def _readonly_float_array(value: np.ndarray) -> np.ndarray:
	"""Own and freeze one floating-point study result array."""
	result = np.asarray(value, dtype=float).copy()
	result.setflags(write=False)
	return result


@dataclass(frozen=True, slots=True)
class ImplicitABBAJacobianParticleStepSeries:
	"""Electric-field context and finite physical increments for one particle.

	Planar arrays use component-major shape ``(2, observed_steps)`` in ``(x, y)``
	order. Electric fields are evaluated at ``(end_times, states_after)`` and
	``state_increments`` contains the finite difference
	``states_after - states_before`` for each observed complete step.
	``state_increment_angles`` is the oriented direction
	``atan2(delta_y, delta_x)`` in radians on ``[-pi, pi]``; it is undefined
	(``NaN``) for an exactly zero increment.
	"""

	particle_index: int
	start_times: np.ndarray
	end_times: np.ndarray
	durations: np.ndarray
	states_before: np.ndarray
	states_after: np.ndarray
	effective_electric_fields_after: np.ndarray
	state_increments: np.ndarray
	state_increment_norms: np.ndarray
	state_increment_angles: np.ndarray


@dataclass(frozen=True, slots=True)
class ImplicitABBAJacobianStudyConfig:
	"""Physical, numerical, sampling, and classification study parameters."""

	formulation: ABBAJacobianFormulation = "implicit_1"
	rho: float = 0.3
	t_span: tuple[float, float] = (0.0, 1.0)
	max_step: float = 0.01
	sample_count: int = 101
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	jacobian_method: ImplicitABBAJacobianMethod = "implicit_function"
	discriminant_relative_tolerance: float = 1e-10
	observer_sample_every: int = 1
	observer_chunk_size: int = 64
	block_name: str = "implicit_abba_jacobian"
	progress: bool = False
	verbose_observer: bool = False

	def __post_init__(self) -> None:
		"""Normalize every parameter that changes the reproduced experiment."""
		if self.formulation not in ABBA_JACOBIAN_FORMULATIONS:
			raise ValueError("Unknown implicit ABBA Jacobian formulation.")
		if self.jacobian_method not in ("implicit_function", "stage_increment"):
			raise ValueError("Unknown implicit ABBA Jacobian method.")
		span = np.asarray(self.t_span, dtype=float)
		if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
			raise ValueError("`t_span` must contain two finite, increasing times.")
		object.__setattr__(self, "t_span", (float(span[0]), float(span[1])))
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		for name in (
			"max_step",
			"newton_absolute_tolerance",
			"newton_relative_tolerance",
			"discriminant_relative_tolerance",
		):
			object.__setattr__(self, name, positive_finite(getattr(self, name), name))
		for name in (
			"newton_max_iterations",
			"observer_sample_every",
			"observer_chunk_size",
		):
			object.__setattr__(self, name, positive_integer(getattr(self, name), name))
		if (
			isinstance(self.sample_count, (bool, np.bool_))
			or not isinstance(self.sample_count, (int, np.integer))
			or self.sample_count < 2
		):
			raise ValueError("`sample_count` must be an integer of at least two.")
		object.__setattr__(self, "sample_count", int(self.sample_count))
		if not isinstance(self.block_name, str) or not self.block_name:
			raise ValueError("`block_name` must be a non-empty string.")
		object.__setattr__(self, "progress", bool(self.progress))
		object.__setattr__(self, "verbose_observer", bool(self.verbose_observer))


@dataclass(frozen=True, slots=True)
class ImplicitABBAJacobianStudyResult:
	"""Solution and local per-step Jacobian analyses from one implicit ABBA run."""

	config: ImplicitABBAJacobianStudyConfig
	dynamics: GuidingCenterDynamics
	solution: Solution
	samples: tuple[ImplicitABBAJacobianSample, ...]
	records: tuple[ImplicitABBAJacobianRecord, ...]
	output_blocks: tuple[ImplicitABBAJacobianOutputBlock, ...]
	output_directory: Path
	runtime_seconds: float

	@property
	def particle_count(self) -> int:
		"""Return the number of independently analyzed planar particles."""
		return self.solution.source.layout.particle_count(self.solution.states[:, 0])

	def spectral_class_counts(
		self,
		*,
		particle_index: int | None = None,
	) -> Mapping[str, int]:
		"""Count spectral classes globally or for one selected particle."""
		if particle_index is not None and not 0 <= particle_index < self.particle_count:
			raise IndexError("`particle_index` is outside the analyzed particle range.")
		counts = {"hyperbolic": 0, "elliptic": 0, "parabolic": 0}
		for record in self.records:
			if particle_index is None or record.particle_index == particle_index:
				counts[record.spectral_class] += 1
		return MappingProxyType(counts)

	def particle_step_series(
		self,
		*,
		particle_index: int = 0,
	) -> ImplicitABBAJacobianParticleStepSeries:
		"""Evaluate endpoint fields and finite state increments for one particle."""
		if isinstance(particle_index, (bool, np.bool_)) or not isinstance(
			particle_index,
			(int, np.integer),
		):
			raise TypeError("`particle_index` must be an integer.")
		index = int(particle_index)
		if not 0 <= index < self.particle_count:
			raise IndexError("`particle_index` is outside the analyzed particle range.")

		count = self.particle_count
		component_indices = (index, count + index)
		states_before = np.column_stack(
			[
				sample.state_before[list(component_indices)]
				for sample in self.samples
			]
		)
		states_after = np.column_stack(
			[
				sample.state_after[list(component_indices)]
				for sample in self.samples
			]
		)
		start_times = np.asarray([sample.start_time for sample in self.samples])
		end_times = np.asarray([sample.end_time for sample in self.samples])
		durations = np.asarray([sample.duration for sample in self.samples])
		field_x, field_y = self.dynamics.effective_potential.electric_field(
			end_times,
			states_after[0],
			states_after[1],
		)
		electric_fields = np.vstack((field_x, field_y))
		state_increments = states_after - states_before
		state_increment_norms = np.linalg.norm(state_increments, axis=0)
		state_increment_angles = np.arctan2(
			state_increments[1],
			state_increments[0],
		)
		# A zero vector has no direction, even though arctan2(0, 0) returns zero.
		state_increment_angles[state_increment_norms == 0.0] = np.nan
		return ImplicitABBAJacobianParticleStepSeries(
			particle_index=index,
			start_times=_readonly_float_array(start_times),
			end_times=_readonly_float_array(end_times),
			durations=_readonly_float_array(durations),
			states_before=_readonly_float_array(states_before),
			states_after=_readonly_float_array(states_after),
			effective_electric_fields_after=_readonly_float_array(
				electric_fields
			),
			state_increments=_readonly_float_array(state_increments),
			state_increment_norms=_readonly_float_array(state_increment_norms),
			state_increment_angles=_readonly_float_array(state_increment_angles),
		)

	def print_summary(self) -> None:
		"""Print a concise local-Jacobian run summary."""
		counts = self.spectral_class_counts()
		maximum_condition = max(record.condition_number for record in self.records)
		maximum_radius = max(record.spectral_radius for record in self.records)
		print(
			f"{self.solution.diagnostics['projection_solver_formulation']} / "
			f"{self.config.jacobian_method}"
		)
		print(
			f"steps={self.solution.n_steps}, observed={len(self.samples)}, "
			f"particles={self.particle_count}, runtime={self.runtime_seconds:.6f} s"
		)
		print(
			"spectral classes: "
			+ ", ".join(f"{name}={count}" for name, count in counts.items())
		)
		print(f"max condition number={maximum_condition:.8e}")
		print(f"max spectral radius={maximum_radius:.8e}")
		step_series = tuple(
			self.particle_step_series(particle_index=index)
			for index in range(self.particle_count)
		)
		maximum_field = max(
			float(
				np.max(
					np.linalg.norm(
						item.effective_electric_fields_after,
						axis=0,
					)
				)
			)
			for item in step_series
		)
		maximum_increment = max(
			float(np.max(item.state_increment_norms)) for item in step_series
		)
		print(f"max effective electric-field magnitude={maximum_field:.8e}")
		print(f"max finite state increment={maximum_increment:.8e}")
		print(f"output directory: {self.output_directory}")


def run_implicit_abba_jacobian_study(
	potential: Potential,
	initial_configuration: InitialConfiguration,
	*,
	notebook_path: str | Path,
	config: ImplicitABBAJacobianStudyConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> ImplicitABBAJacobianStudyResult:
	"""Run one implicit ABBA trajectory with local Jacobian observations."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, InitialConfiguration):
		raise TypeError(
			"`initial_configuration` must implement InitialConfiguration."
		)
	if not isinstance(config, ImplicitABBAJacobianStudyConfig):
		raise TypeError("`config` must be an ImplicitABBAJacobianStudyConfig.")
	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.max_step,
		sample_count=config.sample_count,
	)
	method_type = ImplicitABBA1 if config.formulation == "implicit_1" else ImplicitABBA2
	with ImplicitABBAJacobianObserver(
		notebook_path=notebook_path,
		project_root=project_root,
		block_name=config.block_name,
		sample_every=config.observer_sample_every,
		chunk_size=config.observer_chunk_size,
		jacobian_method=config.jacobian_method,
		discriminant_relative_tolerance=config.discriminant_relative_tolerance,
		verbose=config.verbose_observer,
		metadata={
			"study_config": asdict(config),
			**dict(metadata or {}),
		},
	) as observer:
		started = perf_counter()
		solution = simulate(
			problem,
			method_type(
				newton_absolute_tolerance=config.newton_absolute_tolerance,
				newton_relative_tolerance=config.newton_relative_tolerance,
				newton_max_iterations=config.newton_max_iterations,
				progress=config.progress,
				step_observer=observer,
			),
			request,
		)
		runtime_seconds = perf_counter() - started

	return ImplicitABBAJacobianStudyResult(
		config=config,
		dynamics=dynamics,
		solution=solution,
		samples=observer.samples,
		records=observer.records,
		output_blocks=observer.output_blocks,
		output_directory=observer.output_directory,
		runtime_seconds=runtime_seconds,
	)


__all__ = [
	"ABBA_JACOBIAN_FORMULATIONS",
	"ABBAJacobianFormulation",
	"ImplicitABBAJacobianParticleStepSeries",
	"ImplicitABBAJacobianStudyConfig",
	"ImplicitABBAJacobianStudyResult",
	"run_implicit_abba_jacobian_study",
]
