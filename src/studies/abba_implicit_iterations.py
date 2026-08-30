"""Reusable study of nonlinear iterations in implicit ABBA integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

import numpy as np

from diagnostics import (
	ImplicitABBAIterationObserver,
	ImplicitABBAIterationOutputBlock,
	ImplicitABBAIterationRecord,
)
from dynamics import GuidingCenterDynamics
from potential import Potential
from simulation import (
	ABBA_PROJECTION_FORMULATIONS,
	ABBA2Implicit,
	InitialConfiguration,
	InitialValueProblem,
	NonlinearSolver,
	ProjectionFormulation,
	SimulationRequest,
	Solution,
	simulate,
)

from ._validation import nonnegative_finite, positive_finite, positive_integer


ABBAIterationFormulation: TypeAlias = ProjectionFormulation
ABBA_ITERATION_FORMULATIONS = ABBA_PROJECTION_FORMULATIONS


def _readonly_float_array(value: np.ndarray) -> np.ndarray:
	"""Own and freeze one floating-point series returned by the study."""
	result = np.asarray(value, dtype=float).copy()
	result.setflags(write=False)
	return result


def _readonly_int_array(value: np.ndarray) -> np.ndarray:
	"""Own and freeze one integer series returned by the study."""
	result = np.asarray(value, dtype=int).copy()
	result.setflags(write=False)
	return result


@dataclass(frozen=True, slots=True)
class ImplicitABBAIterationStudyConfig:
	"""Physical, numerical, and observer controls for one iteration study."""

	formulation: ABBAIterationFormulation = "reduced_multiplier"
	rho: float = 0.3
	t_span: tuple[float, float] = (0.0, 1.0)
	max_step: float = 0.01
	sample_count: int = 101
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	nonlinear_solver: NonlinearSolver = "newton"
	observer_sample_every: int = 1
	observer_chunk_size: int = 256
	block_name: str = "implicit_abba_iterations"
	progress: bool = False
	verbose_observer: bool = False

	def __post_init__(self) -> None:
		"""Normalize all parameters that affect reproducibility."""
		if self.formulation not in ABBA_ITERATION_FORMULATIONS:
			raise ValueError("Unknown implicit ABBA iteration-study formulation.")
		if self.nonlinear_solver not in ("newton", "broyden"):
			raise ValueError("Unknown nonlinear solver for the ABBA iteration study.")
		span = np.asarray(self.t_span, dtype=float)
		if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
			raise ValueError("`t_span` must contain two finite, increasing times.")
		object.__setattr__(self, "t_span", (float(span[0]), float(span[1])))
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		for name in (
			"max_step",
			"newton_absolute_tolerance",
			"newton_relative_tolerance",
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
class ImplicitABBAIterationStudyResult:
	"""Trajectory and per-step nonlinear-solver observations from one run."""

	config: ImplicitABBAIterationStudyConfig
	dynamics: GuidingCenterDynamics
	solution: Solution
	records: tuple[ImplicitABBAIterationRecord, ...]
	output_blocks: tuple[ImplicitABBAIterationOutputBlock, ...]
	output_directory: Path
	runtime_seconds: float

	@property
	def step_indices(self) -> np.ndarray:
		"""Return the observed complete-step indices."""
		return _readonly_int_array(
			np.asarray([record.step_index for record in self.records])
		)

	@property
	def end_times(self) -> np.ndarray:
		"""Return the endpoint time of every observed step."""
		return _readonly_float_array(
			np.asarray([record.end_time for record in self.records])
		)

	@property
	def iteration_counts(self) -> np.ndarray:
		"""Return the nonlinear correction count at every observed step."""
		return _readonly_int_array(
			np.asarray([record.newton_iterations for record in self.records])
		)

	@property
	def residual_norms(self) -> np.ndarray:
		"""Return final infinity-norm residuals for the observed solves."""
		return _readonly_float_array(
			np.asarray([record.newton_residual_norm for record in self.records])
		)

	@property
	def residual_to_tolerance_ratios(self) -> np.ndarray:
		"""Return each final residual divided by its effective tolerance."""
		return _readonly_float_array(
			np.asarray(
				[record.residual_to_tolerance_ratio for record in self.records]
			)
		)

	def iteration_frequencies(self) -> Mapping[int, int]:
		"""Count how many observed steps required each iteration count."""
		values, counts = np.unique(self.iteration_counts, return_counts=True)
		return MappingProxyType(
			{
				int(iterations): int(count)
				for iterations, count in zip(values, counts)
			}
		)

	def print_summary(self) -> None:
		"""Print the main nonlinear-work and convergence statistics."""
		iterations = self.iteration_counts
		ratios = self.residual_to_tolerance_ratios
		print(
			f"{self.solution.diagnostics['projection_formulation']}: "
			f"steps={self.solution.n_steps}, observed={len(self.records)}, "
			f"runtime={self.runtime_seconds:.6f} s"
		)
		print(
			f"{self.config.nonlinear_solver.title()} iterations: "
			f"min={int(np.min(iterations))}, "
			f"mean={float(np.mean(iterations)):.6f}, "
			f"max={int(np.max(iterations))}"
		)
		print(f"iteration frequencies: {dict(self.iteration_frequencies())}")
		print(f"maximum final residual/tolerance={float(np.max(ratios)):.8e}")
		print(f"output directory: {self.output_directory}")


def run_implicit_abba_iteration_study(
	potential: Potential,
	initial_configuration: InitialConfiguration,
	*,
	notebook_path: str | Path,
	config: ImplicitABBAIterationStudyConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> ImplicitABBAIterationStudyResult:
	"""Run one implicit ABBA trajectory while recording nonlinear-solver work."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, InitialConfiguration):
		raise TypeError(
			"`initial_configuration` must implement InitialConfiguration."
		)
	if not isinstance(config, ImplicitABBAIterationStudyConfig):
		raise TypeError("`config` must be an ImplicitABBAIterationStudyConfig.")

	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.max_step,
		sample_count=config.sample_count,
	)
	with ImplicitABBAIterationObserver(
		notebook_path=notebook_path,
		project_root=project_root,
		block_name=config.block_name,
		sample_every=config.observer_sample_every,
		chunk_size=config.observer_chunk_size,
		verbose=config.verbose_observer,
		metadata={
			"study_config": asdict(config),
			**dict(metadata or {}),
		},
	) as observer:
		started = perf_counter()
		solution = simulate(
			problem,
			ABBA2Implicit(
				projection_formulation=config.formulation,
				newton_absolute_tolerance=config.newton_absolute_tolerance,
				newton_relative_tolerance=config.newton_relative_tolerance,
				newton_max_iterations=config.newton_max_iterations,
				nonlinear_solver=config.nonlinear_solver,
				progress=config.progress,
				step_observer=observer,
			),
			request,
		)
		runtime_seconds = perf_counter() - started

	return ImplicitABBAIterationStudyResult(
		config=config,
		dynamics=dynamics,
		solution=solution,
		records=observer.records,
		output_blocks=observer.output_blocks,
		output_directory=observer.output_directory,
		runtime_seconds=runtime_seconds,
	)


__all__ = [
	"ABBA_ITERATION_FORMULATIONS",
	"ABBAIterationFormulation",
	"ImplicitABBAIterationStudyConfig",
	"ImplicitABBAIterationStudyResult",
	"run_implicit_abba_iteration_study",
]
