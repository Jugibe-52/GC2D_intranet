"""Reusable study of nonlinear iterations in Hairer-projected BM4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

import numpy as np

from diagnostics import (
	ImplicitBM4IterationObserver,
	ImplicitBM4IterationOutputBlock,
	ImplicitBM4IterationRecord,
)
from dynamics import GuidingCenterDynamics
from potential import Potential
from simulation import (
	BM4Implicit1,
	BM4Implicit2,
	InitialConfiguration,
	InitialValueProblem,
	NonlinearSolver,
	SimulationRequest,
	Solution,
	simulate,
)

from ._validation import nonnegative_finite, positive_finite, positive_integer


BM4IterationFormulation: TypeAlias = Literal["implicit_1", "implicit_2"]
BM4_ITERATION_FORMULATIONS: tuple[BM4IterationFormulation, ...] = (
	"implicit_1",
	"implicit_2",
)


def _readonly_float_array(value: np.ndarray) -> np.ndarray:
	"""Own and freeze one floating-point result series."""
	result = np.asarray(value, dtype=float).copy()
	result.setflags(write=False)
	return result


def _readonly_int_array(value: np.ndarray) -> np.ndarray:
	"""Own and freeze one integer result series."""
	result = np.asarray(value, dtype=int).copy()
	result.setflags(write=False)
	return result


@dataclass(frozen=True, slots=True)
class BM4ImplicitIterationStudyConfig:
	"""Physical, numerical, and observer controls for one BM4 iteration run."""

	formulation: BM4IterationFormulation = "implicit_1"
	rho: float = 0.3
	coupling_frequency: float = float(np.pi / 8.0)
	t_span: tuple[float, float] = (0.0, 1.0)
	max_step: float = 0.01
	sample_count: int = 101
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	nonlinear_solver: NonlinearSolver = "newton"
	observer_sample_every: int = 1
	observer_chunk_size: int = 256
	block_name: str = "implicit_bm4_iterations"
	progress: bool = False
	verbose_observer: bool = False

	def __post_init__(self) -> None:
		"""Normalize all controls that affect the reproduced experiment."""
		if self.formulation not in BM4_ITERATION_FORMULATIONS:
			raise ValueError("Unknown implicit BM4 iteration-study formulation.")
		if self.nonlinear_solver not in ("newton", "broyden"):
			raise ValueError("Unknown nonlinear solver for the BM4 iteration study.")
		span = np.asarray(self.t_span, dtype=float)
		if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
			raise ValueError("`t_span` must contain two finite, increasing times.")
		object.__setattr__(self, "t_span", (float(span[0]), float(span[1])))
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		object.__setattr__(
			self,
			"coupling_frequency",
			nonnegative_finite(self.coupling_frequency, "coupling_frequency"),
		)
		for name in (
			"max_step",
			"newton_absolute_tolerance",
			"newton_relative_tolerance",
			"newton_jacobian_relative_step",
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
class BM4ImplicitIterationStudyResult:
	"""Trajectory and per-step nonlinear-solver metrics from projected BM4."""

	config: BM4ImplicitIterationStudyConfig
	dynamics: GuidingCenterDynamics
	solution: Solution
	records: tuple[ImplicitBM4IterationRecord, ...]
	output_blocks: tuple[ImplicitBM4IterationOutputBlock, ...]
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
		"""Return final infinity-norm residuals for observed solves."""
		return _readonly_float_array(
			np.asarray([record.newton_residual_norm for record in self.records])
		)

	@property
	def residual_to_tolerance_ratios(self) -> np.ndarray:
		"""Return final residuals normalized by effective tolerance."""
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
		"""Print nonlinear-work and convergence statistics."""
		iterations = self.iteration_counts
		ratios = self.residual_to_tolerance_ratios
		print(
			f"{self.solution.diagnostics['projection_solver_formulation']}: "
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


def run_bm4_implicit_iteration_study(
	potential: Potential,
	initial_configuration: InitialConfiguration,
	*,
	notebook_path: str | Path,
	config: BM4ImplicitIterationStudyConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> BM4ImplicitIterationStudyResult:
	"""Run projected BM4 while recording every selected nonlinear solve."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, InitialConfiguration):
		raise TypeError(
			"`initial_configuration` must implement InitialConfiguration."
		)
	if not isinstance(config, BM4ImplicitIterationStudyConfig):
		raise TypeError("`config` must be a BM4ImplicitIterationStudyConfig.")

	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.max_step,
		sample_count=config.sample_count,
	)
	method_type = BM4Implicit1 if config.formulation == "implicit_1" else BM4Implicit2
	with ImplicitBM4IterationObserver(
		notebook_path=notebook_path,
		project_root=project_root,
		block_name=config.block_name,
		sample_every=config.observer_sample_every,
		chunk_size=config.observer_chunk_size,
		verbose=config.verbose_observer,
		metadata={
			"study_config": asdict(config),
			"bm4_projection_scope": "one_complete_twelve_stage_cycle",
			**dict(metadata or {}),
		},
	) as observer:
		started = perf_counter()
		solution = simulate(
			problem,
			method_type(
				coupling_frequency=config.coupling_frequency,
				newton_absolute_tolerance=config.newton_absolute_tolerance,
				newton_relative_tolerance=config.newton_relative_tolerance,
				newton_max_iterations=config.newton_max_iterations,
				newton_jacobian_relative_step=(
					config.newton_jacobian_relative_step
				),
				nonlinear_solver=config.nonlinear_solver,
				progress=config.progress,
				step_observer=observer,
			),
			request,
		)
		runtime_seconds = perf_counter() - started

	return BM4ImplicitIterationStudyResult(
		config=config,
		dynamics=dynamics,
		solution=solution,
		records=observer.records,
		output_blocks=observer.output_blocks,
		output_directory=observer.output_directory,
		runtime_seconds=runtime_seconds,
	)


__all__ = [
	"BM4_ITERATION_FORMULATIONS",
	"BM4ImplicitIterationStudyConfig",
	"BM4ImplicitIterationStudyResult",
	"BM4IterationFormulation",
	"run_bm4_implicit_iteration_study",
]
