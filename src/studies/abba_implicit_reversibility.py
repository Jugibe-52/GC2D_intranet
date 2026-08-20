"""Reusable forward/backward tangent experiment for implicit ABBA."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, TypeAlias

import numpy as np

from diagnostics import (
	ImplicitABBAReversibilityObserver,
	ImplicitABBAReversibilitySample,
)
from dynamics import GuidingCenterDynamics
from potential import Potential
from simulation import (
	ABBA4Implicit1,
	ImplicitABBA1,
	ImplicitABBA2,
	InitialConfiguration,
	InitialValueProblem,
	NONLINEAR_SOLVERS,
	NonlinearSolver,
	SimulationRequest,
	Solution,
	simulate,
)

from ._validation import nonnegative_finite, positive_finite, positive_integer


ABBAReversibilityFormulation: TypeAlias = Literal[
	"implicit_1",
	"implicit_2",
	"abba4_implicit_1",
]
ABBA_REVERSIBILITY_FORMULATIONS: tuple[ABBAReversibilityFormulation, ...] = (
	"implicit_1",
	"implicit_2",
	"abba4_implicit_1",
)


@dataclass(frozen=True, slots=True)
class ImplicitABBAReversibilityStudyConfig:
	"""Physical, integration, nonlinear-solver, and sampling parameters."""

	formulation: ABBAReversibilityFormulation = "implicit_1"
	rho: float = 0.3
	t_span: tuple[float, float] = (0.0, 1.0)
	max_step: float = 0.01
	sample_count: int = 101
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	nonlinear_solver: NonlinearSolver = "newton"
	observer_sample_every: int = 1
	progress: bool = False
	verbose_observer: bool = False

	def __post_init__(self) -> None:
		"""Normalize every parameter that changes the reproduced experiment."""
		if self.formulation not in ABBA_REVERSIBILITY_FORMULATIONS:
			raise ValueError("Unknown implicit ABBA reversibility formulation.")
		if self.nonlinear_solver not in NONLINEAR_SOLVERS:
			raise ValueError("Unknown nonlinear solver for the reversibility study.")
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
		for name in ("newton_max_iterations", "observer_sample_every"):
			object.__setattr__(self, name, positive_integer(getattr(self, name), name))
		if (
			isinstance(self.sample_count, (bool, np.bool_))
			or not isinstance(self.sample_count, (int, np.integer))
			or self.sample_count < 2
		):
			raise ValueError("`sample_count` must be an integer of at least two.")
		object.__setattr__(self, "sample_count", int(self.sample_count))
		object.__setattr__(self, "progress", bool(self.progress))
		object.__setattr__(self, "verbose_observer", bool(self.verbose_observer))


@dataclass(frozen=True, slots=True)
class ImplicitABBAReversibilityStudyResult:
	"""Accepted trajectory and all selected forward/backward tangent samples."""

	config: ImplicitABBAReversibilityStudyConfig
	dynamics: GuidingCenterDynamics
	solution: Solution
	samples: tuple[ImplicitABBAReversibilitySample, ...]
	runtime_seconds: float

	def print_summary(self) -> None:
		"""Print maxima of the principal reversibility and increment diagnostics."""
		if not self.samples:
			print("No implicit ABBA reversibility samples were retained.")
			return
		print(
			f"{self.solution.diagnostics['projection_solver_formulation']} / "
			f"{self.config.nonlinear_solver}"
		)
		print(
			f"steps={self.solution.n_steps}, observed={len(self.samples)}, "
			f"runtime={self.runtime_seconds:.6f} s"
		)
		print(
			"max ||J_minus J_plus - I||="
			f"{max(item.jacobian_composition_defect_norm for item in self.samples):.8e}"
		)
		print(
			"max relative Delta closure="
			f"{max(item.normalized_increment_closure for item in self.samples):.8e}"
		)
		print(
			"max ||Psi_minus(Psi_plus(z_n)) - z_n||="
			f"{max(item.backward_state_error_norm for item in self.samples):.8e}"
		)


def run_implicit_abba_reversibility_study(
	potential: Potential,
	initial_configuration: InitialConfiguration,
	*,
	config: ImplicitABBAReversibilityStudyConfig,
) -> ImplicitABBAReversibilityStudyResult:
	"""Run ABBA while independently solving and differentiating every reverse step."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, InitialConfiguration):
		raise TypeError("`initial_configuration` must implement InitialConfiguration.")
	if not isinstance(config, ImplicitABBAReversibilityStudyConfig):
		raise TypeError(
			"`config` must be an ImplicitABBAReversibilityStudyConfig."
		)
	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.max_step,
		sample_count=config.sample_count,
	)
	observer = ImplicitABBAReversibilityObserver(
		newton_absolute_tolerance=config.newton_absolute_tolerance,
		newton_relative_tolerance=config.newton_relative_tolerance,
		newton_max_iterations=config.newton_max_iterations,
		nonlinear_solver=config.nonlinear_solver,
		sample_every=config.observer_sample_every,
		verbose=config.verbose_observer,
	)
	method = (
		ABBA4Implicit1(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			nonlinear_solver=config.nonlinear_solver,
			progress=config.progress,
			step_observer=observer,
		)
		if config.formulation == "abba4_implicit_1"
		else (
			ImplicitABBA1(
				newton_absolute_tolerance=config.newton_absolute_tolerance,
				newton_relative_tolerance=config.newton_relative_tolerance,
				newton_max_iterations=config.newton_max_iterations,
				nonlinear_solver=config.nonlinear_solver,
				progress=config.progress,
				step_observer=observer,
			)
			if config.formulation == "implicit_1"
			else ImplicitABBA2(
				newton_absolute_tolerance=config.newton_absolute_tolerance,
				newton_relative_tolerance=config.newton_relative_tolerance,
				newton_max_iterations=config.newton_max_iterations,
				nonlinear_solver=config.nonlinear_solver,
				progress=config.progress,
				step_observer=observer,
			)
		)
	)
	started = perf_counter()
	solution = simulate(
		problem,
		method,
		request,
	)
	runtime_seconds = perf_counter() - started
	return ImplicitABBAReversibilityStudyResult(
		config=config,
		dynamics=dynamics,
		solution=solution,
		samples=observer.samples,
		runtime_seconds=runtime_seconds,
	)


__all__ = [
	"ABBA_REVERSIBILITY_FORMULATIONS",
	"ABBAReversibilityFormulation",
	"ImplicitABBAReversibilityStudyConfig",
	"ImplicitABBAReversibilityStudyResult",
	"run_implicit_abba_reversibility_study",
]
