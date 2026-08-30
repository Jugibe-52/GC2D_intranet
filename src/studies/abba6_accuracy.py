"""Reference accuracy and refinement study for sixth-order ABBA."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, ClassVar, Mapping

import numpy as np

from diagnostics import StoredReferenceTrajectory
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ABBA6Implicit,
	InitialValueProblem,
	SimulationRequest,
	Solution,
	simulate,
)

from ._trajectory_accuracy import (
	TrajectoryAccuracySeries,
	accuracy_series,
	reference_indices_for_times,
	reference_distance_convention,
	validate_reference_identity,
)
from .abba4_implicit_accuracy import (
	ABBA4ImplicitAccuracyConfig,
	ABBA4ImplicitAccuracyOrder,
	ABBA4ImplicitAccuracyResult,
	ABBA4ImplicitAccuracySummary,
)


@dataclass(frozen=True, slots=True)
class ABBA6AccuracyConfig(ABBA4ImplicitAccuracyConfig):
	"""Physical, nonlinear, integration, and sampling controls for ABBA6Implicit."""


@dataclass(frozen=True, slots=True)
class ABBA6AccuracySummary(ABBA4ImplicitAccuracySummary):
	"""Accuracy, nonlinear work, and runtime for one ABBA6Implicit step size."""


@dataclass(frozen=True, slots=True)
class ABBA6AccuracyOrder(ABBA4ImplicitAccuracyOrder):
	"""Observed ABBA6Implicit error gains and orders between adjacent nested steps."""


@dataclass(frozen=True, slots=True)
class ABBA6AccuracyResult(ABBA4ImplicitAccuracyResult):
	"""Aligned ABBA6Implicit trajectories and errors across one nested refinement."""

	method_name: ClassVar[str] = "ABBA6Implicit"
	summary_type: ClassVar[type[ABBA4ImplicitAccuracySummary]] = ABBA6AccuracySummary
	order_type: ClassVar[type[ABBA4ImplicitAccuracyOrder]] = ABBA6AccuracyOrder


def run_abba6_accuracy_study(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	reference: StoredReferenceTrajectory,
	*,
	config: ABBA6AccuracyConfig,
	potential_metadata: Mapping[str, Any],
	initial_condition_metadata: Mapping[str, Any],
) -> ABBA6AccuracyResult:
	"""Run ABBA6Implicit on nested steps and compare saved states to one reference."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(reference, StoredReferenceTrajectory):
		raise TypeError("`reference` must be a StoredReferenceTrajectory.")
	if not isinstance(config, ABBA6AccuracyConfig):
		raise TypeError("`config` must be ABBA6AccuracyConfig.")
	validate_reference_identity(
		potential,
		initial_configuration,
		reference,
		config,
		potential_metadata=potential_metadata,
		initial_condition_metadata=initial_condition_metadata,
	)
	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	solutions: dict[float, Solution] = {}
	series: dict[float, TrajectoryAccuracySeries] = {}
	runtimes: dict[float, float] = {}
	reference_indices: np.ndarray | None = None
	distance_convention = reference_distance_convention(reference)
	for step in config.integration_steps:
		request = SimulationRequest.uniform(
			t_span=config.t_span,
			max_step=step,
			sample_count=config.output_sample_count,
		)
		started = perf_counter()
		solution = simulate(
			problem,
			ABBA6Implicit(
				newton_absolute_tolerance=config.absolute_tolerance,
				newton_relative_tolerance=config.relative_tolerance,
				newton_max_iterations=config.max_iterations,
				nonlinear_solver=config.nonlinear_solver,
				progress=config.progress,
			),
			request,
		)
		runtimes[step] = perf_counter() - started
		solutions[step] = solution
		indices = reference_indices_for_times(reference, solution.t)
		if reference_indices is None:
			reference_indices = indices
		elif not np.array_equal(indices, reference_indices):
			raise ValueError("ABBA6Implicit refinements do not share reference sample indices.")
		series[step] = accuracy_series(
			"ABBA6Implicit",
			solution.states,
			reference.states[:, indices],
			period=float(potential.grid.period),
			distance_convention=distance_convention,
		)
	assert reference_indices is not None
	return ABBA6AccuracyResult(
		potential=potential,
		dynamics=dynamics,
		initial_configuration=initial_configuration,
		reference=reference,
		config=config,
		reference_sample_indices=reference_indices,
		solutions=solutions,
		series=series,
		runtimes=runtimes,
	)


__all__ = [
	"ABBA6AccuracyConfig",
	"ABBA6AccuracyOrder",
	"ABBA6AccuracyResult",
	"ABBA6AccuracySummary",
	"run_abba6_accuracy_study",
]
