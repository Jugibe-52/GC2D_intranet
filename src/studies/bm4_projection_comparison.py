"""Audited accuracy, serial timings and multipliers for the two BM4 methods."""

from dataclasses import dataclass, replace
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from dynamics import GuidingCenterDynamics
from diagnostics import StoredReferenceTrajectory
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	BM4Implicit, BM4Midpoint, ImplicitBM4IntegrationStep, InitialValueProblem,
	IntegrationStep, NumericalMethod, SimulationRequest, simulate,
)
from ._gauss_legendre4_common import AdaptiveReference, build_adaptive_reference
from ._trajectory_accuracy import validate_reference_identity
from ._trajectory_distances import particle_distances
from .reference_trajectory import potential_fingerprint
from .three_method_newton_comparison import ThreeMethodNewtonComparisonConfig


BM4_PROJECTION_METHODS = ("BM4Midpoint", "BM4Implicit")


def summarize_bm4_particles(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
	"""Summarize each trajectory separately, without allocating joint runtimes."""
	times = arrays["times"]
	n = arrays["initial_state"].size // 2
	mu = arrays["BM4Implicit.mu"].reshape(2, n, -1)
	def rms(values: np.ndarray, grid: np.ndarray) -> float:
		return float(np.sqrt(np.trapz(values**2, grid) / (grid[-1] - grid[0])))
	rows = []
	for j in range(n):
		row: dict[str, Any] = {"particle": f"P{j+1}", "methods": {}}
		for name in BM4_PROJECTION_METHODS:
			distance = arrays[f"{name}.distance"][j]
			energy = arrays[f"{name}.energy_error"][j]
			row["methods"][name] = {
				"trajectory_rms": rms(distance, times),
				"trajectory_final": float(distance[-1]),
				"trajectory_max": float(np.max(distance)),
				"energy_rms": rms(energy, times),
				"energy_max": float(np.max(np.abs(energy))),
			}
		row["positions"] = {"initial": arrays["initial_state"].reshape(2, n)[:, j].tolist()}
		for name in ("DOP853", *BM4_PROJECTION_METHODS):
			row["positions"][name] = arrays[f"{name}.states"].reshape(2, n, -1)[:, j, -1].tolist()
		norm = np.max(np.abs(mu[:, j]), axis=0)
		row["mu"] = {"mean": float(np.mean(norm)), "rms": float(np.sqrt(np.mean(norm**2))),
			"maximum": float(np.max(norm)), "final": float(norm[-1])}
		if "reference.refinement_distance" in arrays:
			distance = arrays["reference.refinement_distance"][j]
			row["reference_refinement"] = {"rms": rms(distance, arrays["reference.full_times"]),
				"maximum": float(np.max(distance))}
		rows.append(row)
	return rows


@dataclass(frozen=True, slots=True)
class BM4ProjectionComparisonConfig(ThreeMethodNewtonComparisonConfig):
	"""Standard 200-cycle comparison, with all controls available to notebooks."""

	t_span: tuple[float, float] = (0.0, 200.0)
	integration_step: float = 0.1
	absolute_tolerance: float = 1e-12
	relative_tolerance: float = 1e-11
	reference_relative_tolerance: float = 1e-10
	reference_absolute_tolerance: float = 1e-12
	reference_maximum_step: float = 0.025
	audit_relative_tolerance: float = 1e-11
	audit_absolute_tolerance: float = 1e-13
	audit_maximum_step: float = 0.0125


def radial_configuration(
	potential: Potential,
	*,
	radius_fractions: tuple[float, ...] = (0.1, 0.25, 0.4),
	angle: float = 0.0,
) -> GCInitialConfiguration:
	"""Place particles along one radius; fractions use the cell half-width."""
	radii = np.asarray(radius_fractions, dtype=float)
	if (radii.ndim != 1 or radii.size == 0 or not np.all(np.isfinite(radii))
		or np.any(radii < 0) or np.any(radii >= 1) or np.any(np.diff(radii) <= 0)
		or not np.isfinite(angle)):
		raise ValueError("Use finite increasing radius fractions in [0,1) and a finite angle.")
	grid = potential.grid
	radii = radii * grid.period / 2.0
	return GCInitialConfiguration.from_components(
		x=grid.xmin + grid.period / 2 + radii * np.cos(angle),
		y=grid.ymin + grid.period / 2 + radii * np.sin(angle),
	)


def _rms(values: np.ndarray, times: np.ndarray) -> float:
	"""Integrate squared particle errors with the shared trapezoidal convention."""
	return float(np.sqrt(np.trapz(np.mean(values**2, axis=0), times) / (times[-1] - times[0])))


def saved_sample_indices(reference: StoredReferenceTrajectory, times: np.ndarray) -> np.ndarray:
	"""Select saved nodes to roundoff, never interpolate or extrapolate."""
	spacing = float(reference.times[1] - reference.times[0])
	indices = np.rint((times - reference.times[0]) / spacing).astype(np.int64)
	if np.any(indices < 0) or np.any(indices >= reference.times.size):
		raise ValueError("Requested times extend beyond the saved reference.")
	tolerance = float(64 * np.finfo(float).eps * max(1., float(np.max(np.abs(times)))))
	if not np.allclose(reference.times[indices], times, rtol=0, atol=tolerance):
		raise ValueError("Requested times must coincide with saved reference samples.")
	return np.asarray(indices, dtype=np.int64)


def audit_saved_reference_refinement(
	potential: Potential, configuration: GCInitialConfiguration,
	reference: StoredReferenceTrajectory, coarse: StoredReferenceTrajectory,
	*, config: BM4ProjectionComparisonConfig,
	potential_metadata: Mapping[str, object], initial_condition_metadata: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
	"""Compare existing DOP853 refinements over the complete standard prefix.

	Both stored artifacts are validated against the actual interpolated ODE.
	A smaller maximum step and tighter tolerances establish the refinement;
	the measured discrepancy remains empirical, not a rigorous error bound.
	"""
	for stored in (reference, coarse):
		stored_span = stored.metadata["config"]["t_span"]
		validate_reference_identity(potential, configuration, stored,
			replace(config, t_span=(float(stored_span[0]), float(stored_span[1]))),
			potential_metadata=potential_metadata, initial_condition_metadata=initial_condition_metadata)
		if stored.metadata.get("solver") != "DOP853":
			raise ValueError("The refinement pair must contain DOP853 trajectories.")
	fine_config, coarse_config = reference.metadata["config"], coarse.metadata["config"]
	for key in ("relative_tolerance", "absolute_tolerance", "maximum_step"):
		if fine_config[key] >= coarse_config[key]:
			raise ValueError("The standard reference must refine step and both tolerances.")
	indices = saved_sample_indices(coarse, reference.times)
	distance = particle_distances(reference.states, coarse.states[:, indices],
		period=potential.grid.period, distance_convention=config.distance_convention)
	arrays = {"reference.full_times": reference.times,
		"reference.full_audit_distance": reference.audit_distances,
		"reference.refinement_distance": distance}
	summary: dict[str, object] = {
		"coarse_directory": str(coarse.paths.directory),
		"coarse_checksum": coarse.metadata["trajectory_sha256"],
		"fine_checksum": reference.metadata["trajectory_sha256"],
		"coarse_solver_settings": coarse_config, "fine_solver_settings": fine_config,
		"t_span": [float(reference.times[0]), float(reference.times[-1])],
		"refinement_rms": _rms(distance, reference.times),
		"refinement_maximum": float(distance.max()),
		"refinement_maximum_per_particle": distance.max(axis=1).tolist(),
		"radau_maximum": float(reference.audit_distances.max()),
		"radau_maximum_per_particle": reference.audit_distances.max(axis=1).tolist(),
		"arithmetic": "IEEE 754 float64 (53-bit significand), distinct from integration error.",
		"interpretation": "Empirical refinement and cross-solver discrepancies, not rigorous bounds. No independent analytic solution is available for this measured interpolated field.",
	}
	return arrays, summary


def run_bm4_projection_comparison(
	potential: Potential,
	configuration: GCInitialConfiguration,
	*,
	config: BM4ProjectionComparisonConfig,
	saved_reference: StoredReferenceTrajectory | None = None,
	potential_metadata: Mapping[str, object] | None = None,
	initial_condition_metadata: Mapping[str, object] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
	"""Run aligned methods, audited references and a separate multiplier replay.

	Timings exclude reference generation and the observer replay. Every repeat
	advances all particles jointly, with one BLAS thread and alternating method
	order. Arrays retain all accepted step diagnostics and saved trajectories.
	"""
	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, configuration)
	request = SimulationRequest.uniform(t_span=config.t_span, max_step=config.integration_step,
		sample_count=config.output_sample_count)
	times = request.output_times
	period = potential.grid.period if config.distance_convention == "periodic" else None
	implicit = BM4Implicit(coupling_frequency=config.coupling_frequency,
		newton_absolute_tolerance=config.absolute_tolerance,
		newton_relative_tolerance=config.relative_tolerance,
		newton_max_iterations=config.max_iterations,
		newton_jacobian_relative_step=config.jacobian_relative_step,
		newton_jacobian_method="analytic")
	methods: dict[str, NumericalMethod] = {
		"BM4Midpoint": BM4Midpoint(coupling_frequency=config.coupling_frequency),
		"BM4Implicit": implicit,
	}
	arrays: dict[str, np.ndarray] = {
		"times": times, "initial_state": problem.initial_state,
		"step_times": np.linspace(*config.t_span, config.step_count + 1)[1:],
		"domain": np.asarray([potential.grid.xmin, potential.grid.ymin, potential.grid.period]),
	}
	def log(message: str) -> None:
		if config.progress:
			print(message, flush=True)

	with threadpool_limits(limits=1):
		if saved_reference is None:
			log("Computing DOP853 reference and independent Radau audit.")
			reference = build_adaptive_reference(dynamics, problem.initial_state, times,
				period=period, distance_convention=config.distance_convention,
				relative_tolerance=config.reference_relative_tolerance,
				absolute_tolerance=config.reference_absolute_tolerance,
				maximum_step=config.reference_maximum_step,
				audit_relative_tolerance=config.audit_relative_tolerance,
				audit_absolute_tolerance=config.audit_absolute_tolerance,
				audit_maximum_step=config.audit_maximum_step)
		else:
			if potential_metadata is None or initial_condition_metadata is None:
				raise ValueError("Saved reference reuse requires explicit physical metadata.")
			validate_reference_identity(potential, configuration, saved_reference, config,
				potential_metadata=potential_metadata, initial_condition_metadata=initial_condition_metadata)
			if saved_reference.metadata.get("solver") != "DOP853" or saved_reference.metadata.get("audit_solver") != "Radau":
				raise ValueError("Expected a saved DOP853 reference with a Radau audit.")
			for field, key in (("reference_relative_tolerance", "relative_tolerance"),
				("reference_absolute_tolerance", "absolute_tolerance"), ("reference_maximum_step", "maximum_step"),
				("audit_relative_tolerance", "audit_relative_tolerance"),
				("audit_absolute_tolerance", "audit_absolute_tolerance"), ("audit_maximum_step", "audit_maximum_step")):
				if getattr(config, field) != saved_reference.metadata["config"][key]:
					raise ValueError(f"Saved reference setting differs: {field}.")
			indices = saved_sample_indices(saved_reference, times)
			reference = AdaptiveReference(times, saved_reference.states[:, indices],
				saved_reference.audit_states[:, indices], saved_reference.audit_distances[:, indices],
				0.0, 0.0, 0, 0)
			log(f"Reusing verified saved DOP853/Radau reference: {saved_reference.paths.directory}")
		arrays["DOP853.states"] = reference.states
		arrays["Radau.states"] = reference.audit_states
		arrays["reference.distance"] = reference.audit_distances
		if saved_reference is None:
			log(f"References completed: DOP853 {reference.dop853_runtime_seconds:.2f}s; Radau {reference.radau_runtime_seconds:.2f}s.")
		for _ in range(config.timing_warmups):
			for name in BM4_PROJECTION_METHODS:
				log(f"Warming up {name}.")
				simulate(problem, methods[name], request)
		runtime_samples: dict[str, list[float]] = {name: [] for name in methods}
		solutions = {}
		for repeat in range(config.timing_repeats):
			order = BM4_PROJECTION_METHODS if repeat % 2 == 0 else BM4_PROJECTION_METHODS[::-1]
			for name in order:
				start = perf_counter()
				solution = simulate(problem, methods[name], request)
				elapsed = perf_counter() - start
				assert solution.diagnostics["step_count"] == config.step_count
				assert np.array_equal(solution.t, times)
				solutions[name] = solution
				runtime_samples[name].append(elapsed)
				log(f"Repeat {repeat + 1}/{config.timing_repeats}: {name} {elapsed:.3f}s.")
		# Requesting BM4 step observations also reconstructs its twelve base
		# stages. Collect mu in an untimed replay so timings remain comparable.
		multipliers: list[np.ndarray] = []
		def observe(event: IntegrationStep) -> None:
			assert isinstance(event, ImplicitBM4IntegrationStep)
			assert len(event.base_stages) == 12
			multipliers.append(event.multiplier.copy())
		log("Capturing signed per-particle multipliers in an untimed replay.")
		replay = simulate(problem, replace(implicit, step_observer=observe), request)
		np.testing.assert_array_equal(replay.states, solutions["BM4Implicit"].states)
		arrays["BM4Implicit.mu"] = np.stack(multipliers, axis=1)

	for name in ("DOP853", "Radau"):
		arrays[f"{name}.H"] = np.asarray(dynamics.hamiltonian(times, arrays[f"{name}.states"]))
	arrays["reference.energy_error"] = arrays["DOP853.H"] - arrays["Radau.H"]
	summary: dict[str, object] = {
		"potential_fingerprint": potential_fingerprint(potential),
		"particle_count": problem.particle_count, "step_count": config.step_count,
		"reference_rms_floor": reference.time_integrated_rms_floor,
		"reference_final_floor": reference.final_rms_floor,
		"reference_energy_rms_floor": _rms(arrays["reference.energy_error"], times),
		"reference_seconds": {"DOP853": reference.dop853_runtime_seconds, "Radau": reference.radau_runtime_seconds},
		"reference_reused": saved_reference is not None,
		"reference_seconds_scope": "Reference integration time incurred by this comparison; zero when reusing saved arrays.",
		"timing_policy": "Serial alternating order; one BLAS thread; references and observer replay excluded.",
		"mu_note": "Only BM4Implicit has a projection multiplier. Midpoint copy separation is a different diagnostic.",
	}
	method_summaries = {}
	for name in BM4_PROJECTION_METHODS:
		solution = solutions[name]
		arrays[f"{name}.states"] = solution.states
		arrays[f"{name}.runtime_samples"] = np.asarray(runtime_samples[name])
		arrays[f"{name}.distance"] = particle_distances(solution.states, reference.states,
			period=period, distance_convention=config.distance_convention)
		arrays[f"{name}.energy_error"] = np.asarray(dynamics.hamiltonian(times, solution.states)) - arrays["DOP853.H"]
		for key, value in solution.diagnostics.items():
			if isinstance(value, np.ndarray):
				arrays[f"{name}.{key}"] = value
		q1, median, q3 = np.quantile(runtime_samples[name], [.25, .5, .75])
		method_summaries[name] = {
			"runtime_median_seconds": float(median), "runtime_q1_seconds": float(q1), "runtime_q3_seconds": float(q3),
			"trajectory_rms": _rms(arrays[f"{name}.distance"], times),
			"trajectory_final_rms": float(np.sqrt(np.mean(arrays[f"{name}.distance"][:, -1]**2))),
			"trajectory_max": float(np.max(arrays[f"{name}.distance"])),
			"energy_rms": _rms(arrays[f"{name}.energy_error"], times),
			"energy_max": float(np.max(np.abs(arrays[f"{name}.energy_error"]))),
		}
	assert solutions["BM4Midpoint"].diagnostics["composition_stage_count"] == 12
	assert solutions["BM4Midpoint"].diagnostics["nonlinear_unknown_dimension"] == 0
	diag = solutions["BM4Implicit"].diagnostics
	assert diag["projection_solver_formulation"] == "bm4_implicit_reduced"
	assert diag["newton_jacobian_method"] == "analytic" and diag["nonlinear_solver"] == "newton"
	mu_norm = np.max(np.abs(arrays["BM4Implicit.mu"]), axis=0)
	np.testing.assert_allclose(mu_norm, np.asarray(diag["projection_multiplier_norms"]), rtol=0, atol=0)
	iterations = np.asarray(diag["nonlinear_iterations"])
	evaluations = np.asarray(diag["residual_evaluations"])
	assert iterations.shape == evaluations.shape == mu_norm.shape == (config.step_count,)
	summary["methods"] = method_summaries
	summary["mu"] = {"mean": float(np.mean(mu_norm)), "rms": float(np.sqrt(np.mean(mu_norm**2))),
		"maximum": float(np.max(mu_norm)), "final": float(mu_norm[-1])}
	summary["nonlinear_work"] = {
		"solves_per_step": 1, "mean_iterations": float(np.mean(iterations)),
		"max_iterations": int(np.max(iterations)), "total_iterations": int(np.sum(iterations)),
		"mean_residual_evaluations": float(np.mean(evaluations)), "total_residual_evaluations": int(np.sum(evaluations)),
		"maximum_residual_to_tolerance": float(np.max(np.asarray(diag["nonlinear_residual_norms"]) / np.asarray(diag["nonlinear_tolerances"]))),
	}
	if any(not np.all(np.isfinite(value)) for value in arrays.values()):
		raise ValueError("Comparison produced non-finite arrays.")
	return arrays, summary
