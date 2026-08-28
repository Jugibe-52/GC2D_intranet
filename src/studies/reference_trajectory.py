"""Independent high-precision GC trajectory construction and certification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Mapping

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from diagnostics import (
	StoredReferenceTrajectory,
	reference_trajectory_output_directory,
	write_reference_trajectory,
)
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import SimulationRequest

from ._trajectory_distances import (
	DistanceConvention,
	normalized_distance_convention,
	particle_distances,
)
from ._validation import integer_ratio, nonnegative_finite, positive_finite


@dataclass(frozen=True, slots=True)
class HighPrecisionReferenceConfig:
	"""Adaptive DOP853/Radau controls and common saved-time grid."""

	t_span: tuple[float, float] = (0.0, 2.0)
	save_interval: float = 0.05
	rho: float = 0.3
	relative_tolerance: float = 1e-13
	absolute_tolerance: float = 1e-15
	maximum_step: float = 0.005
	audit_relative_tolerance: float = 1e-13
	audit_absolute_tolerance: float = 1e-15
	audit_maximum_step: float = 0.0025
	distance_convention: DistanceConvention = "periodic"

	def __post_init__(self) -> None:
		"""Validate the reference and independent-solver audit controls."""
		span = np.asarray(self.t_span, dtype=float)
		if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
			raise ValueError("`t_span` must contain two finite, increasing times.")
		object.__setattr__(self, "t_span", (float(span[0]), float(span[1])))
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		object.__setattr__(
			self,
			"distance_convention",
			normalized_distance_convention(self.distance_convention),
		)
		for name in (
			"save_interval",
			"relative_tolerance",
			"absolute_tolerance",
			"maximum_step",
			"audit_relative_tolerance",
			"audit_absolute_tolerance",
			"audit_maximum_step",
		):
			object.__setattr__(self, name, positive_finite(getattr(self, name), name))
		integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		)
		if self.audit_maximum_step > self.maximum_step:
			raise ValueError("The Radau audit maximum step cannot exceed the DOP853 step.")
		if self.maximum_step > self.save_interval:
			raise ValueError("The reference maximum step cannot exceed the saved-time interval.")
		if self.audit_relative_tolerance > self.relative_tolerance:
			raise ValueError("The Radau audit relative tolerance cannot be looser than DOP853.")
		if self.audit_absolute_tolerance > self.absolute_tolerance:
			raise ValueError("The Radau audit absolute tolerance cannot be looser than DOP853.")

	@property
	def output_sample_count(self) -> int:
		"""Return the common number of reference samples including endpoints."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		) + 1


@dataclass(frozen=True, slots=True)
class AdaptiveReferenceSolveSummary:
	"""Work and termination information for one adaptive DOP853 solve."""

	method: str
	maximum_step: float
	relative_tolerance: float
	absolute_tolerance: float
	function_evaluations: int
	jacobian_evaluations: int
	lu_decompositions: int
	runtime_seconds: float
	message: str


@dataclass(frozen=True, slots=True)
class ReferenceTrajectoryAuditSummary:
	"""Periodic discrepancy between the fine reference and audit solve."""

	global_rms_distance: float
	maximum_distance: float
	final_rms_distance: float
	final_maximum_distance: float
	maximum_state_component_difference: float


@dataclass(frozen=True, slots=True)
class HighPrecisionReferenceResult:
	"""Certified reference trajectory, audit metrics, and persisted artifact."""

	config: HighPrecisionReferenceConfig
	trajectory: StoredReferenceTrajectory
	reference_solve: AdaptiveReferenceSolveSummary
	audit_solve: AdaptiveReferenceSolveSummary
	audit: ReferenceTrajectoryAuditSummary


def potential_fingerprint(potential: Potential) -> str:
	"""Hash the sampled field, grid, and off-grid interpolated electric field.

	The two phase samples recover the real and imaginary spatial amplitudes via
	the public evaluation API. Applying this helper to the gyroaveraged potential
	identifies the actual interpolated ODE instead of trusting reconstruction
	metadata supplied by a caller. Canonical off-grid probes include the spline
	and electric-field implementation in the fingerprint used for validation.
	"""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	grid = potential.grid
	digest = hashlib.sha256()
	digest.update(b"gc2d-sampled-potential-v1")
	grid_values = np.asarray(
		(
			grid.x0,
			grid.y0,
			grid.dx,
			grid.dy,
			float(grid.nx),
			float(grid.ny),
			grid.period,
			float(potential.interpolation_order),
		),
		dtype=np.float64,
	)
	digest.update(grid_values.tobytes())
	for time in (0.0, np.pi / 2.0):
		sample = np.ascontiguousarray(potential.evaluate(time), dtype=np.float64)
		digest.update(str(sample.shape).encode("ascii"))
		digest.update(sample.tobytes())
	probe_x = grid.xmin + grid.period * np.asarray(
		(0.03125, 0.173, 0.419, 0.731, 0.9375),
		dtype=np.float64,
	)
	probe_y = grid.ymin + grid.period * np.asarray(
		(0.8125, 0.057, 0.563, 0.287, 0.6875),
		dtype=np.float64,
	)
	for time in (0.0, 0.371, 1.234):
		for component in potential.electric_field(time, probe_x, probe_y):
			sample = np.ascontiguousarray(component, dtype=np.float64)
			digest.update(sample.tobytes())
	return digest.hexdigest()


def _potential_grid_metadata(potential: Potential) -> dict[str, Any]:
	"""Return every grid value relevant to the interpolated field identity."""
	grid = potential.grid
	return {
		"xmin": grid.xmin,
		"ymin": grid.ymin,
		"dx": grid.dx,
		"dy": grid.dy,
		"nx": grid.nx,
		"ny": grid.ny,
		"period": grid.period,
		"shape": grid.shape,
		"interpolation_order": potential.interpolation_order,
	}


def _output_times(config: HighPrecisionReferenceConfig) -> np.ndarray:
	"""Use the simulation request grid to match later fixed-step solutions exactly."""
	return SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.save_interval,
		sample_count=config.output_sample_count,
	).output_times


def _solve_adaptive(
	dynamics: GuidingCenterDynamics,
	initial_state: np.ndarray,
	times: np.ndarray,
	*,
	relative_tolerance: float,
	absolute_tolerance: float,
	maximum_step: float,
	method: Literal["DOP853", "Radau"],
) -> tuple[np.ndarray, AdaptiveReferenceSolveSummary]:
	"""Run one adaptive reference or cross-check solve on prescribed times."""
	started = perf_counter()
	result = solve_ivp(
		fun=lambda time, state: dynamics.vector_field(time, state),
		t_span=(float(times[0]), float(times[-1])),
		y0=np.asarray(initial_state, dtype=float),
		method=method,
		t_eval=times,
		rtol=relative_tolerance,
		atol=absolute_tolerance,
		max_step=maximum_step,
		dense_output=False,
		vectorized=False,
	)
	runtime = perf_counter() - started
	if not result.success:
		raise RuntimeError(f"{method} reference integration failed: {result.message}")
	states = np.asarray(result.y, dtype=float)
	if states.shape != (initial_state.size, times.size) or not np.all(
		np.isfinite(states)
	):
		raise ValueError(f"{method} returned an invalid reference state history.")
	# Preserve the exact supplied initial data instead of a round-tripped copy.
	states[:, 0] = initial_state
	return states, AdaptiveReferenceSolveSummary(
		method=method,
		maximum_step=maximum_step,
		relative_tolerance=relative_tolerance,
		absolute_tolerance=absolute_tolerance,
		function_evaluations=int(result.nfev),
		jacobian_evaluations=int(result.njev),
		lu_decompositions=int(result.nlu),
		runtime_seconds=float(runtime),
		message=str(result.message),
	)


def _audit_summary(
	distances: np.ndarray,
	states: np.ndarray,
	audit_states: np.ndarray,
) -> ReferenceTrajectoryAuditSummary:
	"""Reduce the max-step-halving comparison to interpretable scalars."""
	return ReferenceTrajectoryAuditSummary(
		global_rms_distance=float(np.sqrt(np.mean(distances**2))),
		maximum_distance=float(np.max(distances)),
		final_rms_distance=float(np.sqrt(np.mean(distances[:, -1] ** 2))),
		final_maximum_distance=float(np.max(distances[:, -1])),
		maximum_state_component_difference=float(np.max(np.abs(states - audit_states))),
	)


def _reference_explanation(
	*,
	config: HighPrecisionReferenceConfig,
	particle_count: int,
	potential_metadata: Mapping[str, Any],
	initial_condition_metadata: Mapping[str, Any],
	reference_solve: AdaptiveReferenceSolveSummary,
	audit_solve: AdaptiveReferenceSolveSummary,
	audit: ReferenceTrajectoryAuditSummary,
) -> str:
	"""Describe what the artifact represents and how its accuracy was checked."""
	def json_default(value: object) -> object:
		if isinstance(value, np.generic):
			return value.item()
		if isinstance(value, np.ndarray):
			return value.tolist()
		raise TypeError(f"Cannot serialize {type(value).__name__} in the explanation.")

	potential_json = json.dumps(
		dict(potential_metadata),
		indent=2,
		sort_keys=True,
		default=json_default,
	)
	initial_json = json.dumps(
		dict(initial_condition_metadata),
		indent=2,
		sort_keys=True,
		default=json_default,
	)
	distance_description = (
		"minimum-image periodic"
		if config.distance_convention == "periodic"
		else "Euclidean"
	)
	return f"""# High-precision numerical reference trajectory

This directory contains a numerical reference for the same interpolated
guiding-center ordinary differential equation used by the fixed-step methods.
It is not an exact analytical trajectory and it does not validate the physical
guiding-center approximation.

## Construction

- Reference solver: SciPy DOP853 (explicit Runge--Kutta order 8).
- Packed layout: `[x_1, ..., x_N, y_1, ..., y_N]` with N={particle_count}.
- Normalized gyro-radius: {config.rho:.16g}.
- Particle-distance convention: {config.distance_convention}.
- Time interval: {config.t_span}.
- Saved interval: {config.save_interval:.16g}.
- Relative tolerance: {config.relative_tolerance:.16g}.
- Absolute tolerance: {config.absolute_tolerance:.16g}.
- Maximum internal step: {config.maximum_step:.16g}.
- Function evaluations: {reference_solve.function_evaluations}.

Potential reconstruction metadata:

```json
{potential_json}
```

Initial-condition reconstruction metadata:

```json
{initial_json}
```

The saved samples are interpolation outputs of the adaptive DOP853 solve at the
same times later used by every fixed-step comparison method.

## Resolution audit

An independent SciPy Radau collocation solve used maximum step
{config.audit_maximum_step:.16g}, relative tolerance
{config.audit_relative_tolerance:.16g}, and absolute tolerance
{config.audit_absolute_tolerance:.16g}. Radau belongs to a different implicit
solver family. Its maximum step is
{config.audit_maximum_step / config.maximum_step:.6g} times the DOP853 maximum.
Their {distance_description} discrepancy is:

- global RMS distance: {audit.global_rms_distance:.16e};
- maximum distance: {audit.maximum_distance:.16e};
- final RMS distance: {audit.final_rms_distance:.16e};
- final maximum distance: {audit.final_maximum_distance:.16e}.

The reference is suitable only when method errors remain comfortably above
this audit floor. `trajectory.npz` stores both solves and their per-particle
audit distances. `metadata.json` stores every numerical and reproducibility
parameter, a SHA-256 checksum for the arrays, and a fingerprint of the actual
gyroaveraged sampled and off-grid interpolated electric field.
"""


def run_high_precision_reference_trajectory(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	notebook_path: str | Path,
	config: HighPrecisionReferenceConfig,
	potential_metadata: Mapping[str, Any],
	initial_condition_metadata: Mapping[str, Any],
	reference_name: str = "example_trajectory",
	version: str = "v1",
	project_root: str | Path | None = None,
	overwrite: bool = False,
) -> HighPrecisionReferenceResult:
	"""Compute, audit, explain, and persist one versioned GC reference."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(config, HighPrecisionReferenceConfig):
		raise TypeError("`config` must be a HighPrecisionReferenceConfig.")
	if not isinstance(potential_metadata, Mapping) or not potential_metadata:
		raise ValueError("Non-empty potential reproducibility metadata is required.")
	if not isinstance(initial_condition_metadata, Mapping) or not initial_condition_metadata:
		raise ValueError("Non-empty initial-condition metadata is required.")
	initial_state = initial_configuration.initial_state
	if initial_state is None:
		raise ValueError("The reference initial configuration has no state.")

	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	effective_potential = dynamics.effective_potential
	times = _output_times(config)
	audit_states, audit_solve = _solve_adaptive(
		dynamics,
		initial_state,
		times,
		relative_tolerance=config.audit_relative_tolerance,
		absolute_tolerance=config.audit_absolute_tolerance,
		maximum_step=config.audit_maximum_step,
		method="Radau",
	)
	states, reference_solve = _solve_adaptive(
		dynamics,
		initial_state,
		times,
		relative_tolerance=config.relative_tolerance,
		absolute_tolerance=config.absolute_tolerance,
		maximum_step=config.maximum_step,
		method="DOP853",
	)
	audit_distances = particle_distances(
		states,
		audit_states,
		distance_convention=config.distance_convention,
		period=potential.grid.period,
	)
	audit = _audit_summary(audit_distances, states, audit_states)
	particle_count = initial_configuration.layout.particle_count(initial_state)
	output_directory = reference_trajectory_output_directory(
		notebook_path,
		reference_name=reference_name,
		version=version,
		project_root=project_root,
	)
	metadata = {
		"artifact_kind": "high_precision_gc_reference_trajectory",
		"reference_name": reference_name,
		"version": version,
		"solver": "DOP853",
		"audit_solver": "Radau",
		"config": asdict(config),
		"potential": dict(potential_metadata),
		"potential_grid": _potential_grid_metadata(potential),
		"dynamics_fingerprint_algorithm": "gc2d-sampled-potential-v1-sha256",
		"dynamics_fingerprint_sha256": potential_fingerprint(effective_potential),
		"initial_conditions": dict(initial_condition_metadata),
		"particle_count": particle_count,
		"state_layout": "component_major_[x_1..x_N,y_1..y_N]",
		"reference_solve": asdict(reference_solve),
		"audit_solve": asdict(audit_solve),
		"audit": asdict(audit),
		"numpy_version": np.__version__,
		"scipy_version": scipy.__version__,
		"source_notebook": str(notebook_path),
	}
	trajectory = write_reference_trajectory(
		output_directory=output_directory,
		times=times,
		states=states,
		initial_state=initial_state,
		audit_states=audit_states,
		audit_distances=audit_distances,
		metadata=metadata,
		explanation=_reference_explanation(
			config=config,
			particle_count=particle_count,
			potential_metadata=potential_metadata,
			initial_condition_metadata=initial_condition_metadata,
			reference_solve=reference_solve,
			audit_solve=audit_solve,
			audit=audit,
		),
		overwrite=overwrite,
	)
	return HighPrecisionReferenceResult(
		config=config,
		trajectory=trajectory,
		reference_solve=reference_solve,
		audit_solve=audit_solve,
		audit=audit,
	)


__all__ = [
	"AdaptiveReferenceSolveSummary",
	"HighPrecisionReferenceConfig",
	"HighPrecisionReferenceResult",
	"ReferenceTrajectoryAuditSummary",
	"potential_fingerprint",
	"run_high_precision_reference_trajectory",
]
