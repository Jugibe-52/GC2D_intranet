"""Repeat the radial BM4Midpoint study with a cycle-time stream function."""

from studies._validation import require_uncoupled_bm4

from formulations.layout import GCStateLayout

from studies.historical_h5 import historical_h5_hamiltonian

from dataclasses import asdict, dataclass
from hashlib import file_digest
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING
from states.history import States

import numpy as np

from execution.options import ExecutionOptions
from problem import InitialValueProblem
from request import SimulationRequest
from diagnostics.io.persistence import StoredSolution, load_solution, save_solution
from diagnostics.io.storage import ArtifactStore
from dynamics.gc import GuidingCenterDynamics
from execution.execution import Execution

from methods.extended.bm4 import BM4Midpoint
from potential.load import DEFAULT_FIELD_INDICES
from potential import Potential
from simulation.runner import simulate
from studies.dimensional_h5_midpoint import resolve_h5_source


if TYPE_CHECKING:
    from contracts.result import DiagnosticValue
    from typing import Mapping


@dataclass(frozen=True)
class RadialCycleConfig:
    """Explicit physical and numerical inputs; radii are fractions of cell width."""

    radial_fractions: tuple[float, ...]
    cycles: int = 5000
    steps_per_cycle: int = 20
    radial_angle: float = 0.0
    rho: float = 0.3
    coupling_frequency: float = 0.0
    magnetic_field: float = 1.5
    characteristic_length: float = 0.06
    source_selection: tuple[int, ...] = DEFAULT_FIELD_INDICES
    interpolation_order: int = 3
    stream_scale: float = 2 * np.pi

    def __post_init__(self) -> None:
        require_uncoupled_bm4(self.coupling_frequency)
        radii: np.ndarray = np.asarray(self.radial_fractions)
        if (radii.ndim != 1 or radii.size < 2 or not np.isfinite(radii).all()
                or radii[0] != 0 or np.any(np.diff(radii) <= 0) or radii[-1] >= .5):
            raise ValueError("Radii must increase from zero to less than half the cell width.")
        for name in ("cycles", "steps_per_cycle"):
            value: Any = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("rho", "coupling_frequency"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not np.isfinite(self.radial_angle):
            raise ValueError("The radial angle must be finite.")
        if self.stream_scale != 2 * np.pi:
            raise ValueError("This experiment requires the cycle-time factor 2*pi.")


@dataclass(frozen=True)
class PreparedRadialCycle:
    config: RadialCycleConfig
    problem: InitialValueProblem
    request: SimulationRequest
    metadata: dict


def build_radial_cycle(potential: Potential, config: RadialCycleConfig, baseline: dict[str, Any],
                       *, source_sha256: str, baseline_sha256: str) -> PreparedRadialCycle:
    """Scale both field components and reconstruct B's documented initial radius."""
    provenance = baseline["field_provenance"]
    expected = (provenance["B_tesla"], provenance["characteristic_length_m"],
                tuple(provenance["source_field_indices"]), provenance["interpolation_order"],
                baseline["rho"], baseline["coupling_frequency"], baseline["radial_angle_rad"])
    actual = (config.magnetic_field, config.characteristic_length, config.source_selection,
              config.interpolation_order, config.rho, config.coupling_frequency, config.radial_angle)
    if actual != expected or source_sha256 != provenance["source_hdf5_sha256"]:
        raise ValueError("Physical inputs do not match the original radial study.")
    np.testing.assert_array_equal(config.radial_fractions, baseline["radial_fractions"])
    if baseline["particle_ids"] != list(range(1, len(config.radial_fractions) + 1)):
        raise ValueError("Expected the original consecutive radial particle IDs.")
    grid = potential.grid
    old_grid = provenance["grid"]
    np.testing.assert_allclose([grid.x0, grid.y0, grid.period],
                               [old_grid["x0"], old_grid["y0"], old_grid["period"]],
                               rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(potential.frequencies, [1.0], rtol=0, atol=1e-14)
    # Preserve B's documented geometry using its archived period, not a new grid.
    origin: np.ndarray = np.array([old_grid["x0"], old_grid["y0"]])
    length = old_grid["period"]
    radii = np.asarray(config.radial_fractions) * length
    direction: np.ndarray = np.array([np.cos(config.radial_angle), np.sin(config.radial_angle)])
    xy = origin + length / 2 + radii[:, None] * direction
    initial: np.ndarray = GCStateLayout().pack_components(xy[:, 0], xy[:, 1])
    # The forcing cyclic frequencies stay in cycles per normalized time. Only the
    # drift amplitude changes: H = 2*pi*Phi_hat at tau = t/T0.
    stream: Potential = Potential(grid, mean=config.stream_scale * potential.mean,
                       modes=config.stream_scale * potential.modes,
                       frequencies=potential.frequencies,
                       interpolation_order=potential.interpolation_order)
    count = config.cycles * config.steps_per_cycle
    request: SimulationRequest = SimulationRequest(t_span=(0., float(config.cycles)),
                                max_step=1 / config.steps_per_cycle,
                                output_times=np.arange(count + 1) / config.steps_per_cycle)
    metadata: dict[str, Any] = dict(
        study="bm4midpoint_radial_cycle_time", config=asdict(config),
        source_sha256=source_sha256, baseline_metadata_sha256=baseline_sha256,
        baseline_run_id=baseline["run_id"], particle_ids=baseline["particle_ids"],
        colors=[baseline["colours"][str(pid)] for pid in baseline["particle_ids"]],
        initial_positions=xy.tolist(), initial_positions_origin="reconstructed from archived radial geometry",
        cell_origin=origin.tolist(), cell_period=length,
        physical_origin_m=provenance["source_origin_m"],
        time_unit_seconds=provenance["characteristic_period_s"],
        length_unit_m=config.characteristic_length / (2 * np.pi),
        dynamics_stream_function="H = 2*pi*Phi_hat = (2*pi/lambda)^2*(T0/B)*Phi",
        time_axis="tau = t_seconds / T0", method="BM4Midpoint",
        saved_positions="unwrapped", plotted_positions="wrapped and divided by cell width",
        trajectory_accuracy_certified=False,
        execution_note="Current implementation; jointly integrated particles, not the frozen 16-process runtime.",
    )
    return PreparedRadialCycle(config, InitialValueProblem(GuidingCenterDynamics(stream, rho=config.rho), initial, layout=GCStateLayout()),
                               request, metadata)


def prepare_radial_cycle(source: str | Path, baseline_metadata: str | Path,
                         config: RadialCycleConfig) -> PreparedRadialCycle:
    """Verify archived provenance and load the original normalized HDF5 samples."""
    baseline_metadata = Path(baseline_metadata)
    raw: bytes = baseline_metadata.read_bytes()
    import hashlib
    baseline_sha: str = hashlib.sha256(raw).hexdigest()
    marker: Any = json.loads(baseline_metadata.with_name("COMPLETE.json").read_text())
    if marker["sha256"]["metadata.json"] != baseline_sha:
        raise ValueError("Original metadata checksum mismatch.")
    baseline: Any = json.loads(raw)
    source = resolve_h5_source(source)
    with source.open("rb") as handle:
        source_sha: str = file_digest(handle, "sha256").hexdigest()
    if source_sha != baseline["field_provenance"]["source_hdf5_sha256"]:
        raise ValueError("Original HDF5 checksum mismatch.")
    potential: Potential = historical_h5_hamiltonian(
        source, magnetic_field=config.magnetic_field, length=config.characteristic_length,
        indx=config.source_selection, interpolation_order=config.interpolation_order,
        angular_reference=2 * np.pi / baseline["field_provenance"]["characteristic_period_s"],
    )
    return build_radial_cycle(potential, config, baseline, source_sha256=source_sha,
                              baseline_sha256=baseline_sha)


def validate_radial_solution(solution: States, prepared: PreparedRadialCycle) -> None:
    """Require every complete step and the explicit arithmetic projection."""
    c: RadialCycleConfig
    d: Mapping[str, DiagnosticValue]
    c, d = prepared.config, solution.diagnostics
    count = c.cycles * c.steps_per_cycle
    np.testing.assert_array_equal(solution.t, prepared.request.output_times)
    np.testing.assert_array_equal(solution.states[:, 0], prepared.problem.initial_state)
    if (solution.states.shape != (2 * len(c.radial_fractions), count + 1)
            or not np.isfinite(solution.states).all() or d["step_count"] != count
            or d["projection_kind"] != "arithmetic_mean"
            or d["coupling_frequency"] != c.coupling_frequency
            or d["output_interpolation_count"] != 0
            or np.asarray(d["copy_separation_norms"]).shape != (count,)):
        raise ValueError("Saved trajectory does not satisfy the radial study contract.")


def run_and_save_radial_cycle(prepared: PreparedRadialCycle, destination: str | Path,
                             *, options: ExecutionOptions = ExecutionOptions()) -> StoredSolution:
    """Run locally, preserving all states and diagnostics; reuse matching archives."""
    # Match JSON representation before comparing archived tuples and scalars.
    expected: Any = json.loads(json.dumps(prepared.metadata))
    if ArtifactStore(destination).exists("manifest.json"):
        saved: StoredSolution = load_solution(destination)
        actual: dict[str, Any] = dict(saved.metadata)
        old_config: dict[Any, Any] = dict(actual.get("config", {}))
        if (0 in old_config.get("source_selection", ())
                and actual.get("baseline_metadata_sha256") == expected["baseline_metadata_sha256"]
                and actual.get("source_sha256") == expected["source_sha256"]):
            # Preparation already checked the original field indices against
            # this exact baseline. Compare its direct selection without
            # rewriting the historical rank labels stored in the archive.
            old_config["source_selection"] = expected["config"]["source_selection"]
            actual["config"] = old_config
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("Destination contains a different experiment; choose a new run ID.")
        validate_radial_solution(saved.solution, prepared)
        return saved
    solution: States = simulate(prepared.problem, BM4Midpoint(),
                        prepared.request, execution=Execution(), options=options)
    validate_radial_solution(solution, prepared)
    metadata = {**expected, "execution_options": asdict(options)}
    dynamics = prepared.problem.dynamics
    if not isinstance(dynamics, GuidingCenterDynamics):
        raise TypeError("The radial study requires guiding-center dynamics.")
    potential = dynamics.potential
    save_solution(solution, destination, metadata=metadata, potential=potential)
    return StoredSolution(solution, metadata, potential)


__all__ = ["RadialCycleConfig", "PreparedRadialCycle", "build_radial_cycle",
           "prepare_radial_cycle", "validate_radial_solution", "run_and_save_radial_cycle"]
