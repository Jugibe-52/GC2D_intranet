"""Matched Poincare stars in physical or normalized space on Modal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from contracts.result import DiagnosticValue
from diagnostics.persistence import StoredSolution, load_solution, save_solution
from diagnostics.storage import ArtifactStore
from dynamics.gc import GuidingCenterDynamics
from execution.execution import Execution
from execution.execution_modal import Execution_Modal
from initial_conditions.star import ranked_radial_star
from initial_conditions import GCInitialConfiguration
from methods.classical.gauss_legendre import GaussLegendre4
from methods.classical.rk4 import RK4
from methods.extended.bm4 import BM4Implicit, BM4Midpoint
from potential import Grid, Potential
from simulation.runner import simulate
from solution import Solution
from studies.dimensional_h5_midpoint import (
    DimensionalH5Field, load_dimensional_h5_field, resolve_h5_source,
)


@dataclass(frozen=True)
class RhoStarConfig:
    """Scientific inputs; rho_hat always denotes the usual dimensionless radius."""

    rho_hat: float
    particles: int = 40
    arms: int = 8
    outer_radius_fraction: float = 0.85
    first_angle: float = 0.0
    magnetic_field: float = 1.5
    characteristic_length: float = 0.06
    source_selection: tuple[int, ...] = (0, 1)
    interpolation_order: int = 3
    cycles: int = 5000
    steps_per_cycle: int = 50
    coupling_frequency: float = 0.0
    spatial_normalization: str = "none"
    hamiltonian_convention: str = "cycle_time"
    method: str = "BM4Midpoint"
    # Shared implicit stopping controls in the runtime spatial coordinates.
    newton_absolute_tolerance: float = 1e-12
    newton_relative_tolerance: float = 1e-11
    newton_max_iterations: int = 40
    newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
    newton_jacobian_method: str = "analytic"

    def __post_init__(self) -> None:
        """Reject invalid scientific inputs before creating a remote job."""
        _validate_rho_star_modes(
            self.spatial_normalization, self.hamiltonian_convention,
            self.method, self.newton_jacobian_method,
        )
        for name in ("particles", "arms", "cycles", "steps_per_cycle", "newton_max_iterations"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.particles < 2 or self.particles % self.arms:
            raise ValueError("particles must be at least two and divisible by arms.")
        for name in ("rho_hat", "coupling_frequency"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        _validate_rho_star_extent(self.characteristic_length, self.outer_radius_fraction, self.first_angle)
        for name in ("newton_absolute_tolerance", "newton_relative_tolerance", "newton_jacobian_relative_step"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.method in ("RK4", "GaussLegendre4") and self.coupling_frequency != 0:
            raise ValueError("Classical methods require coupling_frequency=0 because they have no doubled copies.")


@dataclass(frozen=True)
class PreparedRhoStar:
    """Complete physical problem and reproducibility metadata before execution."""

    config: RhoStarConfig
    problem: InitialValueProblem
    request: SimulationRequest
    metadata: dict[str, Any]


def build_rho_star(field: DimensionalH5Field, config: RhoStarConfig, *,
                   source_sha256: str) -> PreparedRhoStar:
    """Assemble the same physical star in either space; save integer cycles only."""
    if field.raw.frequencies.size != 1 or not np.isclose(field.raw.frequencies[0], 1.0):
        raise ValueError("Select the mean and one dominant mode for cycle sampling.")
    grid = field.raw.grid
    center = np.asarray((grid.x0, grid.y0)) + grid.period / 2
    cell_radius = grid.period / 2
    radii = np.linspace(0.0, config.outer_radius_fraction * cell_radius, config.particles)
    initial = ranked_radial_star(
        center=tuple(center), outer_radius=float(radii[-1]), particles=config.particles,
        arms=config.arms, first_angle=config.first_angle,
    )
    assert initial.state is not None
    physical_xy = np.column_stack(initial.positions(initial.state))
    # rho_hat is invariant across representations; the physical radius is meters.
    rho_m = config.rho_hat * config.characteristic_length / (2 * np.pi)
    stream, runtime_rho = field.stream, rho_m
    # The historical radial study advances Phi_hat at unit forcing period.
    # In meters this is H_X=(T0/B)*Phi/(2*pi); it changes drift, not forcing time.
    hamiltonian_scale = 1.0 if config.hamiltonian_convention == "cycle_time" else 1.0 / (2 * np.pi)
    if config.hamiltonian_convention == "radial":
        stream = Potential(grid, mean=stream.mean * hamiltonian_scale,
                           modes=stream.modes * hamiltonian_scale,
                           frequencies=stream.frequencies,
                           interpolation_order=stream.interpolation_order)
    spatial_scale, spatial_origin = 1.0, np.zeros(2)
    normalized = config.spatial_normalization == "characteristic_length"
    if normalized:
        spatial_scale = 2 * np.pi / config.characteristic_length
        spatial_origin = np.asarray((grid.x0, grid.y0))
        normalized_grid = Grid(0., 0., grid.dx * spatial_scale, grid.dy * spatial_scale,
                               grid.nx, grid.ny, grid.period * spatial_scale)
        # At unchanged tau=t/T0, q=s*(X-X0) requires H_q=s^2*H_X:
        # J grad_q(H_q) = s J grad_X(H_X). Rescaling coordinates alone is wrong.
        stream = Potential(normalized_grid, mean=stream.mean * spatial_scale**2,
                           modes=stream.modes * spatial_scale**2,
                           frequencies=stream.frequencies,
                           interpolation_order=stream.interpolation_order)
        xy = (physical_xy - spatial_origin) * spatial_scale
        initial = GCInitialConfiguration.from_components(x=xy[:, 0], y=xy[:, 1])
        runtime_rho = config.rho_hat
    problem = InitialValueProblem(GuidingCenterDynamics(stream, rho=runtime_rho), initial)
    request = SimulationRequest(
        t_span=(0.0, float(config.cycles)), max_step=1.0 / config.steps_per_cycle,
        output_times=np.arange(config.cycles + 1, dtype=float),
    )
    x, y = initial.positions(problem.initial_state)
    metadata = {
        "study": ("poincare_bm4midpoint_ranked_star_rho_sweep" if config.method == "BM4Midpoint"
                  else "poincare_ranked_star_rho_sweep"),
        "config": asdict(config), "source_sha256": source_sha256,
        "source_field_indices": field.source_indices,
        "space_unit": "dimensionless" if normalized else "m",
        "spatial_normalization": config.spatial_normalization, "position_wrapping": "none",
        "spatial_scale_per_m": spatial_scale, "spatial_origin_m": spatial_origin,
        "rho_runtime": runtime_rho,
        "time_axis": "tau = t_seconds / T0", "time_unit_seconds": field.cycle_seconds,
        "dynamics_stream_function": ("(2*pi/lambda)^2 * " if normalized else "")
            + "(T0/B) * Phi" + (" / (2*pi)" if config.hamiltonian_convention == "radial" else ""),
        "hamiltonian_convention": config.hamiltonian_convention,
        "hamiltonian_scale_from_cycle_time": hamiltonian_scale,
        "rho_m": rho_m,
        "cell_bounds_m": (grid.x0, grid.y0, grid.period),
        "star_center_m": center, "cell_radius_m": cell_radius,
        "initial_radii_m": radii, "initial_radius_fraction": radii / cell_radius,
        "initial_positions_m": physical_xy,
        "initial_positions": np.column_stack((x, y)),
        "cell_bounds": (stream.grid.x0, stream.grid.y0, stream.grid.period),
        "star_center": (center - spatial_origin) * spatial_scale,
        "initial_radii": radii * spatial_scale,
        "particle_id": np.arange(1, config.particles + 1),
        "arm_id": np.arange(config.particles) % config.arms + 1,
        "particle_order": "increasing distinct distance, alternating arms; one center particle",
        "method": config.method, "samples_per_cycle": 1,
        "initial_state_saved": True, "saved_cycles": config.cycles,
        "expected_step_count": config.cycles * config.steps_per_cycle,
    }
    return PreparedRhoStar(config, problem, request, metadata)


def prepare_rho_star(source: str | Path, config: RhoStarConfig) -> PreparedRhoStar:
    """Load verified HDF5 samples in physical units without spatial resampling."""
    resolved = resolve_h5_source(source)
    with resolved.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    field = load_dimensional_h5_field(
        resolved, magnetic_field=config.magnetic_field,
        characteristic_length=config.characteristic_length,
        selectors=config.source_selection, interpolation_order=config.interpolation_order,
    )
    return build_rho_star(field, config, source_sha256=digest)


def sample_rho_dynamics(prepared: PreparedRhoStar, *, grid_size: int = 64,
                        vector_grid_size: int = 17, phase_steps: int = 50) -> dict:
    """Sample the effective Hamiltonian and GC velocity over one forcing cycle.

    Arrays use (phase, y, x) ordering, with a final xy component for velocity.
    Coordinates and velocities retain runtime units and tau=t/T0. No trajectory
    integration is performed. Both endpoint phases are evaluated and checked.
    """
    for value in (grid_size, vector_grid_size, phase_steps):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError('Field sampling sizes must be integers of at least two.')
    x0, y0, length = prepared.metadata['cell_bounds']
    phases = np.linspace(0., 1., phase_steps + 1)
    # Cell-centred grids avoid duplicate spatial endpoints in the heatmap.
    q = (np.arange(grid_size) + .5) / grid_size
    xx, yy = np.meshgrid(x0 + length*q, y0 + length*q)
    vq = (np.arange(vector_grid_size) + .5) / vector_grid_size
    vx, vy = np.meshgrid(x0 + length*vq, y0 + length*vq)
    state = np.concatenate((vx.ravel(), vy.ravel()))
    dynamics = prepared.problem.dynamics
    assert isinstance(dynamics, GuidingCenterDynamics)
    potential = np.stack([dynamics.effective_potential.evaluate(t, xx, yy) for t in phases])
    velocity = np.stack([dynamics.vector_field(t, state).reshape(2, vector_grid_size, vector_grid_size)
                         .transpose(1, 2, 0) for t in phases])
    for values in (potential, velocity):
        if not np.isfinite(values).all():
            raise ValueError('The sampled dynamics contain nonfinite values.')
        np.testing.assert_allclose(values[-1], values[0], rtol=1e-11,
                                   atol=1e-13 * max(float(np.max(np.abs(values))), 1e-30))
        values[-1] = values[0]  # Exact closure in the browser after the audit.
    return dict(phases=phases, potential=potential, velocity=velocity,
                bounds=np.array((x0, y0, length)))


def validate_rho_solution(solution: Solution, prepared: PreparedRhoStar, *,
                          options: ExecutionOptions | None = None) -> None:
    """Check complete float64 cycle samples and the selected method's contract."""
    _validate_rho_cycle_solution(solution, prepared.config,
                                output_times=prepared.request.output_times,
                                initial_state=prepared.problem.initial_state, options=options)


def _validate_rho_cycle_solution(solution: Solution, config: RhoStarConfig, *,
                                output_times: np.ndarray, initial_state: np.ndarray,
                                options: ExecutionOptions | None = None) -> None:
    """Share cycle and solver checks with explicit-seed saved-data consumers."""
    diagnostics = solution.diagnostics
    if (solution.states.shape != (2 * config.particles, config.cycles + 1)
            or not np.array_equal(solution.t, output_times)
            or not np.isfinite(solution.states).all()
            or solution.states.dtype != np.dtype("float64")
            or diagnostics["step_count"] != config.cycles * config.steps_per_cycle
            or diagnostics["output_interpolation_count"] != 0):
        raise ValueError("Result does not match the requested float64 cycle sampling.")
    if options is not None and options.backend == "jax":
        if (diagnostics.get("execution_backend") != "jax"
                or diagnostics.get("execution_device") != options.device
                or diagnostics.get("execution_mode") != "device_resident"):
            raise ValueError("Result does not match the requested device-resident JAX execution.")
    if config.method in ("BM4Midpoint", "BM4Implicit"):
        if diagnostics.get("coupling_frequency") != config.coupling_frequency:
            raise ValueError("Result does not match the requested BM4 coupling frequency.")
        if config.method == "BM4Midpoint" and diagnostics.get("projection_kind") != "arithmetic_mean":
            raise ValueError("BM4Midpoint requires arithmetic-mean projection.")
        if config.method == "BM4Implicit" and diagnostics.get("projection_solver_formulation") != "bm4_implicit_reduced":
            raise ValueError("BM4Implicit requires one reduced projection around the complete composition.")
    if config.method in ("BM4Implicit", "GaussLegendre4"):
        expected = {
            "nonlinear_solver": "newton", "nonlinear_solves_per_step": 1,
            "nonlinear_absolute_tolerance": config.newton_absolute_tolerance,
            "nonlinear_relative_tolerance": config.newton_relative_tolerance,
            "nonlinear_max_iterations": config.newton_max_iterations,
            "newton_jacobian_method": config.newton_jacobian_method,
            "newton_jacobian_relative_step": config.newton_jacobian_relative_step,
        }
        if config.method == "GaussLegendre4":
            expected.update(stage_count=2, designed_order=4)
        if any(diagnostics.get(key) != value for key, value in expected.items()):
            raise ValueError("Result does not match the requested implicit-method controls.")
    elif "nonlinear_solver" in diagnostics:
        raise ValueError("Explicit methods must not report nonlinear solves.")
    if config.method in ("RK4", "GaussLegendre4") and any(
            key in diagnostics for key in ("projection_kind", "projection_solver_formulation", "coupling_frequency")):
        raise ValueError("Classical methods must not report doubled-copy projection.")
    if not np.array_equal(solution.states[:, 0], initial_state):
        raise ValueError("Result does not match the requested initial positions.")


def _rho_star_method(config: RhoStarConfig) -> BM4Midpoint | BM4Implicit | RK4 | GaussLegendre4:
    """Bind the explicit scientific method selection to its canonical class."""
    if config.method == "BM4Midpoint":
        return BM4Midpoint(coupling_frequency=config.coupling_frequency)
    if config.method == "RK4":
        return RK4()
    if config.method == "BM4Implicit":
        return BM4Implicit(
            coupling_frequency=config.coupling_frequency,
            newton_absolute_tolerance=config.newton_absolute_tolerance,
            newton_relative_tolerance=config.newton_relative_tolerance,
            newton_max_iterations=config.newton_max_iterations,
            newton_jacobian_relative_step=config.newton_jacobian_relative_step,
            newton_jacobian_method="analytic",
        )
    return GaussLegendre4(
        newton_absolute_tolerance=config.newton_absolute_tolerance,
        newton_relative_tolerance=config.newton_relative_tolerance,
        newton_max_iterations=config.newton_max_iterations,
        newton_jacobian_relative_step=config.newton_jacobian_relative_step,
        newton_jacobian_method="analytic",
    )


def folded_rho_positions(solution: Solution, metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return folded cycle positions in runtime units and cell fractions.

    Both arrays have shape (saved cycles including zero, particles, xy).
    Folding is postprocessing only; the integrated states remain unwrapped.
    Cell fractions measure displacement from the lower corner, divided by L.
    """
    cell = np.asarray(metadata.get("cell_bounds", metadata["cell_bounds_m"]), dtype=float)
    if cell.shape != (3,) or not np.isfinite(cell).all() or cell[2] <= 0:
        raise ValueError("Cell bounds must describe a finite positive square.")
    x, y = solution.positions()
    xy = np.stack((x.T, y.T), axis=-1)
    if not np.isfinite(xy).all():
        raise ValueError("Cycle positions must be finite before periodic folding.")
    offsets = (xy - cell[:2]) % cell[2]
    return offsets + cell[:2], offsets / cell[2]


def run_rho_star(prepared: PreparedRhoStar, *, executor: Execution,
                 options: ExecutionOptions, save_folded_returns: bool = False) -> StoredSolution:
    """Integrate once and retain cycle states, optionally with folded copies."""
    started = perf_counter()
    solution = simulate(
        prepared.problem, _rho_star_method(prepared.config),
        prepared.request, execution=executor, options=options,
    )
    validate_rho_solution(solution, prepared, options=options)
    # Per-step arrays are internal diagnostics, not requested observations.
    # Drop internal arrays before adding any requested cycle-only representations.
    diagnostics: dict[str, DiagnosticValue] = {
        key: value for key, value in solution.diagnostics.items()
        if not isinstance(value, np.ndarray)
    }
    if save_folded_returns:
        wrapped, fractions = folded_rho_positions(solution, prepared.metadata)
        diagnostics.update(cycle_positions_wrapped=wrapped, cycle_positions_cell_fraction=fractions)
    solution = Solution(t=solution.t, states=solution.states,
                        source=solution.source, diagnostics=diagnostics)
    metadata = {**prepared.metadata, "execution_options": asdict(options),
                "client_wall_seconds": perf_counter() - started,
                "execution_executor": "modal" if isinstance(executor, Execution_Modal) else "local"}
    if save_folded_returns:
        metadata.update(
            saved_folded_returns=True,
            folded_return_layout="(saved cycle including zero, particle, xy)",
            folded_return_units=prepared.metadata["space_unit"],
            cell_fraction_definition="((position - cell_origin) modulo cell_width) / cell_width",
        )
    if isinstance(executor, Execution_Modal):
        metadata["modal_call_id"] = executor.last_call_id
        metadata["modal_receipt"] = str(executor.last_record_path)
    return StoredSolution(solution, metadata, None)


def run_and_save_rho_star(prepared: PreparedRhoStar, destination: str | Path, *,
                          executor: Execution, options: ExecutionOptions,
                          save_folded_returns: bool = False) -> StoredSolution:
    """Reuse a matching archive or run once, optionally saving folded returns."""
    if ArtifactStore(destination).exists("manifest.json"):
        stored = load_solution(destination)
        # JSON arrays replace config tuples when reading a saved archive.
        expected = dict(asdict(prepared.config))
        expected["source_selection"] = list(expected["source_selection"])
        actual = dict(stored.metadata.get("config", {}))
        actual.setdefault("spatial_normalization", "none")
        actual.setdefault("hamiltonian_convention", "cycle_time")
        defaults = asdict(RhoStarConfig(rho_hat=0.0))
        for key in ("method", "newton_absolute_tolerance", "newton_relative_tolerance",
                    "newton_max_iterations", "newton_jacobian_relative_step", "newton_jacobian_method"):
            actual.setdefault(key, defaults[key])
        if (actual != expected
                or stored.metadata.get("source_sha256") != prepared.metadata["source_sha256"]
                or stored.metadata.get("method", "BM4Midpoint") != prepared.config.method
                or stored.metadata.get("probe_config") != prepared.metadata.get("probe_config")):
            raise ValueError("The completed destination belongs to different scientific inputs.")
        validate_rho_solution(stored.solution, prepared, options=options)
        if save_folded_returns:
            wrapped, fractions = folded_rho_positions(stored.solution, stored.metadata)
            for name, expected_array in (("cycle_positions_wrapped", wrapped),
                                         ("cycle_positions_cell_fraction", fractions)):
                actual_array = stored.solution.diagnostics.get(name)
                if not isinstance(actual_array, np.ndarray) or not np.array_equal(actual_array, expected_array):
                    raise ValueError("The completed archive has missing or inconsistent folded returns.")
        return stored
    stored = run_rho_star(prepared, executor=executor, options=options,
                          save_folded_returns=save_folded_returns)
    save_solution(stored.solution, destination, metadata=stored.metadata)
    return stored


def modal_rho_executor(record_directory: str | Path, *,
                       app_name: str = "gc2d-poincare-rho-sweep",
                       resume_record: str | Path | None = None) -> Execution_Modal:
    """Resume an existing confirmed job instead of duplicating a long calculation."""
    directory = Path(record_directory)
    if resume_record is not None:
        return Execution_Modal.from_record(resume_record)
    records = [(path, json.loads(path.read_text())) for path in sorted(directory.glob('*.json'))]
    if records:
        call_ids = {record.get('call_id') for _, record in records}
        if None in call_ids or len(call_ids) != 1:
            raise ValueError('Inspect existing Modal receipts and select resume_record explicitly before continuing.')
        path, record = records[-1]
        if record['app_name'] != app_name or record['function_name'] != 'integrate':
            raise ValueError('The existing receipt belongs to another Modal worker.')
        return Execution_Modal.from_record(path)
    return Execution_Modal(app_name=app_name, function_name='integrate', record_directory=directory)


def _validate_rho_star_modes(
	spatial_normalization: str, hamiltonian_convention: str, method: str, jacobian_method: str,
) -> None:
	"""Require the supported coordinate, Hamiltonian, integrator, and Jacobian conventions."""
	if spatial_normalization not in ("none", "characteristic_length"):
		raise ValueError("spatial_normalization must be none or characteristic_length.")
	if hamiltonian_convention not in ("cycle_time", "radial"):
		raise ValueError("hamiltonian_convention must be cycle_time or radial.")
	if method not in ("BM4Midpoint", "BM4Implicit", "RK4", "GaussLegendre4"):
		raise ValueError("method must be BM4Midpoint, BM4Implicit, RK4 or GaussLegendre4.")
	if jacobian_method != "analytic":
		raise ValueError("Matched Poincare stars require analytic guiding-center Jacobians.")


def _validate_rho_star_extent(characteristic_length: float, outer_radius_fraction: float, first_angle: float) -> None:
	"""Require finite physical length and the existing radial extent and angle limits."""
	if not np.isfinite(characteristic_length) or characteristic_length <= 0:
		raise ValueError("characteristic_length must be finite and positive.")
	if not np.isfinite(outer_radius_fraction) or not 0 < outer_radius_fraction <= 1:
		raise ValueError("outer_radius_fraction must lie in (0, 1].")
	if not np.isfinite(first_angle):
		raise ValueError("first_angle must be finite.")


__all__ = ["RhoStarConfig", "PreparedRhoStar", "build_rho_star", "prepare_rho_star",
           "sample_rho_dynamics", "validate_rho_solution", "folded_rho_positions", "run_rho_star",
           "run_and_save_rho_star", "modal_rho_executor"]
