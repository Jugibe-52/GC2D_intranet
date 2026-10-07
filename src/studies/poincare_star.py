"""BM4Midpoint Poincare calculation with dimensional HDF5 star positions."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from dynamics.gc import GuidingCenterDynamics
from initial_conditions.star import radial_star
from methods.extended.bm4 import BM4Midpoint
from potential.load import DEFAULT_FIELD_INDICES
from potential.potential import Potential
from simulation.runner import simulate
from solution import Solution
from studies.dimensional_h5_midpoint import load_dimensional_h5_field, resolve_h5_source


@dataclass(frozen=True)
class PoincareStarConfig:
    """Scientific inputs; rho_hat uses the established lambda/(2*pi) scale.

    Center fractions locate the star within the original HDF5 periodic cell.
    Time is tau=t/T0; stored particle positions and rho_m are in meters.
    """

    magnetic_field: float
    characteristic_length: float
    selectors: tuple[int, ...] = field(default=DEFAULT_FIELD_INDICES, kw_only=True)
    interpolation_order: int
    rho_hat: float
    arms: int
    particles_per_arm: int
    arm_length_fraction: float
    center_fraction: tuple[float, float]
    first_angle: float
    cycles: int
    steps_per_cycle: int
    coupling_frequency: float


@dataclass(frozen=True)
class PoincareStarResult:
    """Physical solution, original unaveraged potential and reproducibility data."""

    solution: Solution
    potential: Potential
    metadata: dict[str, Any]


def run_poincare_star(source: str | Path, config: PoincareStarConfig, *,
                      execution: ExecutionOptions) -> PoincareStarResult:
    """Integrate every particle together, saving the initial state and each cycle.

    The dimensional stream function is (T0/B)*Phi, so the GC drift has units
    meters per normalized cycle. No output wrapping or spatial normalization
    is applied. Field construction is excluded from integration_seconds;
    device setup, any JIT compilation and final synchronization are included.
    """
    fraction = _validated_star_center(config)
    resolved = resolve_h5_source(source)
    field = load_dimensional_h5_field(
        resolved, magnetic_field=config.magnetic_field,
        characteristic_length=config.characteristic_length,
        selectors=config.selectors, interpolation_order=config.interpolation_order,
    )
    # Each saved integer time is one forcing cycle, for the selected variable field.
    if field.raw.frequencies.size != 1 or not np.isclose(field.raw.frequencies[0], 1.0):
        raise ValueError("This stroboscopic study requires one variable field at normalized frequency 1.")
    grid = field.raw.grid
    center = np.array([grid.x0, grid.y0]) + grid.period * fraction
    length = config.arm_length_fraction * grid.period
    rho_m = config.rho_hat * config.characteristic_length / (2 * np.pi)
    initial = radial_star(center=(float(center[0]), float(center[1])), arm_length=length,
                          arms=config.arms, particles_per_arm=config.particles_per_arm,
                          first_angle=config.first_angle)
    problem = InitialValueProblem(GuidingCenterDynamics(field.stream, rho=rho_m), initial)
    request = SimulationRequest(
        t_span=(0.0, float(config.cycles)), max_step=1.0 / config.steps_per_cycle,
        output_times=np.arange(config.cycles + 1, dtype=float),
    )
    method = BM4Midpoint(coupling_frequency=config.coupling_frequency,
                         track_energy=False, progress=False, step_observer=None)
    start = perf_counter()
    solution = simulate(problem, method, request, options=execution)
    elapsed = perf_counter() - start
    particles = config.arms * config.particles_per_arm
    if (solution.states.shape != (2 * particles, config.cycles + 1)
            or solution.diagnostics["step_count"] != config.cycles * config.steps_per_cycle
            or solution.diagnostics["output_interpolation_count"] != 0
            or solution.diagnostics["projection_kind"] != "arithmetic_mean"):
        raise RuntimeError("The calculation did not preserve the configured BM4 cycle sampling.")
    x, y = initial.positions(problem.initial_state)
    metadata = {
        "study": "poincare_star_bm4_midpoint", "config": asdict(config),
        "source": str(Path(source).resolve()), "source_sha256": sha256(resolved.read_bytes()).hexdigest(),
        "source_field_indices": field.source_indices,
        "source_grid_period_m": grid.period, "space_unit": "m",
        "spatial_normalization": "none", "position_wrapping": "none",
        "time_axis": "tau = t_seconds / T0", "time_unit_seconds": field.cycle_seconds,
        "saved_time_seconds": solution.t * field.cycle_seconds,
        "stored_potential": "Original HDF5 Phi in volts, before gyroaveraging",
        "dynamics_stream_function": "(T0/B) * Phi", "rho_m": rho_m,
        "star_center_m": center, "arm_length_m": length,
        "radial_spacing_m": length / config.particles_per_arm,
        "angular_spacing_rad": 2 * np.pi / config.arms,
        "particle_order": "arm-major, then increasing radius; center excluded",
        "particle_id": np.arange(1, particles + 1),
        "arm_id": np.repeat(np.arange(config.arms), config.particles_per_arm),
        "radial_index": np.tile(np.arange(1, config.particles_per_arm + 1), config.arms),
        "initial_positions_m": np.column_stack((x, y)),
        "execution": asdict(execution), "integration_seconds": elapsed,
        "timing_scope": "Single complete integration, including any compilation and transfers",
        "track_energy": False, "samples_per_cycle": 1, "initial_state_saved": True,
    }
    return PoincareStarResult(solution, field.raw, metadata)


def _validated_star_center(config: PoincareStarConfig) -> np.ndarray:
	"""Check cycle counts and the full stellar extent before loading the physical field."""
	for name in ("cycles", "steps_per_cycle"):
		value = getattr(config, name)
		if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
			raise ValueError(f"{name} must be a positive integer.")
	fraction = np.asarray(config.center_fraction, dtype=float)
	if fraction.shape != (2,) or not np.all(np.isfinite(fraction)):
		raise ValueError("center_fraction must contain two finite coordinates.")
	if (not np.isfinite(config.arm_length_fraction) or config.arm_length_fraction <= 0
			or np.any(fraction - config.arm_length_fraction < 0)
			or np.any(fraction + config.arm_length_fraction > 1)):
		raise ValueError("The complete star must fit inside the source periodic cell.")
	return fraction


__all__ = ["PoincareStarConfig", "PoincareStarResult", "run_poincare_star"]
