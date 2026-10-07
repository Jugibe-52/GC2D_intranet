"""Local BM4Midpoint star study with consecutive JAX blocks and live logging."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from contracts.result import DiagnosticValue
from diagnostics.persistence import save_solution
from diagnostics.run_progress import RunProgress, progress_log
from dynamics.gc import GuidingCenterDynamics
from execution.execution import Execution
from initial_conditions.gc import GCInitialConfiguration
from initial_conditions.star import radial_star
from methods.extended.bm4 import BM4Midpoint
from potential.potential import Potential
from simulation.runner import simulate
from solution import Solution
from studies.h5_provenance import prepare_verified_h5_field
from studies.reference_trajectory import potential_fingerprint


@dataclass(frozen=True)
class LocalJAXStarConfig:
    """Explicit physical parameters, positive arm particles and progress blocks.

    include_center adds one shared particle at radius zero. Each arm contains
    particles_per_arm positive radii spaced uniformly up to outer_radius_fraction
    of R=radius_over_period*L. All saved coordinates use normalized spatial units.
    """

    magnetic_field: float
    characteristic_length: float
    source_selection: tuple[int, ...]
    interpolation_order: int
    rho: float
    coupling_frequency: float
    arms: int
    particles_per_arm: int
    include_center: bool
    radius_over_period: float
    outer_radius_fraction: float
    center_fraction: tuple[float, float]
    first_angle: float
    cycles: int
    steps_per_cycle: int
    block_cycles: int


@dataclass(frozen=True)
class LocalJAXStarResult:
    """Every saved step, the unaveraged field and reproducibility metadata."""

    solution: Solution
    potential: Potential
    metadata: dict[str, Any]


def build_local_star(potential: Potential, config: LocalJAXStarConfig) -> InitialValueProblem:
    """Validate the cycle schedule and assemble one central/shared-arm geometry."""
    fraction, outer = _validated_local_star_geometry(potential, config)
    origin = np.array([potential.grid.x0, potential.grid.y0]) + fraction * potential.grid.period
    initial = radial_star(center=(float(origin[0]), float(origin[1])),
        arm_length=outer * potential.grid.period, arms=config.arms,
        particles_per_arm=config.particles_per_arm, first_angle=config.first_angle,
        include_center=config.include_center)
    return InitialValueProblem(GuidingCenterDynamics(potential, rho=config.rho), initial)


def integrate_local_star(
    potential: Potential, config: LocalJAXStarConfig, *, progress: RunProgress,
) -> LocalJAXStarResult:
    """Advance consecutive cycle blocks and retain all effective step states.

    BM4Midpoint projects both spatial copies onto their arithmetic mean after
    every complete step, so its final physical state is sufficient to continue.
    Energy tracking is off. Blocks use absolute time and the same physical field;
    they neither reset forcing phase nor carry a hidden nonlinear solver state.
    The same dynamics object preserves compiled JAX bindings across blocks.
    """
    problem = build_local_star(potential, config)
    total_steps = config.cycles * config.steps_per_cycle
    # Component-major history: (2*N, total_steps+1), float64. Preallocation avoids
    # retaining a second list of all completed blocks before concatenation.
    times = np.arange(total_steps + 1, dtype=float) / config.steps_per_cycle
    states = np.empty((problem.initial_state.size, times.size), dtype=np.float64)
    states[:, 0] = problem.initial_state
    step_arrays: dict[str, np.ndarray] = {}
    diagnostics: dict[str, DiagnosticValue] = {}
    block_seconds, block_ends = [], []
    method = BM4Midpoint(coupling_frequency=config.coupling_frequency, track_energy=False,
                         progress=False, step_observer=None)
    executor, options = Execution(), ExecutionOptions(backend='jax', device='cpu')
    started = perf_counter()
    for first_cycle in range(0, config.cycles, config.block_cycles):
        last_cycle = min(first_cycle + config.block_cycles, config.cycles)
        first, last = first_cycle * config.steps_per_cycle, last_cycle * config.steps_per_cycle
        phase = f'Integrating cycles {first_cycle}–{last_cycle}'
        if first == 0:
            phase += ' (first JAX compilation included)'
        progress.phase(phase)
        block_problem = InitialValueProblem(problem.dynamics, GCInitialConfiguration(states[:, first]))
        request = SimulationRequest((float(first_cycle), float(last_cycle)),
                                    1. / config.steps_per_cycle, times[first:last + 1])
        start = perf_counter()
        block = simulate(block_problem, method, request, execution=executor, options=options)
        seconds = perf_counter() - start  # NumPy output has synchronized JAX.
        data = block.diagnostics
        if (data['step_count'] != last - first or data['output_interpolation_count'] != 0
                or data['projection_kind'] != 'arithmetic_mean' or data['nonlinear_unknown_dimension'] != 0):
            raise RuntimeError('The block does not match the BM4Midpoint step and sampling contract.')
        states[:, first + 1:last + 1] = block.states[:, 1:]
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 0 or value.shape[0] != last - first:
                    raise RuntimeError(f'Unexpected non-step diagnostic array: {key}.')
                if first == 0:
                    step_arrays[key] = np.empty((total_steps, *value.shape[1:]), dtype=value.dtype)
                step_arrays[key][first:last] = value
            elif first == 0:
                diagnostics[key] = value
        block_seconds.append(seconds)
        block_ends.append(last_cycle)
        progress.phase(f'Block complete ({last - first} steps in {seconds:.3f}s)', completed=last_cycle)
    elapsed = perf_counter() - started
    diagnostics.update(step_arrays)
    diagnostics.update(step_count=total_steps, output_interpolation_count=0,
                       execution_block_cycles=config.block_cycles,
                       execution_block_end_cycles=np.asarray(block_ends),
                       execution_block_wall_seconds=np.asarray(block_seconds))
    solution = Solution(t=times, states=states, source=problem.initial_configuration, diagnostics=diagnostics)
    x, y = problem.layout.positions(problem.initial_state)
    arms = np.repeat(np.arange(config.arms), config.particles_per_arm)
    if config.include_center:
        arms = np.r_[-1, arms]  # -1 denotes the unique shared central particle.
    assert isinstance(problem.dynamics, GuidingCenterDynamics)
    metadata = {
        'study': 'local_jax_bm4midpoint_star', 'config': asdict(config),
        'particle_ids': np.arange(1, problem.particle_count + 1), 'arm_ids': arms,
        'initial_positions': np.column_stack((x, y)),
        'particle_order': 'Shared center first when enabled, then arm-major positive radii',
        'positive_radii_over_R': config.outer_radius_fraction * np.arange(1, config.particles_per_arm + 1) / config.particles_per_arm,
        'reference_radius': config.radius_over_period * potential.grid.period,
        'effective_field_fingerprint': potential_fingerprint(problem.dynamics.effective_potential),
        'arithmetic': 'float64', 'execution': 'local_jax_cpu', 'method': 'BM4Midpoint',
        'integration_seconds': elapsed, 'timing_scope': 'All blocks, including first JIT compilation and logging',
        'samples_per_cycle': config.steps_per_cycle, 'accuracy_certified': False,
        'track_energy': False, 'newton_solves': 0,
    }
    return LocalJAXStarResult(solution, potential, metadata)


def calculate_local_star(
    source: str | Path, config: LocalJAXStarConfig, *, expected_source_sha256: str,
    original_provenance: str | Path, destination: str, log_path: str | Path,
    heartbeat_seconds: float = 10.,
) -> LocalJAXStarResult:
    """Prepare, compute locally and publish; log and propagate every failure."""
    with progress_log(log_path, total=config.cycles, heartbeat_seconds=heartbeat_seconds) as progress:
        progress.phase('Verifying source checksum and loading the HDF5 field')
        potential, provenance = prepare_verified_h5_field(
            source, magnetic_field=config.magnetic_field, characteristic_length=config.characteristic_length,
            source_selection=config.source_selection, interpolation_order=config.interpolation_order,
            expected_source_sha256=expected_source_sha256, original_provenance=original_provenance)
        import jax
        progress.logger.info('Local CPU | JAX=%s | x64=%s | destination=%s',
                             jax.__version__, jax.config.read('jax_enable_x64'), destination)
        result = integrate_local_star(potential, config, progress=progress)
        result.metadata.update(provenance, log_path=str(Path(log_path).resolve()))
        progress.phase('Integration complete; serializing and publishing the archive')
        save_solution(result.solution, destination, metadata=result.metadata, potential=potential)
        progress.phase(f'COMPLETED AND SAVED | destination={destination}')
        return result


def _validated_local_star_geometry(
	potential: Potential, config: LocalJAXStarConfig,
) -> tuple[np.ndarray, float]:
	"""Require a valid cycle schedule and a full star within the periodic cell."""
	for name in ('arms', 'particles_per_arm', 'cycles', 'steps_per_cycle', 'block_cycles'):
		value = getattr(config, name)
		if isinstance(value, bool) or not isinstance(value, int) or value < 1:
			raise ValueError(f'{name} must be a positive integer.')
	if not (0 < config.radius_over_period <= .5 and 0 < config.outer_radius_fraction <= 1):
		raise ValueError('Require 0 < R/L <= 0.5 and 0 < outer_radius_fraction <= 1.')
	fraction = np.asarray(config.center_fraction)
	outer = config.radius_over_period * config.outer_radius_fraction
	if (fraction.shape != (2,) or not np.isfinite(fraction).all()
			or np.any(fraction - outer < 0) or np.any(fraction + outer > 1)):
		raise ValueError('The full initial star must lie inside the periodic cell.')
	if potential.frequencies.shape != (1,) or not np.allclose(potential.frequencies, [1.], rtol=0, atol=1e-14):
		raise ValueError('Integer-cycle sections require one mode with normalized frequency one.')
	return fraction, outer


__all__ = ['LocalJAXStarConfig', 'LocalJAXStarResult', 'build_local_star',
           'integrate_local_star', 'calculate_local_star']
